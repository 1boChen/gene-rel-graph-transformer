# %%
# %%
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %%
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from torch_geometric.utils import from_networkx
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import Optional

import optuna

# ============================================================
# 0) Repro (optional)
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1) Labels: subtype -> multi-hot vector
# ============================================================
relation_types = [
    "activation",
    "compound",
    "inhibition",
    "binding/association",
    "expression",
    "phosphorylation",
    "dephosphorylation",
    "state change",
    "ubiquitination",
    "repression",
    "dissociation",
]
NUM_EDGE_TYPES = len(relation_types)
relation_to_index = {r: i for i, r in enumerate(relation_types)}

def subtype_to_vector(subtype_name: str):
    if subtype_name == "no_interaction":
        return [0] * NUM_EDGE_TYPES
    vec = [0] * NUM_EDGE_TYPES
    for subtype in str(subtype_name).split(",, "):
        if subtype in relation_to_index:
            vec[relation_to_index[subtype]] = 1
    return vec

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relation_vector"] = df["subtype_name"].apply(subtype_to_vector)
    return df

# ============================================================
# 2) Load CSVs + add edge_id for stable join-back export
# ============================================================
TRAIN_CSV = "../../data/GNN/multilabel_train.csv"
VAL_CSV   = "../../data/GNN/multilabel_val.csv"
TEST_CSV  = "../../data/GNN/multilabel_test.csv"

train_df_raw = pd.read_csv(TRAIN_CSV)
val_df_raw   = pd.read_csv(VAL_CSV)
test_df_raw  = pd.read_csv(TEST_CSV)

for _df in (train_df_raw, val_df_raw, test_df_raw):
    _df["edge_id"] = _df.index.astype(int)

train_df = process_dataset(train_df_raw)
val_df   = process_dataset(val_df_raw)
test_df  = process_dataset(test_df_raw)

# ============================================================
# 3) Pathway source mapping (fit on train)
# ============================================================
unique_pathway_sources = train_df["pathway_source"].unique()
pathway_source_to_int = {p: i for i, p in enumerate(unique_pathway_sources)}
int_to_pathway_source = {v: k for k, v in pathway_source_to_int.items()}

def convert_pathway_source_to_int(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["pathway_source"] = df["pathway_source"].map(mapping).fillna(-1).astype(int)  # unseen -> -1
    return df

train_df = convert_pathway_source_to_int(train_df, pathway_source_to_int)
val_df   = convert_pathway_source_to_int(val_df, pathway_source_to_int)
test_df  = convert_pathway_source_to_int(test_df, pathway_source_to_int)

# ============================================================
# 4) DataFrame -> NetworkX MultiDiGraph (carry edge_id)
# ============================================================
def df_to_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        u = row["starter_ID"]
        v = row["receiver_ID"]
        G.add_node(u, name=u)
        G.add_node(v, name=v)
        G.add_edge(
            u,
            v,
            interaction_type=row["relation_vector"],
            pathway_source=int(row["pathway_source"]),
            edge_id=int(row["edge_id"]),
        )
    return G

train_graph = df_to_graph(train_df)
val_graph   = df_to_graph(val_df)
test_graph  = df_to_graph(test_df)

# ============================================================
# 5) Convert to PyG Data + set edge_attr/edge_label
# ============================================================
def setup_edge(MDG: nx.MultiDiGraph, data):
    pathway_sources = []
    edge_labels_list = []
    for _, _, edge_data in MDG.edges(data=True):
        pathway_sources.append(int(edge_data["pathway_source"]))
        edge_labels_list.append(torch.tensor(edge_data["interaction_type"], dtype=torch.float))
    data.edge_attr  = torch.tensor(pathway_sources, dtype=torch.long).unsqueeze(1)  # [E,1]
    data.edge_label = torch.stack(edge_labels_list)                                 # [E,C]

train_data = from_networkx(train_graph)
val_data   = from_networkx(val_graph)
test_data  = from_networkx(test_graph)

setup_edge(train_graph, train_data)
setup_edge(val_graph, val_data)
setup_edge(test_graph, test_data)

# ============================================================
# 6) Entity id mappings + embedding loader
# ============================================================
def create_entity_id_to_local_index_mapping(graph: nx.MultiDiGraph):
    return {node: idx for idx, node in enumerate(graph.nodes())}

def create_entity_id_to_train_index_mapping(train_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph):
    mapping = {}
    train_nodes = list(train_graph.nodes())
    train_pos = {n: i for i, n in enumerate(train_nodes)}
    unseen_index = len(train_nodes)
    for node in graph.nodes():
        mapping[node] = train_pos.get(node, unseen_index)
    return mapping

train_entity_id_to_local_index = create_entity_id_to_local_index_mapping(train_graph)
val_entity_id_to_local_index   = create_entity_id_to_local_index_mapping(val_graph)
test_entity_id_to_local_index  = create_entity_id_to_local_index_mapping(test_graph)

train_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, train_graph)
val_entity_id_to_train_index   = create_entity_id_to_train_index_mapping(train_graph, val_graph)
test_entity_id_to_train_index  = create_entity_id_to_train_index_mapping(train_graph, test_graph)

def load_dna_embeddings(embedding_dim: int, csv_file: str, entity_id_to_local_index: dict):
    df = pd.read_csv(csv_file)
    all_embeddings = np.zeros((len(entity_id_to_local_index), embedding_dim), dtype=float)
    for _, row in df.iterrows():
        entity_id = row["entity_ID"]
        if entity_id in entity_id_to_local_index:
            local_index = entity_id_to_local_index[entity_id]
            emb = np.array([float(x) for x in str(row["embedding"]).split(",")], dtype=float)
            all_embeddings[local_index] = emb
    scaler = StandardScaler()
    normalized = scaler.fit_transform(all_embeddings)
    return torch.tensor(normalized, dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 7) Dataset wrapper + loaders
# ============================================================
class CustomGraphDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def len(self):
        return 1
    def get(self, idx):
        return self.data

train_loader = DataLoader(CustomGraphDataset(train_data), batch_size=1, shuffle=False)
val_loader   = DataLoader(CustomGraphDataset(val_data),   batch_size=1, shuffle=False)
test_loader  = DataLoader(CustomGraphDataset(test_data),  batch_size=1, shuffle=False)

# ============================================================
# 8) Model definitions (parametrized for Optuna)
# ============================================================
EMBEDDING_DIM = 768

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries= nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]
        v = value.reshape(N, self.num_heads, -1)
        k = key.reshape(N, self.num_heads, -1)
        q = query.reshape(N, self.num_heads, -1)

        v = self.values(v)
        k = self.keys(k)
        q = self.queries(q)

        scores = (q * k).sum(dim=-1) / (self.embed_size ** 0.5)  # [N,H]
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)       # [N,H,1]
        out = (v * attn).reshape(N, self.embed_size)
        return self.fc_out(out)

class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_genes: int,
        gene_embedding_dim: int,
        projected_dim_1: int,
        projected_dim_2: int,
        projected_dim_3: int,
        num_heads: int,
        pathway_embedding_dim: int,
        dropout: float,
        embedding_dim: int = EMBEDDING_DIM,
        num_pathway_sources: Optional[int] = None,
    ):
        super().__init__()
        if num_pathway_sources is None:
            num_pathway_sources = len(pathway_source_to_int) + 1  # +1 bucket for unseen (-1)

        num_hidden_units = projected_dim_1 + projected_dim_2 + projected_dim_3 + gene_embedding_dim

        self.gene_embedding = nn.Embedding(num_genes, gene_embedding_dim)
        self.pathway_embedding = nn.Embedding(num_pathway_sources, pathway_embedding_dim)

        self.dna_projection_1 = nn.Linear(embedding_dim, projected_dim_1)
        self.dna_projection_2 = nn.Linear(embedding_dim, projected_dim_2)
        self.dna_projection_3 = nn.Linear(2560, projected_dim_3)

        self.mha = MultiHeadAttention(embed_size=projected_dim_1 + projected_dim_2 + projected_dim_3, num_heads=4)

        self.conv1 = TransformerConv(
            num_hidden_units, num_hidden_units, heads=num_heads,
            dropout=dropout, edge_dim=pathway_embedding_dim, concat=False
        )
        self.conv2 = TransformerConv(
            num_hidden_units, num_hidden_units, heads=1,
            dropout=dropout, edge_dim=pathway_embedding_dim
        )

        self.norm1 = nn.LayerNorm(num_hidden_units)
        self.norm2 = nn.LayerNorm(num_hidden_units)

        self.beta_layer_1 = nn.Linear(num_hidden_units * 2, 1)
        self.beta_layer_2 = nn.Linear(num_hidden_units * 2, 1)

        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def forward(self, data, combined_embeddings, gene_node_indices, dna_node_indices):
        max_index = self.gene_embedding.num_embeddings - 1
        gene_emb = self.gene_embedding(torch.clamp(gene_node_indices, min=0, max=max_index))
        gene_emb = F.normalize(gene_emb, p=2, dim=1)

        pathway_ids = data.edge_attr.squeeze(-1)
        pathway_ids = torch.where(
            pathway_ids < 0,
            torch.tensor(self.pathway_embedding.num_embeddings - 1, device=pathway_ids.device),
            pathway_ids,
        )
        pathway_emb_edges = self.pathway_embedding(pathway_ids)

        ed = self.embedding_dim
        dna_1 = combined_embeddings[dna_node_indices, :ed]
        dna_2 = combined_embeddings[dna_node_indices, ed:2*ed]
        dna_3 = combined_embeddings[dna_node_indices, 2*ed:]

        p1 = self.dna_projection_1(dna_1)
        p2 = self.dna_projection_2(dna_2)
        p3 = self.dna_projection_3(dna_3)

        dna_cat = torch.cat((p1, p2, p3), dim=1)
        dna_cat = self.mha(dna_cat, dna_cat, dna_cat)

        x_in = torch.cat((dna_cat, gene_emb), dim=1)

        x = F.dropout(x_in, p=self.dropout, training=self.training)
        x1 = self.conv1(x, data.edge_index, edge_attr=pathway_emb_edges)
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)

        beta1 = torch.sigmoid(self.beta_layer_1(torch.cat((x_in, x1), dim=1)))
        x = beta1 * x_in + (1.0 - beta1) * x1

        x = F.dropout(x, p=self.dropout, training=self.training)
        x2 = self.conv2(x, data.edge_index, edge_attr=pathway_emb_edges)
        x2 = self.norm2(x2)
        x2 = F.gelu(x2)

        beta2 = torch.sigmoid(self.beta_layer_2(torch.cat((x, x2), dim=1)))
        x_out = beta2 * x + (1.0 - beta2) * x2

        return x_out, self.pathway_embedding.weight

class EdgeTypeClassifier(nn.Module):
    def __init__(
        self,
        gene_embedding_dim: int,
        projected_dim_1: int,
        projected_dim_2: int,
        projected_dim_3: int,
        pathway_embedding_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_edge_types: int = NUM_EDGE_TYPES,
    ):
        super().__init__()
        num_hidden_units = projected_dim_1 + projected_dim_2 + projected_dim_3 + gene_embedding_dim
        in_dim = (num_hidden_units * 2) + pathway_embedding_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, num_edge_types)]  # logits
        self.layers = nn.Sequential(*layers)

    def forward(self, u, v, p):
        return self.layers(torch.cat((u, v, p), dim=1))

# ============================================================
# 9) Embeddings helper (cached per split so Optuna doesn’t reload every epoch)
# ============================================================
GENE_EMB_CSV    = "../../data/GNN/gene_embeddings.csv"
BIOBERT_EMB_CSV = "../../data/GNN/gene_biobert_embeddings.csv"
ESM2_EMB_CSV    = "../../data/GNN/esm2_embeddings.csv"

def make_combined_embeddings(entity_id_to_local_index: dict):
    dna     = load_dna_embeddings(EMBEDDING_DIM, GENE_EMB_CSV, entity_id_to_local_index).to(device)
    biobert = load_dna_embeddings(EMBEDDING_DIM, BIOBERT_EMB_CSV, entity_id_to_local_index).to(device)
    prot    = load_dna_embeddings(2560,          ESM2_EMB_CSV, entity_id_to_local_index).to(device)
    return torch.cat((dna, biobert, prot), dim=1)

# Cache (big speedup during Optuna)
TRAIN_COMBINED = make_combined_embeddings(train_entity_id_to_local_index)
VAL_COMBINED   = make_combined_embeddings(val_entity_id_to_local_index)
TEST_COMBINED  = make_combined_embeddings(test_entity_id_to_local_index)

TRAIN_DNA_NODE_INDICES = torch.tensor(list(train_entity_id_to_local_index.values()), dtype=torch.long, device=device)
VAL_DNA_NODE_INDICES   = torch.tensor(list(val_entity_id_to_local_index.values()),   dtype=torch.long, device=device)
TEST_DNA_NODE_INDICES  = torch.tensor(list(test_entity_id_to_local_index.values()),  dtype=torch.long, device=device)

# Gene-node indices (train space) for each split
TRAIN_GENE_NODE_INDICES = torch.tensor(list(train_entity_id_to_local_index.values()), dtype=torch.long, device=device)
VAL_GENE_NODE_INDICES   = torch.tensor(
    [val_entity_id_to_train_index.get(node, -1) for node in val_entity_id_to_local_index.keys()],
    dtype=torch.long, device=device
)
TEST_GENE_NODE_INDICES  = torch.tensor(
    [test_entity_id_to_train_index.get(node, -1) for node in test_entity_id_to_local_index.keys()],
    dtype=torch.long, device=device
)

# ============================================================
# 10) Metrics utilities
# ============================================================
def compute_metrics_for_each_relation(preds01: np.ndarray, labels01: np.ndarray):
    metrics = {}
    for i, rel in enumerate(relation_types):
        metrics[rel] = {
            "precision": precision_score(labels01[:, i], preds01[:, i], zero_division=0),
            "recall":    recall_score(labels01[:, i], preds01[:, i], zero_division=0),
            "f1":        f1_score(labels01[:, i], preds01[:, i], zero_division=0),
        }
    metrics["Micro Average"] = {
        "precision": precision_score(labels01, preds01, average="micro", zero_division=0),
        "recall":    recall_score(labels01, preds01, average="micro", zero_division=0),
        "f1":        f1_score(labels01, preds01, average="micro", zero_division=0),
    }
    metrics["Macro Average"] = {
        "precision": precision_score(labels01, preds01, average="macro", zero_division=0),
        "recall":    recall_score(labels01, preds01, average="macro", zero_division=0),
        "f1":        f1_score(labels01, preds01, average="macro", zero_division=0),
    }
    return metrics

def print_metrics_table(metrics: dict):
    print("{:<22} {:<10} {:<10} {:<10}".format("Relation", "Precision", "Recall", "F1"))
    for rel, s in metrics.items():
        print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f}".format(rel, s["precision"], s["recall"], s["f1"]))

# ============================================================
# 11) Train / Eval functions for Optuna (no gradient in eval)
# ============================================================
def build_loss_pos_weight():
    all_labels = train_data.edge_label
    num_samples, num_classes = all_labels.shape
    class_counts = all_labels.sum(dim=0).clamp(min=1.0)
    pos_weight = (num_samples / (num_classes * class_counts)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

@torch.no_grad()
def forward_split(
    data_obj,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
    model,
    clf,
    threshold=0.5,
):
    batch = data_obj.to(device)
    node_embs, pathway_table = model(batch, combined_embeddings, gene_node_indices, dna_node_indices)

    src, dst = batch.edge_index[0], batch.edge_index[1]
    u = node_embs[src]
    v = node_embs[dst]

    pathway_ids = batch.edge_attr.squeeze(-1)
    pathway_ids = torch.where(pathway_ids < 0, torch.tensor(pathway_table.shape[0] - 1, device=device), pathway_ids)
    pathway_edge_emb = pathway_table[pathway_ids]

    logits = clf(u, v, pathway_edge_emb)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).int().cpu().numpy()
    labels = batch.edge_label.int().cpu().numpy()
    return logits, probs.cpu().numpy(), preds, labels

def train_one_epoch(model, clf, optimizer, criterion):
    model.train()
    clf.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        node_embs, pathway_table = model(batch, TRAIN_COMBINED, TRAIN_GENE_NODE_INDICES, TRAIN_DNA_NODE_INDICES)

        src, dst = batch.edge_index[0], batch.edge_index[1]
        u = node_embs[src]
        v = node_embs[dst]

        pathway_ids = batch.edge_attr.squeeze(-1)
        pathway_ids = torch.where(pathway_ids < 0, torch.tensor(pathway_table.shape[0] - 1, device=device), pathway_ids)
        pathway_edge_emb = pathway_table[pathway_ids]

        logits = clf(u, v, pathway_edge_emb)
        loss = criterion(logits, batch.edge_label)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(train_loader)

@torch.no_grad()
def eval_loss(model, clf, criterion, split="val", threshold=0.5):
    model.eval()
    clf.eval()
    if split == "train":
        logits, _, preds, labels = forward_split(
            train_data, TRAIN_COMBINED, TRAIN_GENE_NODE_INDICES, TRAIN_DNA_NODE_INDICES, model, clf, threshold=threshold
        )
        loss = float(criterion(logits, train_data.to(device).edge_label).item())
        return loss, preds, labels
    if split == "val":
        logits, _, preds, labels = forward_split(
            val_data, VAL_COMBINED, VAL_GENE_NODE_INDICES, VAL_DNA_NODE_INDICES, model, clf, threshold=threshold
        )
        loss = float(criterion(logits, val_data.to(device).edge_label).item())
        return loss, preds, labels
    if split == "test":
        logits, _, preds, labels = forward_split(
            test_data, TEST_COMBINED, TEST_GENE_NODE_INDICES, TEST_DNA_NODE_INDICES, model, clf, threshold=threshold
        )
        loss = float(criterion(logits, test_data.to(device).edge_label).item())
        return loss, preds, labels
    raise ValueError("split must be train/val/test")

# ============================================================
# 12) Optuna objective
# - Optimize VAL MICRO-F1 (maximize).
# - Early stopping inside each trial.
# ===========================================================
def objective(trial: optuna.Trial):
    # Search space (adjust as you like)
    gene_dim = trial.suggest_categorical("NODE_EMBEDDING_DIM", [32, 64, 128, 256])
    num_heads = trial.suggest_categorical("NUM_HEADS", [2, 3, 4, 5, 6])
    lr = trial.suggest_float("LEARNING_RATE", 1e-5, 5e-4, log=True)

    proj1 = trial.suggest_categorical("PROJECTED_DIM_1", [64, 128, 256])
    proj2 = trial.suggest_categorical("PROJECTED_DIM_2", [64, 128, 256])
    proj3 = trial.suggest_categorical("PROJECTED_DIM_3", [8, 16, 32, 64])

    pathway_dim = trial.suggest_categorical("PATHWAY_EMBEDDING_DIM", [32, 64, 128])
    clf_hidden = trial.suggest_categorical("CLF_HIDDEN_DIM", [128, 256, 512])
    clf_layers = trial.suggest_int("CLF_NUM_LAYERS", 2, 4)
    dropout = trial.suggest_float("DROPOUT", 0.2, 0.7)

    num_genes = len(train_entity_id_to_local_index) + 1

    model = GraphTransformer(
        num_genes=num_genes,
        gene_embedding_dim=gene_dim,
        projected_dim_1=proj1,
        projected_dim_2=proj2,
        projected_dim_3=proj3,
        num_heads=num_heads,
        pathway_embedding_dim=pathway_dim,
        dropout=dropout,
    ).to(device)

    clf = EdgeTypeClassifier(
        gene_embedding_dim=gene_dim,
        projected_dim_1=proj1,
        projected_dim_2=proj2,
        projected_dim_3=proj3,
        pathway_embedding_dim=pathway_dim,
        hidden_dim=clf_hidden,
        num_hidden_layers=clf_layers,
    ).to(device)

    criterion = build_loss_pos_weight()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_val_micro_f1 = 0.0
    best_val_macro_f1 = 0.0

    epochs_no_improve = 0
    patience = 10
    max_epochs = 80  # keep trials reasonably fast

    for epoch in range(max_epochs):
        _ = train_one_epoch(model, clf, optimizer, criterion)

        # Compute val loss + preds/labels
        val_loss, val_preds, val_labels = eval_loss(model, clf, criterion, split="val", threshold=0.5)

        # F1 reporting (same as your old code)
        val_metrics = compute_metrics_for_each_relation(val_preds, val_labels)
        val_micro_f1 = val_metrics["Micro Average"]["f1"]
        val_macro_f1 = val_metrics["Macro Average"]["f1"]

        scheduler.step(val_loss)

        # Report val loss for pruning + tracking
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Track best (by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_micro_f1 = val_micro_f1
            best_val_macro_f1 = val_macro_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Store helpful info in trial (so you can see it later)
    trial.set_user_attr("best_val_micro_f1", float(best_val_micro_f1))
    trial.set_user_attr("best_val_macro_f1", float(best_val_macro_f1))

    # Optional: print per trial summary (helps you interpret in real time)
    print(
        f"Trial {trial.number} summary | best_val_loss={best_val_loss:.6f} "
        f"| best_micro_f1={best_val_micro_f1:.4f} | best_macro_f1={best_val_macro_f1:.4f}"
    )

    # IMPORTANT: minimize loss (matches your old approach)
    return best_val_loss

# ============================================================
# 13) Run Optuna study
# ============================================================
def run_optuna(n_trials=300, timeout=None):
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("\n=== OPTUNA BEST (by val loss) ===")
    print("Best value (val loss):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # You can also inspect the stored F1 on the best trial:
    best_trial = study.best_trial
    print("\nBest trial extra metrics:")
    print("  best_val_micro_f1:", best_trial.user_attrs.get("best_val_micro_f1"))
    print("  best_val_macro_f1:", best_trial.user_attrs.get("best_val_macro_f1"))

    return study.best_params, study

# ============================================================
# 14) Train final model with best params, then metrics + FP export
# ============================================================
def ensure_pathway_name(df_raw: pd.DataFrame):
    out = df_raw.copy()
    if pd.api.types.is_numeric_dtype(out["pathway_source"]):
        out["pathway_source_name"] = out["pathway_source"].map(int_to_pathway_source)
    else:
        out["pathway_source_name"] = out["pathway_source"].astype(str)
    return out

@torch.no_grad()
def collect_false_positive_rows(
    split_name: str,
    data_obj,
    df_raw_with_edge_id: pd.DataFrame,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
    model,
    clf,
    threshold: float = 0.5,
):
    model.eval()
    clf.eval()

    logits, probs, preds, labels = forward_split(
        data_obj, combined_embeddings, gene_node_indices, dna_node_indices, model, clf, threshold=threshold
    )

    edge_ids = data_obj.edge_id.detach().cpu().numpy().astype(int)

    df_raw = ensure_pathway_name(df_raw_with_edge_id)
    base = df_raw[["edge_id", "starter_ID", "receiver_ID", "pathway_source_name"]].copy()

    align = pd.DataFrame({"edge_id": edge_ids}).merge(base, on="edge_id", how="left")

    fp_mask = (preds == 1) & (labels == 0)  # [E,C]

    out_rows = []
    for e in range(fp_mask.shape[0]):
        fp_rels = np.where(fp_mask[e])[0]
        for j in fp_rels:
            out_rows.append({
                "split": split_name,
                "source_gene": align.loc[e, "starter_ID"],
                "target_gene": align.loc[e, "receiver_ID"],
                "gene_pathway": align.loc[e, "pathway_source_name"],
                "false_positive_relation": relation_types[j],
                "probability": float(probs[e, j]),
            })

    return pd.DataFrame(out_rows)

def train_final_and_export(best_params: dict, max_epochs=250, patience=20, threshold=0.5):
    gene_dim = best_params["NODE_EMBEDDING_DIM"]
    num_heads = best_params["NUM_HEADS"]
    lr = best_params["LEARNING_RATE"]
    proj1 = best_params["PROJECTED_DIM_1"]
    proj2 = best_params["PROJECTED_DIM_2"]
    proj3 = best_params["PROJECTED_DIM_3"]
    pathway_dim = best_params["PATHWAY_EMBEDDING_DIM"]
    clf_hidden = best_params["CLF_HIDDEN_DIM"]
    clf_layers = best_params["CLF_NUM_LAYERS"]
    dropout = best_params["DROPOUT"]

    num_genes = len(train_entity_id_to_local_index) + 1

    model = GraphTransformer(
        num_genes=num_genes,
        gene_embedding_dim=gene_dim,
        projected_dim_1=proj1,
        projected_dim_2=proj2,
        projected_dim_3=proj3,
        num_heads=num_heads,
        pathway_embedding_dim=pathway_dim,
        dropout=dropout,
    ).to(device)

    clf = EdgeTypeClassifier(
        gene_embedding_dim=gene_dim,
        projected_dim_1=proj1,
        projected_dim_2=proj2,
        projected_dim_3=proj3,
        pathway_embedding_dim=pathway_dim,
        hidden_dim=clf_hidden,
        num_hidden_layers=clf_layers,
    ).to(device)

    criterion = build_loss_pos_weight()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(model, clf, optimizer, criterion)
        val_loss, val_preds, val_labels = eval_loss(model, clf, criterion, split="val", threshold=threshold)
        scheduler.step(val_loss)

        if epoch % 50 == 0:
            print(f"\nEpoch {epoch} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
            print_metrics_table(compute_metrics_for_each_relation(val_preds, val_labels))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {"model": model.state_dict(), "clf": clf.state_dict()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("\nFinal training: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        clf.load_state_dict(best_state["clf"])

    # Metrics on train/val/test
    train_loss, train_preds, train_labels = eval_loss(model, clf, criterion, split="train", threshold=threshold)
    val_loss,   val_preds,   val_labels   = eval_loss(model, clf, criterion, split="val",   threshold=threshold)
    test_loss,  test_preds,  test_labels  = eval_loss(model, clf, criterion, split="test",  threshold=threshold)

    print("\n=== TRAIN METRICS ===")
    print(f"Loss: {train_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(train_preds, train_labels))

    print("\n=== VAL METRICS ===")
    print(f"Loss: {val_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(val_preds, val_labels))

    print("\n=== TEST METRICS ===")
    print(f"Loss: {test_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(test_preds, test_labels))

    # One CSV with false positives across all splits (edge-relation level)
    train_fp = collect_false_positive_rows(
        "train", train_data, train_df_raw,
        TRAIN_COMBINED, TRAIN_GENE_NODE_INDICES, TRAIN_DNA_NODE_INDICES,
        model, clf, threshold=threshold
    )
    val_fp = collect_false_positive_rows(
        "val", val_data, val_df_raw,
        VAL_COMBINED, VAL_GENE_NODE_INDICES, VAL_DNA_NODE_INDICES,
        model, clf, threshold=threshold
    )
    test_fp = collect_false_positive_rows(
        "test", test_data, test_df_raw,
        TEST_COMBINED, TEST_GENE_NODE_INDICES, TEST_DNA_NODE_INDICES,
        model, clf, threshold=threshold
    )

    all_fp = pd.concat([train_fp, val_fp, test_fp], ignore_index=True)
    out_csv = "false_positives_all_splits.csv"
    all_fp.to_csv(out_csv, index=False)

    print("\nFalse positives export:")
    print(f"  saved: {out_csv}")
    print(f"  rows:  {len(all_fp)} (each row = one (edge, relation) false positive)")
    return model, clf, all_fp

# ============================================================
# 15) MAIN
# ============================================================
# 1) Run hyperparameter tuning
best_params, study = run_optuna(n_trials=300)   # change n_trials as needed

# 2) Train final model using best params + metrics + FP export
final_model, final_clf, fp_df = train_final_and_export(best_params, max_epochs=250, patience=20, threshold=0.5)

print("\nDone.")


# %%
best_params = {
    "NODE_EMBEDDING_DIM": 128,
    "NUM_HEADS": 5,
    "LEARNING_RATE": 0.0004651831416437624,
    "PROJECTED_DIM_1": 64,
    "PROJECTED_DIM_2": 256,
    "PROJECTED_DIM_3": 16,
    "PATHWAY_EMBEDDING_DIM": 128,
    "CLF_HIDDEN_DIM": 512,
    "CLF_NUM_LAYERS": 2,
    "DROPOUT": 0.2480908508818471,
}

# Unpack
gene_dim    = best_params["NODE_EMBEDDING_DIM"]
num_heads   = best_params["NUM_HEADS"]
lr          = best_params["LEARNING_RATE"]
proj1       = best_params["PROJECTED_DIM_1"]
proj2       = best_params["PROJECTED_DIM_2"]
proj3       = best_params["PROJECTED_DIM_3"]
pathway_dim = best_params["PATHWAY_EMBEDDING_DIM"]
clf_hidden  = best_params["CLF_HIDDEN_DIM"]
clf_layers  = best_params["CLF_NUM_LAYERS"]
dropout     = best_params["DROPOUT"]

# num_genes should match your train mapping space
num_genes = len(train_entity_id_to_local_index) + 1

# Initialize model + classifier
model = GraphTransformer(
    num_genes=num_genes,
    gene_embedding_dim=gene_dim,
    projected_dim_1=proj1,
    projected_dim_2=proj2,
    projected_dim_3=proj3,
    num_heads=num_heads,
    pathway_embedding_dim=pathway_dim,
    dropout=dropout,
).to(device)

edge_type_classifier = EdgeTypeClassifier(
    gene_embedding_dim=gene_dim,
    projected_dim_1=proj1,
    projected_dim_2=proj2,
    projected_dim_3=proj3,
    pathway_embedding_dim=pathway_dim,
    hidden_dim=clf_hidden,
    num_hidden_layers=clf_layers,
).to(device)

criterion = build_loss_pos_weight()
optimizer = torch.optim.Adam(list(model.parameters()) + list(edge_type_classifier.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print("Initialized final model with best_params.")


# %%
max_epochs = 250
patience = 20
threshold = 0.5

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(max_epochs):
    tr_loss = train_one_epoch(model, edge_type_classifier, optimizer, criterion)
    val_loss, val_preds, val_labels = eval_loss(model, edge_type_classifier, criterion, split="val", threshold=threshold)
    scheduler.step(val_loss)

    if epoch % 50 == 0:
        print(f"\nEpoch {epoch} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
        val_metrics = compute_metrics_for_each_relation(val_preds, val_labels)
        print_metrics_table(val_metrics)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {"model": model.state_dict(), "clf": edge_type_classifier.state_dict()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

# Load best checkpoint
if best_state is not None:
    model.load_state_dict(best_state["model"])
    edge_type_classifier.load_state_dict(best_state["clf"])

# Final metrics on train/val/test
train_loss, train_preds, train_labels = eval_loss(model, edge_type_classifier, criterion, split="train", threshold=threshold)
val_loss,   val_preds,   val_labels   = eval_loss(model, edge_type_classifier, criterion, split="val",   threshold=threshold)
test_loss,  test_preds,  test_labels  = eval_loss(model, edge_type_classifier, criterion, split="test",  threshold=threshold)

print("\n=== TRAIN METRICS ===")
print(f"Loss: {train_loss:.4f}")
print_metrics_table(compute_metrics_for_each_relation(train_preds, train_labels))

print("\n=== VAL METRICS ===")
print(f"Loss: {val_loss:.4f}")
print_metrics_table(compute_metrics_for_each_relation(val_preds, val_labels))

print("\n=== TEST METRICS ===")
print(f"Loss: {test_loss:.4f}")
print_metrics_table(compute_metrics_for_each_relation(test_preds, test_labels))




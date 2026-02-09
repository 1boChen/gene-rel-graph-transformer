# src/gene_rel_gt/cli/tune.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from gene_rel_gt.constants import RELATION_TYPES
from gene_rel_gt.preprocessing.io import load_split_csv
from gene_rel_gt.preprocessing.labels import process_dataset
from gene_rel_gt.preprocessing.graph_build import add_edge_id, df_to_graph
from gene_rel_gt.preprocessing.mappings import (
    fit_pathway_source_mapping,
    apply_pathway_source_mapping,
    create_entity_id_to_local_index_mapping,
    create_entity_id_to_train_index_mapping,
)
from gene_rel_gt.preprocessing.pyg_convert import nx_to_pyg_data
from gene_rel_gt.preprocessing.embeddings import (
    make_combined_embeddings,
    make_node_index_tensor,
    make_gene_index_tensor_in_train_space,
)

from gene_rel_gt.models.graph_transformer import GraphTransformer
from gene_rel_gt.models.edge_classifier import EdgeTypeClassifier
from gene_rel_gt.training.losses import build_bce_with_logits_pos_weight
from gene_rel_gt.training.loops import train_one_epoch, eval_loss
from gene_rel_gt.training.metrics import compute_metrics_for_each_relation


class SingleGraphDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def len(self):
        return 1

    def get(self, idx):
        return self.data


def pick_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = pick_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    out_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Build data once ----------
    train_raw, val_raw, test_raw = load_split_csv(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )

    train_raw = add_edge_id(train_raw)
    val_raw = add_edge_id(val_raw)
    test_raw = add_edge_id(test_raw)

    train_df = process_dataset(train_raw)
    val_df = process_dataset(val_raw)
    test_df = process_dataset(test_raw)

    pathway_source_to_int, _ = fit_pathway_source_mapping(train_df)
    train_df = apply_pathway_source_mapping(train_df, pathway_source_to_int)
    val_df = apply_pathway_source_mapping(val_df, pathway_source_to_int)
    test_df = apply_pathway_source_mapping(test_df, pathway_source_to_int)

    train_graph = df_to_graph(train_df)
    val_graph = df_to_graph(val_df)
    test_graph = df_to_graph(test_df)

    train_data = nx_to_pyg_data(train_graph)
    val_data = nx_to_pyg_data(val_graph)
    test_data = nx_to_pyg_data(test_graph)

    train_loader = DataLoader(SingleGraphDataset(train_data), batch_size=1, shuffle=False)

    # ---------- Mappings ----------
    train_entity_id_to_local_index = create_entity_id_to_local_index_mapping(train_graph)
    val_entity_id_to_local_index = create_entity_id_to_local_index_mapping(val_graph)
    test_entity_id_to_local_index = create_entity_id_to_local_index_mapping(test_graph)

    val_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, val_graph)
    test_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, test_graph)

    # ---------- Embeddings cached ----------
    emb_cfg = cfg["embeddings"]
    dna_dim = int(emb_cfg.get("dna_dim", 768))
    biobert_dim = int(emb_cfg.get("biobert_dim", 768))
    esm2_dim = int(emb_cfg.get("esm2_dim", 2560))

    TRAIN_COMBINED = make_combined_embeddings(
        train_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )
    VAL_COMBINED = make_combined_embeddings(
        val_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )
    TEST_COMBINED = make_combined_embeddings(
        test_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )

    TRAIN_DNA_IDX = make_node_index_tensor(train_entity_id_to_local_index, device=device)
    VAL_DNA_IDX = make_node_index_tensor(val_entity_id_to_local_index, device=device)
    TEST_DNA_IDX = make_node_index_tensor(test_entity_id_to_local_index, device=device)

    TRAIN_GENE_IDX = make_node_index_tensor(train_entity_id_to_local_index, device=device)
    VAL_GENE_IDX = make_gene_index_tensor_in_train_space(
        split_entity_ids=list(val_entity_id_to_local_index.keys()),
        entity_id_to_train_index=val_entity_id_to_train_index,
        device=device,
    )
    TEST_GENE_IDX = make_gene_index_tensor_in_train_space(
        split_entity_ids=list(test_entity_id_to_local_index.keys()),
        entity_id_to_train_index=test_entity_id_to_train_index,
        device=device,
    )

    criterion = build_bce_with_logits_pos_weight(train_data.edge_label, device=device)

    num_genes = len(train_entity_id_to_local_index) + 1
    num_pathway_sources = len(pathway_source_to_int) + 1

    tune_cfg = cfg["optuna"]
    max_epochs = int(tune_cfg.get("max_epochs", 80))
    patience = int(tune_cfg.get("patience", 10))
    threshold = float(tune_cfg.get("threshold", 0.5))

    def objective(trial: optuna.Trial):
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

        model = GraphTransformer(
            num_genes=num_genes,
            gene_embedding_dim=gene_dim,
            projected_dim_1=proj1,
            projected_dim_2=proj2,
            projected_dim_3=proj3,
            num_heads=num_heads,
            pathway_embedding_dim=pathway_dim,
            dropout=dropout,
            embedding_dim=768,
            num_pathway_sources=num_pathway_sources,
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

        optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val_loss = float("inf")
        best_micro_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            _ = train_one_epoch(
                train_loader=train_loader,
                model=model,
                clf=clf,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                combined_embeddings=TRAIN_COMBINED,
                gene_node_indices=TRAIN_GENE_IDX,
                dna_node_indices=TRAIN_DNA_IDX,
            )

            val_loss, val_preds, val_labels = eval_loss(
                data_obj=val_data,
                model=model,
                clf=clf,
                criterion=criterion,
                device=device,
                combined_embeddings=VAL_COMBINED,
                gene_node_indices=VAL_GENE_IDX,
                dna_node_indices=VAL_DNA_IDX,
                threshold=threshold,
            )

            scheduler.step(val_loss)

            val_metrics = compute_metrics_for_each_relation(val_preds, val_labels, RELATION_TYPES)
            val_micro_f1 = val_metrics["Micro Average"]["f1"]
            best_micro_f1 = max(best_micro_f1, float(val_micro_f1))

            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        trial.set_user_attr("best_val_micro_f1", best_micro_f1)
        return best_val_loss

    # Storage so you can resume
    study_db = cfg.get("outputs", {}).get("study_db", str(out_dir / "optuna_study.db"))
    storage = f"sqlite:///{study_db}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)

    n_trials = int(tune_cfg.get("n_trials", 50))
    timeout = tune_cfg.get("timeout_sec", None)

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("\n=== OPTUNA BEST ===")
    print("Best val loss:", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial micro-F1 (stored):", study.best_trial.user_attrs.get("best_val_micro_f1"))

    # Save best params as YAML
    best_params_yaml = cfg.get("outputs", {}).get("best_params_yaml", str(out_dir / "best_params.yaml"))
    best_params_path = Path(best_params_yaml)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(study.best_params, f, sort_keys=False)

    print(f"Saved best params YAML: {best_params_path}")


if __name__ == "__main__":
    main()

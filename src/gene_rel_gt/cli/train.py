# src/gene_rel_gt/cli/train.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
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
from gene_rel_gt.training.metrics import compute_metrics_for_each_relation, print_metrics_table


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

    # Optional: debug CUDA sync (you used this before)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    device = pick_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    out_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Load raw CSVs
    # -------------------------
    train_raw, val_raw, test_raw = load_split_csv(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )

    # Add stable edge_id
    train_raw = add_edge_id(train_raw)
    val_raw = add_edge_id(val_raw)
    test_raw = add_edge_id(test_raw)

    # Labels -> multi-hot vectors
    train_df = process_dataset(train_raw)
    val_df = process_dataset(val_raw)
    test_df = process_dataset(test_raw)

    # Pathway mapping on train only
    pathway_source_to_int, int_to_pathway_source = fit_pathway_source_mapping(train_df)

    train_df = apply_pathway_source_mapping(train_df, pathway_source_to_int)
    val_df = apply_pathway_source_mapping(val_df, pathway_source_to_int)
    test_df = apply_pathway_source_mapping(test_df, pathway_source_to_int)

    # -------------------------
    # 2) DataFrame -> NX -> PyG
    # -------------------------
    train_graph = df_to_graph(train_df)
    val_graph = df_to_graph(val_df)
    test_graph = df_to_graph(test_df)

    train_data = nx_to_pyg_data(train_graph)
    val_data = nx_to_pyg_data(val_graph)
    test_data = nx_to_pyg_data(test_graph)

    # -------------------------
    # 3) Entity mappings
    # -------------------------
    train_entity_id_to_local_index = create_entity_id_to_local_index_mapping(train_graph)
    val_entity_id_to_local_index = create_entity_id_to_local_index_mapping(val_graph)
    test_entity_id_to_local_index = create_entity_id_to_local_index_mapping(test_graph)

    val_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, val_graph)
    test_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, test_graph)

    # -------------------------
    # 4) Embeddings (cached tensors)
    # -------------------------
    emb_cfg = cfg["embeddings"]
    dna_dim = int(emb_cfg.get("dna_dim", 768))
    biobert_dim = int(emb_cfg.get("biobert_dim", 768))
    esm2_dim = int(emb_cfg.get("esm2_dim", 2560))

    train_combined = make_combined_embeddings(
        train_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )
    val_combined = make_combined_embeddings(
        val_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )
    test_combined = make_combined_embeddings(
        test_entity_id_to_local_index,
        dna_csv=emb_cfg["dna_csv"],
        biobert_csv=emb_cfg["biobert_csv"],
        esm2_csv=emb_cfg["esm2_csv"],
        dna_dim=dna_dim,
        biobert_dim=biobert_dim,
        esm2_dim=esm2_dim,
        device=device,
    )

    # dna_node_indices are local indices (same as you did)
    train_dna_node_indices = make_node_index_tensor(train_entity_id_to_local_index, device=device)
    val_dna_node_indices = make_node_index_tensor(val_entity_id_to_local_index, device=device)
    test_dna_node_indices = make_node_index_tensor(test_entity_id_to_local_index, device=device)

    # gene_node_indices:
    # - train uses local indices (train space)
    # - val/test map into train space; unseen -> -1 (clamped in model)
    train_gene_node_indices = make_node_index_tensor(train_entity_id_to_local_index, device=device)

    val_gene_node_indices = make_gene_index_tensor_in_train_space(
        split_entity_ids=list(val_entity_id_to_local_index.keys()),
        entity_id_to_train_index=val_entity_id_to_train_index,
        device=device,
    )
    test_gene_node_indices = make_gene_index_tensor_in_train_space(
        split_entity_ids=list(test_entity_id_to_local_index.keys()),
        entity_id_to_train_index=test_entity_id_to_train_index,
        device=device,
    )

    # -------------------------
    # 5) DataLoaders
    # -------------------------
    batch_size = int(cfg["train"].get("batch_size", 1))
    train_loader = DataLoader(SingleGraphDataset(train_data), batch_size=batch_size, shuffle=False)

    # -------------------------
    # 6) Build model + classifier
    # -------------------------
    mcfg = cfg["model"]
    ccfg = cfg["clf"]

    num_genes = len(train_entity_id_to_local_index) + 1
    num_pathway_sources = len(pathway_source_to_int) + 1  # unseen bucket

    model = GraphTransformer(
        num_genes=num_genes,
        gene_embedding_dim=int(mcfg["node_embedding_dim"]),
        projected_dim_1=int(mcfg["projected_dim_1"]),
        projected_dim_2=int(mcfg["projected_dim_2"]),
        projected_dim_3=int(mcfg["projected_dim_3"]),
        num_heads=int(mcfg["num_heads"]),
        pathway_embedding_dim=int(mcfg["pathway_embedding_dim"]),
        dropout=float(mcfg["dropout"]),
        embedding_dim=int(mcfg.get("embedding_dim", 768)),
        num_pathway_sources=num_pathway_sources,
    ).to(device)

    clf = EdgeTypeClassifier(
        gene_embedding_dim=int(mcfg["node_embedding_dim"]),
        projected_dim_1=int(mcfg["projected_dim_1"]),
        projected_dim_2=int(mcfg["projected_dim_2"]),
        projected_dim_3=int(mcfg["projected_dim_3"]),
        pathway_embedding_dim=int(mcfg["pathway_embedding_dim"]),
        hidden_dim=int(ccfg["hidden_dim"]),
        num_hidden_layers=int(ccfg["num_layers"]),
    ).to(device)

    # Loss + optimizer
    criterion = build_bce_with_logits_pos_weight(train_data.edge_label, device=device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=float(cfg["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # -------------------------
    # 7) Training loop (early stopping)
    # -------------------------
    max_epochs = int(cfg["train"]["max_epochs"])
    patience = int(cfg["train"]["patience"])
    threshold = float(cfg["train"].get("threshold", 0.5))
    log_every = int(cfg["train"].get("log_every", 50))

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            clf=clf,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            combined_embeddings=train_combined,
            gene_node_indices=train_gene_node_indices,
            dna_node_indices=train_dna_node_indices,
        )

        val_loss, val_preds, val_labels = eval_loss(
            data_obj=val_data,
            model=model,
            clf=clf,
            criterion=criterion,
            device=device,
            combined_embeddings=val_combined,
            gene_node_indices=val_gene_node_indices,
            dna_node_indices=val_dna_node_indices,
            threshold=threshold,
        )

        scheduler.step(val_loss)

        if epoch % log_every == 0:
            print(f"\nEpoch {epoch} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
            val_metrics = compute_metrics_for_each_relation(val_preds, val_labels, RELATION_TYPES)
            print_metrics_table(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {"model": model.state_dict(), "clf": clf.state_dict()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("\nEarly stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        clf.load_state_dict(best_state["clf"])

    # -------------------------
    # 8) Final evaluation
    # -------------------------
    train_loss, train_preds, train_labels = eval_loss(
        data_obj=train_data,
        model=model,
        clf=clf,
        criterion=criterion,
        device=device,
        combined_embeddings=train_combined,
        gene_node_indices=train_gene_node_indices,
        dna_node_indices=train_dna_node_indices,
        threshold=threshold,
    )
    val_loss, val_preds, val_labels = eval_loss(
        data_obj=val_data,
        model=model,
        clf=clf,
        criterion=criterion,
        device=device,
        combined_embeddings=val_combined,
        gene_node_indices=val_gene_node_indices,
        dna_node_indices=val_dna_node_indices,
        threshold=threshold,
    )
    test_loss, test_preds, test_labels = eval_loss(
        data_obj=test_data,
        model=model,
        clf=clf,
        criterion=criterion,
        device=device,
        combined_embeddings=test_combined,
        gene_node_indices=test_gene_node_indices,
        dna_node_indices=test_dna_node_indices,
        threshold=threshold,
    )

    print("\n=== TRAIN METRICS ===")
    print(f"Loss: {train_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(train_preds, train_labels, RELATION_TYPES))

    print("\n=== VAL METRICS ===")
    print(f"Loss: {val_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(val_preds, val_labels, RELATION_TYPES))

    print("\n=== TEST METRICS ===")
    print(f"Loss: {test_loss:.4f}")
    print_metrics_table(compute_metrics_for_each_relation(test_preds, test_labels, RELATION_TYPES))

    # Save checkpoint
    ckpt_path = out_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "clf_state_dict": clf.state_dict(),
            "config": cfg,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

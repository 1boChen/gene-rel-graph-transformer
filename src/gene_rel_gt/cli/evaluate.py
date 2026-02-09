# src/gene_rel_gt/cli/evaluate.py

from __future__ import annotations

import argparse
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
from gene_rel_gt.training.loops import eval_loss
from gene_rel_gt.training.metrics import compute_metrics_for_each_relation, print_metrics_table

from gene_rel_gt.inference.false_positives import collect_false_positive_rows


def pick_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class SingleGraphDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def len(self):
        return 1

    def get(self, idx):
        return self.data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint (from train).")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--export-fp", action="store_true", help="Export false positives CSV.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = pick_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    out_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Rebuild data pipeline (same as train) ----------
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

    pathway_source_to_int, int_to_pathway_source = fit_pathway_source_mapping(train_df)

    train_df = apply_pathway_source_mapping(train_df, pathway_source_to_int)
    val_df = apply_pathway_source_mapping(val_df, pathway_source_to_int)
    test_df = apply_pathway_source_mapping(test_df, pathway_source_to_int)

    train_graph = df_to_graph(train_df)
    val_graph = df_to_graph(val_df)
    test_graph = df_to_graph(test_df)

    train_data = nx_to_pyg_data(train_graph)
    val_data = nx_to_pyg_data(val_graph)
    test_data = nx_to_pyg_data(test_graph)

    train_entity_id_to_local_index = create_entity_id_to_local_index_mapping(train_graph)
    val_entity_id_to_local_index = create_entity_id_to_local_index_mapping(val_graph)
    test_entity_id_to_local_index = create_entity_id_to_local_index_mapping(test_graph)

    val_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, val_graph)
    test_entity_id_to_train_index = create_entity_id_to_train_index_mapping(train_graph, test_graph)

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

    train_dna_node_indices = make_node_index_tensor(train_entity_id_to_local_index, device=device)
    val_dna_node_indices = make_node_index_tensor(val_entity_id_to_local_index, device=device)
    test_dna_node_indices = make_node_index_tensor(test_entity_id_to_local_index, device=device)

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

    # ---------- Build model objects (same hyperparams as config) ----------
    mcfg = cfg["model"]
    ccfg = cfg["clf"]

    num_genes = len(train_entity_id_to_local_index) + 1
    num_pathway_sources = len(pathway_source_to_int) + 1

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

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    clf.load_state_dict(ckpt["clf_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    threshold = float(cfg["train"].get("threshold", 0.5))

    # Criterion for loss reporting (needs train labels for pos_weight)
    criterion = build_bce_with_logits_pos_weight(train_data.edge_label, device=device)

    def eval_one(name: str):
        if name == "train":
            data_obj = train_data
            combined = train_combined
            gene_idx = train_gene_node_indices
            dna_idx = train_dna_node_indices
            df_raw = train_raw
        elif name == "val":
            data_obj = val_data
            combined = val_combined
            gene_idx = val_gene_node_indices
            dna_idx = val_dna_node_indices
            df_raw = val_raw
        else:
            data_obj = test_data
            combined = test_combined
            gene_idx = test_gene_node_indices
            dna_idx = test_dna_node_indices
            df_raw = test_raw

        loss, preds, labels = eval_loss(
            data_obj=data_obj,
            model=model,
            clf=clf,
            criterion=criterion,
            device=device,
            combined_embeddings=combined,
            gene_node_indices=gene_idx,
            dna_node_indices=dna_idx,
            threshold=threshold,
        )

        print(f"\n=== {name.upper()} METRICS ===")
        print(f"Loss: {loss:.4f}")
        metrics = compute_metrics_for_each_relation(preds, labels, RELATION_TYPES)
        print_metrics_table(metrics)

        if args.export_fp:
            fp_df = collect_false_positive_rows(
                split_name=name,
                data_obj=data_obj,
                df_raw_with_edge_id=df_raw,
                int_to_pathway_source=int_to_pathway_source,
                combined_embeddings=combined,
                gene_node_indices=gene_idx,
                dna_node_indices=dna_idx,
                model=model,
                clf=clf,
                device=device,
                threshold=threshold,
            )
            out_csv = out_dir / f"false_positives_{name}.csv"
            fp_df.to_csv(out_csv, index=False)
            print(f"Exported false positives: {out_csv} | rows={len(fp_df)}")

    if args.split == "all":
        for s in ["train", "val", "test"]:
            eval_one(s)
    else:
        eval_one(args.split)


if __name__ == "__main__":
    main()

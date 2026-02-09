# src/gene_rel_gt/inference/false_positives.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gene_rel_gt.constants import RELATION_TYPES
from gene_rel_gt.training.loops import forward_split


def ensure_pathway_name(df_raw: pd.DataFrame, int_to_pathway_source: dict) -> pd.DataFrame:
    """
    Ensure df has pathway_source_name string column.
    Works whether df_raw['pathway_source'] is already int-coded or still strings.
    """
    out = df_raw.copy()
    if pd.api.types.is_numeric_dtype(out["pathway_source"]):
        out["pathway_source_name"] = out["pathway_source"].map(int_to_pathway_source)
    else:
        out["pathway_source_name"] = out["pathway_source"].astype(str)
    return out


@torch.no_grad()
def collect_false_positive_rows(
    *,
    split_name: str,
    data_obj,
    df_raw_with_edge_id: pd.DataFrame,
    int_to_pathway_source: dict,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
    model,
    clf,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Output rows at (edge, relation) level where pred=1 and label=0.
    """
    model.eval()
    clf.eval()

    logits, probs, preds, labels = forward_split(
        data_obj=data_obj,
        combined_embeddings=combined_embeddings,
        gene_node_indices=gene_node_indices,
        dna_node_indices=dna_node_indices,
        model=model,
        clf=clf,
        device=device,
        threshold=threshold,
    )

    # edge_id is stored in PyG Data by from_networkx as "edge_id"
    edge_ids = data_obj.edge_id.detach().cpu().numpy().astype(int)

    df_raw = ensure_pathway_name(df_raw_with_edge_id, int_to_pathway_source)
    base = df_raw[["edge_id", "starter_ID", "receiver_ID", "pathway_source_name"]].copy()
    align = pd.DataFrame({"edge_id": edge_ids}).merge(base, on="edge_id", how="left")

    fp_mask = (preds == 1) & (labels == 0)  # [E,C]

    out_rows = []
    for e in range(fp_mask.shape[0]):
        fp_rels = np.where(fp_mask[e])[0]
        for j in fp_rels:
            out_rows.append(
                {
                    "split": split_name,
                    "edge_id": int(edge_ids[e]),
                    "source_gene": align.loc[e, "starter_ID"],
                    "target_gene": align.loc[e, "receiver_ID"],
                    "gene_pathway": align.loc[e, "pathway_source_name"],
                    "false_positive_relation": RELATION_TYPES[j],
                    "probability": float(probs[e, j]),
                }
            )

    return pd.DataFrame(out_rows)

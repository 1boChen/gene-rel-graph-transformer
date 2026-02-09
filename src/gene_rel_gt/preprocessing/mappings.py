# src/gene_rel_gt/preprocessing/mappings.py

from __future__ import annotations
import pandas as pd
import networkx as nx


def fit_pathway_source_mapping(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Fit pathway_source -> int mapping on TRAIN only.
    Returns (pathway_source_to_int, int_to_pathway_source).
    """
    unique_sources = train_df["pathway_source"].unique()
    pathway_source_to_int = {p: i for i, p in enumerate(unique_sources)}
    int_to_pathway_source = {v: k for k, v in pathway_source_to_int.items()}
    return pathway_source_to_int, int_to_pathway_source


def apply_pathway_source_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Apply mapping. Unseen pathway_source -> -1.
    """
    out = df.copy()
    out["pathway_source"] = out["pathway_source"].map(mapping).fillna(-1).astype(int)
    return out


def create_entity_id_to_local_index_mapping(graph: nx.MultiDiGraph) -> dict:
    """
    Map each node ID in the graph to local index [0..N-1] for that split.
    """
    return {node: idx for idx, node in enumerate(graph.nodes())}


def create_entity_id_to_train_index_mapping(train_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph) -> dict:
    """
    Map nodes in a split graph to the TRAIN node index space.
    Unseen nodes get mapped to unseen_index = len(train_nodes).
    """
    mapping = {}
    train_nodes = list(train_graph.nodes())
    train_pos = {n: i for i, n in enumerate(train_nodes)}
    unseen_index = len(train_nodes)

    for node in graph.nodes():
        mapping[node] = train_pos.get(node, unseen_index)

    return mapping

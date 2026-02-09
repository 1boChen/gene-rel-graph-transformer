# src/gene_rel_gt/preprocessing/graph_build.py

from __future__ import annotations
import pandas as pd
import networkx as nx


def add_edge_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a stable edge_id column based on row index (per split).
    """
    out = df.copy()
    out["edge_id"] = out.index.astype(int)
    return out


def df_to_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build a NetworkX MultiDiGraph from a dataframe that contains:
      starter_ID, receiver_ID, relation_vector, pathway_source, edge_id
    """
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

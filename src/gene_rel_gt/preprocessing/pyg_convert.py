# src/gene_rel_gt/preprocessing/pyg_convert.py

from __future__ import annotations
import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def nx_to_pyg_data(graph: nx.MultiDiGraph):
    """
    Convert NetworkX MultiDiGraph -> PyG Data and attach:
      - edge_attr (pathway_source) as LongTensor [E,1]
      - edge_label multi-hot float tensor [E,C]
    """
    data = from_networkx(graph)

    pathway_sources = []
    edge_labels_list = []

    for _, _, edge_data in graph.edges(data=True):
        pathway_sources.append(int(edge_data["pathway_source"]))
        edge_labels_list.append(torch.tensor(edge_data["interaction_type"], dtype=torch.float))

    data.edge_attr = torch.tensor(pathway_sources, dtype=torch.long).unsqueeze(1)
    data.edge_label = torch.stack(edge_labels_list)

    return data

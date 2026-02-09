# src/gene_rel_gt/models/edge_classifier.py

from __future__ import annotations

import torch
import torch.nn as nn

from gene_rel_gt.constants import NUM_EDGE_TYPES


class EdgeTypeClassifier(nn.Module):
    def __init__(
        self,
        *,
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

    def forward(self, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((u, v, p), dim=1))

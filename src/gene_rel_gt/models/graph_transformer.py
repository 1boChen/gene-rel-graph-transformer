# src/gene_rel_gt/models/graph_transformer.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from gene_rel_gt.models.layers import MultiHeadAttention


class GraphTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_genes: int,
        gene_embedding_dim: int,
        projected_dim_1: int,
        projected_dim_2: int,
        projected_dim_3: int,
        num_heads: int,
        pathway_embedding_dim: int,
        dropout: float,
        embedding_dim: int,
        num_pathway_sources: Optional[int] = None,
    ):
        super().__init__()

        if num_pathway_sources is None:
            # Caller should pass correct value, but we keep this for safety.
            # In your old code you used len(pathway_source_to_int) + 1.
            raise ValueError("num_pathway_sources must be provided explicitly.")

        num_hidden_units = projected_dim_1 + projected_dim_2 + projected_dim_3 + gene_embedding_dim

        self.gene_embedding = nn.Embedding(num_genes, gene_embedding_dim)
        self.pathway_embedding = nn.Embedding(num_pathway_sources, pathway_embedding_dim)

        self.dna_projection_1 = nn.Linear(embedding_dim, projected_dim_1)
        self.dna_projection_2 = nn.Linear(embedding_dim, projected_dim_2)
        self.dna_projection_3 = nn.Linear(2560, projected_dim_3)  # keep fixed as in your current code

        self.mha = MultiHeadAttention(
            embed_size=projected_dim_1 + projected_dim_2 + projected_dim_3,
            num_heads=4,  # keep fixed as you currently do
        )

        self.conv1 = TransformerConv(
            num_hidden_units,
            num_hidden_units,
            heads=num_heads,
            dropout=dropout,
            edge_dim=pathway_embedding_dim,
            concat=False,
        )
        self.conv2 = TransformerConv(
            num_hidden_units,
            num_hidden_units,
            heads=1,
            dropout=dropout,
            edge_dim=pathway_embedding_dim,
        )

        self.norm1 = nn.LayerNorm(num_hidden_units)
        self.norm2 = nn.LayerNorm(num_hidden_units)

        self.beta_layer_1 = nn.Linear(num_hidden_units * 2, 1)
        self.beta_layer_2 = nn.Linear(num_hidden_units * 2, 1)

        self.embedding_dim = embedding_dim
        self.dropout = dropout

    def forward(self, data, combined_embeddings, gene_node_indices, dna_node_indices):
        # Gene embedding lookup in train-space (clamped)
        max_index = self.gene_embedding.num_embeddings - 1
        gene_emb = self.gene_embedding(torch.clamp(gene_node_indices, min=0, max=max_index))
        gene_emb = F.normalize(gene_emb, p=2, dim=1)

        # Pathway edge embeddings; negative -> unseen bucket = last index
        pathway_ids = data.edge_attr.squeeze(-1)
        pathway_ids = torch.where(
            pathway_ids < 0,
            torch.tensor(self.pathway_embedding.num_embeddings - 1, device=pathway_ids.device),
            pathway_ids,
        )
        pathway_emb_edges = self.pathway_embedding(pathway_ids)

        # Split combined embeddings into dna / biobert / esm2
        ed = self.embedding_dim
        dna_1 = combined_embeddings[dna_node_indices, :ed]
        dna_2 = combined_embeddings[dna_node_indices, ed : 2 * ed]
        dna_3 = combined_embeddings[dna_node_indices, 2 * ed :]  # expected 2560 dims

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

# src/gene_rel_gt/models/layers.py

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    NOTE: This is the same implementation you currently have.
    It is not a full Transformer attention; it does per-node attention across heads
    using elementwise q*k and softmax over the head dimension.
    """

    def __init__(self, embed_size: int, num_heads: int = 4):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
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

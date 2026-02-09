# src/gene_rel_gt/preprocessing/embeddings.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_embedding_csv_to_tensor(
    csv_file: str,
    embedding_dim: int,
    entity_id_to_local_index: dict,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Loads embeddings from a CSV with columns:
      - entity_ID
      - embedding (comma-separated floats)
    Returns a float tensor [num_nodes, embedding_dim] with StandardScaler normalization.
    Missing entities get zero vector before normalization (same as your current logic).
    """
    df = pd.read_csv(csv_file)

    all_embeddings = np.zeros((len(entity_id_to_local_index), embedding_dim), dtype=float)

    for _, row in df.iterrows():
        entity_id = row["entity_ID"]
        if entity_id in entity_id_to_local_index:
            local_index = entity_id_to_local_index[entity_id]
            emb = np.array([float(x) for x in str(row["embedding"]).split(",")], dtype=float)
            all_embeddings[local_index] = emb

    scaler = StandardScaler()
    normalized = scaler.fit_transform(all_embeddings)

    t = torch.tensor(normalized, dtype=torch.float)
    if device is not None:
        t = t.to(device)
    return t


def make_combined_embeddings(
    entity_id_to_local_index: dict,
    dna_csv: str,
    biobert_csv: str,
    esm2_csv: str,
    dna_dim: int = 768,
    biobert_dim: int = 768,
    esm2_dim: int = 2560,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns concatenated embeddings: [dna, biobert, esm2] => [N, dna_dim + biobert_dim + esm2_dim]
    """
    dna = load_embedding_csv_to_tensor(dna_csv, dna_dim, entity_id_to_local_index, device=device)
    biobert = load_embedding_csv_to_tensor(biobert_csv, biobert_dim, entity_id_to_local_index, device=device)
    esm2 = load_embedding_csv_to_tensor(esm2_csv, esm2_dim, entity_id_to_local_index, device=device)
    return torch.cat((dna, biobert, esm2), dim=1)


def make_node_index_tensor(entity_id_to_local_index: dict, device: torch.device) -> torch.Tensor:
    """
    For your current pattern, this returns tensor(list(local indices)).
    """
    return torch.tensor(list(entity_id_to_local_index.values()), dtype=torch.long, device=device)


def make_gene_index_tensor_in_train_space(
    split_entity_ids: list,
    entity_id_to_train_index: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    For val/test: map split nodes into train space; unseen -> -1 (you clamp later).
    """
    return torch.tensor(
        [entity_id_to_train_index.get(node, -1) for node in split_entity_ids],
        dtype=torch.long,
        device=device,
    )

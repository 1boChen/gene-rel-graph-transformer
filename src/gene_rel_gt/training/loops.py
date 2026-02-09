# src/gene_rel_gt/training/loops.py

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def forward_split(
    *,
    data_obj,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
    model,
    clf,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Mirrors your forward_split exactly.
    Returns: logits (tensor on device), probs (np), preds (np int), labels (np int)
    """
    batch = data_obj.to(device)

    node_embs, pathway_table = model(batch, combined_embeddings, gene_node_indices, dna_node_indices)

    src, dst = batch.edge_index[0], batch.edge_index[1]
    u = node_embs[src]
    v = node_embs[dst]

    pathway_ids = batch.edge_attr.squeeze(-1)
    pathway_ids = torch.where(
        pathway_ids < 0,
        torch.tensor(pathway_table.shape[0] - 1, device=device),
        pathway_ids,
    )
    pathway_edge_emb = pathway_table[pathway_ids]

    logits = clf(u, v, pathway_edge_emb)
    probs = torch.sigmoid(logits)

    preds = (probs > threshold).int().cpu().numpy()
    labels = batch.edge_label.int().cpu().numpy()

    return logits, probs.cpu().numpy(), preds, labels


def train_one_epoch(
    *,
    train_loader,
    model,
    clf,
    optimizer,
    criterion,
    device: torch.device,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
) -> float:
    model.train()
    clf.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        node_embs, pathway_table = model(batch, combined_embeddings, gene_node_indices, dna_node_indices)

        src, dst = batch.edge_index[0], batch.edge_index[1]
        u = node_embs[src]
        v = node_embs[dst]

        pathway_ids = batch.edge_attr.squeeze(-1)
        pathway_ids = torch.where(
            pathway_ids < 0,
            torch.tensor(pathway_table.shape[0] - 1, device=device),
            pathway_ids,
        )
        pathway_edge_emb = pathway_table[pathway_ids]

        logits = clf(u, v, pathway_edge_emb)
        loss = criterion(logits, batch.edge_label)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def eval_loss(
    *,
    data_obj,
    model,
    clf,
    criterion,
    device: torch.device,
    combined_embeddings,
    gene_node_indices,
    dna_node_indices,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    clf.eval()

    logits, _, preds, labels = forward_split(
        data_obj=data_obj,
        combined_embeddings=combined_embeddings,
        gene_node_indices=gene_node_indices,
        dna_node_indices=dna_node_indices,
        model=model,
        clf=clf,
        device=device,
        threshold=threshold,
    )

    loss = float(criterion(logits, data_obj.to(device).edge_label).item())
    return loss, preds, labels

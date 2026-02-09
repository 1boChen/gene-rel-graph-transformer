# src/gene_rel_gt/training/losses.py

from __future__ import annotations

import torch
import torch.nn as nn


def build_bce_with_logits_pos_weight(train_edge_label: torch.Tensor, device: torch.device) -> nn.Module:
    """
    Same idea as your build_loss_pos_weight():
    pos_weight = num_samples / (num_classes * class_counts)
    """
    all_labels = train_edge_label
    num_samples, num_classes = all_labels.shape
    class_counts = all_labels.sum(dim=0).clamp(min=1.0)
    pos_weight = (num_samples / (num_classes * class_counts)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

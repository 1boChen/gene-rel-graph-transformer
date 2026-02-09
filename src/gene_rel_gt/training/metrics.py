# src/gene_rel_gt/training/metrics.py

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics_for_each_relation(
    preds01: np.ndarray,
    labels01: np.ndarray,
    relation_types: list[str],
) -> dict:
    metrics = {}

    for i, rel in enumerate(relation_types):
        metrics[rel] = {
            "precision": precision_score(labels01[:, i], preds01[:, i], zero_division=0),
            "recall": recall_score(labels01[:, i], preds01[:, i], zero_division=0),
            "f1": f1_score(labels01[:, i], preds01[:, i], zero_division=0),
        }

    metrics["Micro Average"] = {
        "precision": precision_score(labels01, preds01, average="micro", zero_division=0),
        "recall": recall_score(labels01, preds01, average="micro", zero_division=0),
        "f1": f1_score(labels01, preds01, average="micro", zero_division=0),
    }
    metrics["Macro Average"] = {
        "precision": precision_score(labels01, preds01, average="macro", zero_division=0),
        "recall": recall_score(labels01, preds01, average="macro", zero_division=0),
        "f1": f1_score(labels01, preds01, average="macro", zero_division=0),
    }
    return metrics


def print_metrics_table(metrics: dict) -> None:
    print("{:<22} {:<10} {:<10} {:<10}".format("Relation", "Precision", "Recall", "F1"))
    for rel, s in metrics.items():
        print("{:<22} {:<10.4f} {:<10.4f} {:<10.4f}".format(rel, s["precision"], s["recall"], s["f1"]))

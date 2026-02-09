# src/gene_rel_gt/preprocessing/io.py

from __future__ import annotations
import pandas as pd


def load_split_csv(train_csv: str, val_csv: str, test_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, val_df, test_df

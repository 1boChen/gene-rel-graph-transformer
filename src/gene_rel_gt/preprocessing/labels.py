# src/gene_rel_gt/preprocessing/labels.py

from __future__ import annotations

import pandas as pd
from gene_rel_gt.constants import NUM_EDGE_TYPES, RELATION_TO_INDEX


def subtype_to_vector(subtype_name: str) -> list[int]:
    """
    Convert subtype string (possibly multi-label like 'activation,, inhibition')
    into a multi-hot vector of length NUM_EDGE_TYPES.
    """
    if subtype_name == "no_interaction":
        return [0] * NUM_EDGE_TYPES

    vec = [0] * NUM_EDGE_TYPES
    for subtype in str(subtype_name).split(",, "):
        if subtype in RELATION_TO_INDEX:
            vec[RELATION_TO_INDEX[subtype]] = 1
    return vec


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'relation_vector' column to the dataframe.
    """
    out = df.copy()
    out["relation_vector"] = out["subtype_name"].apply(subtype_to_vector)
    return out

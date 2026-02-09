# src/gene_rel_gt/constants.py

from __future__ import annotations

RELATION_TYPES = [
    "activation",
    "compound",
    "inhibition",
    "binding/association",
    "expression",
    "phosphorylation",
    "dephosphorylation",
    "state change",
    "ubiquitination",
    "repression",
    "dissociation",
]

NUM_EDGE_TYPES = len(RELATION_TYPES)

RELATION_TO_INDEX = {r: i for i, r in enumerate(RELATION_TYPES)}

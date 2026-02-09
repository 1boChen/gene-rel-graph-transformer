# Gene Relation Graph Transformer (Gene-Rel-GT)

Graph Transformer + multi-label edge classifier for predicting gene–gene relation types using **precomputed embeddings**
(e.g., DNA/BioBERT/ESM2) and a pathway-source edge attribute.

This repository provides:
- Training (early stopping) and evaluation scripts
- Optuna hyperparameter tuning
- False-positive export for error analysis

---

## Related paper (please cite)

This codebase is tightly related to:

Y. Chen, D. Xu, R. Hammer and M. Popescu, “Predicting Gene Relations with a Graph Transformer Network Integrating DNA, Protein, and Descriptive Data,” *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Lisbon, Portugal, 2024, pp. 3101–3104, doi: 10.1109/BIBM62325.2024.10821812.

If you use this repository in your work, please cite the paper (and optionally this software repo; see `CITATION.cff`).

---

## Installation

### Editable install (recommended for dev)

```bash
pip install -e .
```

## Data format
### 1. Edge CSVs (train / val / test)
a. Each split CSV must contain the following columns:
b. starter_ID (source gene/entity ID)
c. receiver_ID (target gene/entity ID)
d. subtype_name (string label; may contain multiple labels separated by ",, ")
e. pathway_source (string pathway/source name)

Notes:
The pipeline will add an edge_id column automatically.
The label separator is assumed to be exactly ",, " (comma-comma-space), matching your current preprocessing.

### 2. Embedding CSVs (precomputed)

Each embedding CSV must contain:
a. entity_ID
b. embedding (comma-separated float string)

You provide three embedding files:
a. DNA embeddings (default dim = 768)
b. BioBERT embeddings (default dim = 768)
c. ESM2 embeddings (default dim = 2560)

Missing entities are assigned zero vectors prior to normalization.

## Configuration
Edit configs/default.yaml to point to your files:
data.train_csv, data.val_csv, data.test_csv
embeddings.dna_csv, embeddings.biobert_csv, embeddings.esm2_csv

## Train
```bash
python -m gene_rel_gt.cli.train --config configs/default.yaml
```

Outputs: outputs/best_model.pt

## Evaluate
Evaluate a checkpoint on a specific split:
```bash
python -m gene_rel_gt.cli.evaluate --config configs/default.yaml --checkpoint outputs/best_model.pt --split test
```

Evaluate all splits and export false positives:
```bash
python -m gene_rel_gt.cli.evaluate --config configs/default.yaml --checkpoint outputs/best_model.pt --split all --export-fp
```

Outputs (when --export-fp is enabled):
a. outputs/false_positives_train.csv
b. outputs/false_positives_val.csv
c. outputs/false_positives_test.csv
Each row corresponds to a single (edge, relation) false positive with an associated probability.

## Hyperparameter tuning (Optuna)
```bash
python -m gene_rel_gt.cli.tune --config configs/optuna.yaml
```
Outputs:
outputs/optuna_study.db (SQLite; resumable study)
outputs/best_params.yaml

## Project structure (high level)

src/gene_rel_gt/ : library code
src/gene_rel_gt/cli/ : runnable commands (train/evaluate/tune)
src/gene_rel_gt/models/ : GraphTransformer + classifier
src/gene_rel_gt/preprocessing/ : CSV → graph → PyG + embeddings
src/gene_rel_gt/training/ : loss, metrics, loops
src/gene_rel_gt/inference/ : false positive export utilities
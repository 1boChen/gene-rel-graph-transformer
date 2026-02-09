import os
from pathlib import Path
import pandas as pd

# ----- CONFIG YOU EDIT -----
FULL_DATA_DIR = Path(r"C:\yc\gene-rel-graph-transformer\data\full")  # put your full files here (not committed)
OUT_DIR = Path(r"C:\yc\gene-rel-graph-transformer\data\sample")     # committed sample folder

TRAIN_IN = FULL_DATA_DIR / "multilabel_train.csv"
VAL_IN   = FULL_DATA_DIR / "multilabel_val.csv"
TEST_IN  = FULL_DATA_DIR / "multilabel_test.csv"

DNA_IN     = FULL_DATA_DIR / "gene_embeddings.csv"
BIOBERT_IN = FULL_DATA_DIR / "gene_biobert_embeddings.csv"
ESM2_IN    = FULL_DATA_DIR / "esm2_embeddings.csv"

# Choose small numbers
N_TRAIN_EDGES = 80
N_VAL_EDGES   = 20
N_TEST_EDGES  = 20
RANDOM_SEED   = 42
# ---------------------------


def sample_edges(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=RANDOM_SEED).copy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load full split CSVs
    train = pd.read_csv(TRAIN_IN)
    val   = pd.read_csv(VAL_IN)
    test  = pd.read_csv(TEST_IN)

    # 2) Sample edges
    train_s = sample_edges(train, N_TRAIN_EDGES)
    val_s   = sample_edges(val, N_VAL_EDGES)
    test_s  = sample_edges(test, N_TEST_EDGES)

    # 3) Collect required entity IDs
    needed_ids = set(train_s["starter_ID"]).union(train_s["receiver_ID"])
    needed_ids |= set(val_s["starter_ID"]).union(val_s["receiver_ID"])
    needed_ids |= set(test_s["starter_ID"]).union(test_s["receiver_ID"])

    print(f"Sample edges: train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    print(f"Unique entities needed: {len(needed_ids)}")

    # 4) Save sampled split CSVs
    train_s.to_csv(OUT_DIR / "multilabel_train.csv", index=False)
    val_s.to_csv(OUT_DIR / "multilabel_val.csv", index=False)
    test_s.to_csv(OUT_DIR / "multilabel_test.csv", index=False)

    # 5) Filter embedding files to needed ids
    def filter_emb(in_path: Path, out_name: str):
        df = pd.read_csv(in_path)
        df_s = df[df["entity_ID"].isin(needed_ids)].copy()
        df_s.to_csv(OUT_DIR / out_name, index=False)
        print(f"{out_name}: {len(df_s)} rows")

    filter_emb(DNA_IN, "gene_embeddings.csv")
    filter_emb(BIOBERT_IN, "gene_biobert_embeddings.csv")
    filter_emb(ESM2_IN, "esm2_embeddings.csv")

    print("\nDone. Sample data written to:", OUT_DIR)


if __name__ == "__main__":
    main()

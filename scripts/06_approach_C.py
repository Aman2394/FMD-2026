"""
Script 06 — Approach C: Fold-Safe kNN Memory (R006).
TF-IDF embeddings + kNN retrieval features + meta-LR.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.models.approach_C import KNNCalibrator
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
RUN_ID     = "R006"
K          = 15
SEEDS      = [0, 1, 2]


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    seed_val_f1s = []
    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n=== Approach C | Seed {seed} ===")

        tfidf   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30_000)
        X_train = tfidf.fit_transform(train["claim_text"]).toarray()
        X_val   = tfidf.transform(val["claim_text"]).toarray()
        X_test  = tfidf.transform(test["claim_text"]).toarray()
        y_train = train["label"].values
        y_val   = val["label"].values
        y_test  = test["label"].values

        cal = KNNCalibrator(k=K)
        cal.fit(X_train, y_train, X_val, y_val)

        for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_prob = cal.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            m = compute_all(y_true, y_pred, y_prob)
            print(f"  [{split_name}] macro_f1={m['macro_f1']:.4f}  f1_false={m['f1_false']:.4f}")
            if split_name == "val":
                m_val = m
            else:
                m_test = m

        seed_val_f1s.append(m_val["macro_f1"])
        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}", "model": "Approach C (kNN Memory)",
            "dedup": "Y", "split": "stratified", "seed": seed,
            **{k: f"{v:.4f}" for k, v in m_val.items()},
            "notes": f"k={K} cosine kNN meta-LR test_f1={m_test['macro_f1']:.4f}",
        })

    save_manifest(RUN_ID, config={"model": "Approach C", "k": K, "seeds": SEEDS}, hashes=hashes)
    print(f"\nR006 avg val macro_f1={np.mean(seed_val_f1s):.4f}")


if __name__ == "__main__":
    main()

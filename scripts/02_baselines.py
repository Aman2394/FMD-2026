"""
Script 02 — Baseline model evaluation (stratified train/val/test split, seeds 0/1/2).

Logs R001 (TF-IDF+LR) and R002 (kNN vote) to runs/run_log.csv.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.models.baselines import tfidf_lr, knn_vote
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"

SEEDS  = [0, 1, 2]
MODELS = {
    "R001": ("TF-IDF+LR", tfidf_lr),
    "R002": ("kNN vote",  knn_vote),
}


def run_once(model_fn, df: pd.DataFrame, run_id: str, model_name: str, seed: int):
    np.random.seed(seed)
    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    clf = model_fn()
    clf.fit(train["claim_text"], train["label"])

    # Evaluate on val and test
    results = {}
    for split_name, split_df in [("val", val), ("test", test)]:
        y_true = split_df["label"].values
        y_pred = clf.predict(split_df["claim_text"])
        y_prob = clf.predict_proba(split_df["claim_text"])[:, 1]
        m = compute_all(y_true, y_pred, y_prob)
        results[split_name] = m
        print(f"  [{split_name}] macro_f1={m['macro_f1']:.4f}  f1_false={m['f1_false']:.4f}  "
              f"pr_auc={m['pr_auc_false']:.4f}  roc_auc={m['roc_auc']:.4f}")

    log_run(RUN_LOG, {
        "run_id":    f"{run_id}_seed{seed}",
        "model":     model_name,
        "dedup":     "Y",
        "split":     "stratified_val",
        "seed":      seed,
        **{k: f"{v:.4f}" for k, v in results["val"].items()},
        "threshold": 0.5,
        "notes":     f"test_macro_f1={results['test']['macro_f1']:.4f}",
    })
    return results


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    print("Hashes:", hashes)

    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)
    print(f"Loaded {len(df)} rows  "
          f"(train={len(df[df['split']=='train'])}  "
          f"val={len(df[df['split']=='val'])}  "
          f"test={len(df[df['split']=='test'])})")

    os.makedirs("runs", exist_ok=True)

    for run_id, (model_name, model_fn) in MODELS.items():
        print(f"\n=== {run_id}: {model_name} ===")
        seed_val_f1s = []
        for seed in SEEDS:
            print(f"  seed {seed}")
            results = run_once(model_fn, df, run_id, model_name, seed)
            seed_val_f1s.append(results["val"]["macro_f1"])

        print(f"  → avg val macro_f1={np.mean(seed_val_f1s):.4f}")
        save_manifest(run_id, config={"model": model_name, "seeds": SEEDS}, hashes=hashes)

    print("\nBaseline evaluation complete. Results in", RUN_LOG)


if __name__ == "__main__":
    main()

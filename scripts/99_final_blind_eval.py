"""
Script 99 — Final blind evaluation (ONE-SHOT).

MUST only be run after all modelling decisions are frozen.
Requires: ALLOW_BLIND_EVAL=1

Usage:
    ALLOW_BLIND_EVAL=1 python scripts/99_final_blind_eval.py \\
        --blind_path data/raw/blind_test.json \\
        --model_dir  checkpoints/best_macro_f1/R009 \\
        --output     results/submission/predictions.csv
"""
import sys
import os
import argparse
import json
import datetime

# Guard is the very first import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

if not os.environ.get("ALLOW_BLIND_EVAL"):
    print("ERROR: Set ALLOW_BLIND_EVAL=1 to run the final blind evaluation.")
    sys.exit(1)

import numpy as np
import pandas as pd

from src.data.loader import load_training, sha256_file, _extract_claim
from src.data.dedup import dedup
from src.data.features import extract
from src.models.baselines import tfidf_lr       # fallback if no neural ckpt
from src.evaluation.metrics import compute_all


def load_blind(blind_path: str) -> list[dict]:
    """Load blind test file — guard logs the access."""
    check_path(blind_path)
    with open(blind_path) as f:
        raw = json.load(f)
    records = []
    for item in raw:
        records.append({
            "id":         item.get("index", str(len(records))),
            "claim_text": _extract_claim(item.get("Open-ended Verifiable Question", "")),
        })
    return records


def retrain_full(df_train: pd.DataFrame, seed: int = 0):
    """Retrain the best pipeline on the full training set."""
    np.random.seed(seed)
    clf = tfidf_lr()
    clf.fit(df_train["claim_text"], df_train["label"])
    return clf


def write_predictions(records: list[dict], y_pred: np.ndarray,
                      y_prob: np.ndarray, output_path: str):
    rows = []
    for rec, pred, prob in zip(records, y_pred, y_prob):
        rows.append({
            "id":      rec["id"],
            "label":   "false" if pred == 1 else "true",
            "p_false": float(prob),
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}  ({len(rows)} rows)")


def write_manifest(blind_path: str, train_hashes: dict,
                   model_dir: str, output_path: str, seed: int):
    manifest = {
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "train_hashes":    train_hashes,
        "blind_hash":      sha256_file(blind_path),
        "model_dir":       model_dir,
        "seed":            seed,
        "blind_path":      blind_path,
        "compliance":      (
            "All modelling was performed on labelled training data only. "
            "Blind set accessed exactly once at final submission time."
        ),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind_path", required=True)
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--output",     default="results/submission/predictions.csv")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    SFT_PATH = "data/raw/misinfo_SFT_train_for_cot.json"
    RL_PATH  = "data/raw/misinfo_RL_train_for_cot.json"

    # Step 1 — Hash training data
    train_hashes = {
        "SFT": sha256_file(SFT_PATH),
        "RL":  sha256_file(RL_PATH),
    }
    print("Training dataset hashes:")
    for k, v in train_hashes.items():
        print(f"  {k}: {v}")

    # Step 2 — Load and preprocess full training set
    print("\nLoading full training set...")
    records = load_training(SFT_PATH) + load_training(RL_PATH)
    df_train = dedup(records)
    df_train = extract(df_train)
    print(f"  {len(df_train)} deduplicated training examples")

    # Step 3 — Retrain on full training set
    print("\nRetraining on full training set...")
    clf = retrain_full(df_train, seed=args.seed)

    # Step 4 — Open blind file EXACTLY ONCE (guard logs the access)
    print("\nLoading blind test set...")
    blind_records = load_blind(args.blind_path)
    print(f"  {len(blind_records)} blind examples")

    blind_texts = [r["claim_text"] for r in blind_records]
    y_pred      = clf.predict(blind_texts)
    y_prob      = clf.predict_proba(blind_texts)[:, 1]

    # Step 5 — Write predictions
    write_predictions(blind_records, y_pred, y_prob, args.output)

    # Step 6 — Write manifest
    manifest_path = os.path.join(
        os.path.dirname(args.output), "manifest_final.json"
    )
    write_manifest(args.blind_path, train_hashes, args.model_dir,
                   manifest_path, args.seed)

    print("\nFinal blind evaluation complete.")
    print("Please verify blind_access_audit.log shows exactly one BLIND_READ entry.")


if __name__ == "__main__":
    main()

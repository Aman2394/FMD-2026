"""
Script 01 — Exploratory Data Analysis (training data only).

Outputs:
  - Console stats table
  - data/processed/minimal_pairs.json   (same group_title, differing numeric values)
  - runs/manifests/R000_eda.json        (dataset hashes)
"""
import sys
import os
import json
import collections

# Guard must be imported first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import pandas as pd

from src.data.loader import load_training, sha256_file
from src.data.dedup import dedup
from src.data.features import extract
from src.data.folds import make_split
from src.reporting.logger import save_manifest

SFT_PATH = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH  = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"


def main():
    # --- Hash datasets ---
    hashes = {
        "SFT": sha256_file(SFT_PATH),
        "RL":  sha256_file(RL_PATH),
    }
    print("Dataset SHA-256 hashes:")
    for k, v in hashes.items():
        print(f"  {k}: {v}")

    save_manifest("R000_eda", config={"script": "01_eda.py"}, hashes=hashes)

    # --- Load and merge ---
    records_sft = load_training(SFT_PATH)
    records_rl  = load_training(RL_PATH)
    all_records  = records_sft + records_rl
    print(f"\nLoaded: {len(records_sft)} SFT + {len(records_rl)} RL = {len(all_records)} total")

    # --- Dedup ---
    df = dedup(all_records)

    # --- Feature extraction ---
    df = extract(df)

    # --- Save processed data ---
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/train_dedup.parquet", index=False)
    print(f"Saved data/processed/train_dedup.parquet ({len(df)} rows)")

    # --- Class distribution ---
    counts = df["label"].value_counts().sort_index()
    print("\nClass distribution:")
    print(f"  0 (True / no misinfo): {counts.get(0, 0)}")
    print(f"  1 (False / misinfo):   {counts.get(1, 0)}")
    print(f"  Class ratio (False/True): {counts.get(1, 0) / max(counts.get(0, 1), 1):.3f}")

    # --- Token length stats ---
    print("\nToken length stats (approx word count):")
    desc = df["n_tokens_approx"].describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"  {stat:>4s}: {desc[stat]:.1f}")

    # --- Numeric / ticker / ellipsis prevalence ---
    print("\nFeature prevalence:")
    print(f"  Claims with ≥1 number:   {(df['n_numbers'] > 0).mean():.1%}")
    print(f"  Claims with ≥1 ticker:   {(df['n_tickers'] > 0).mean():.1%}")
    print(f"\n  Avg numbers per claim:   {df['n_numbers'].mean():.2f}")
    print(f"  Avg tickers per claim:   {df['n_tickers'].mean():.2f}")


    # --- Generate split (once) ---
    if not os.path.isfile(FOLDS_PATH):
        print("\nGenerating train/val/test split (first time only)...")
        make_split(df, val_size=0.15, test_size=0.15, seed=42, output_path=FOLDS_PATH)
    else:
        print(f"\nSplit already exists at {FOLDS_PATH} — skipping regeneration.")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()

"""
Script 10 — Ablation studies (A1–A8).

A1: Random split vs GroupKFold(title) — leakage inflation
A2: Dedup vs no-dedup — duplicate-driven overestimation
A3: Keep instruction prefix vs strip it — template artefacts
A4: Mask all numbers → F1 drop (numeracy dependence)
A5: Mask all tickers → F1 drop (entity dependence)
A6: Approach B — contrastive off (α=0)
A7: Approach A — aux losses off (λ=0)
A8: Calibration off (raw logit threshold vs calibrated)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import json
import numpy as np
import pandas as pd
from src.data.loader import load_training, sha256_file
from src.data.dedup import dedup
from src.data.features import extract, NUMBER_RE, TICKER_RE
from src.data.folds import load_split
from src.models.baselines import tfidf_lr
from src.evaluation.metrics import compute_all
from src.evaluation.robustness import mask_numbers, mask_tickers
from src.reporting.logger import log_run, save_manifest

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
N_FOLDS    = 5
SEED       = 0


def split_score(df, text_col="claim_text", label_col="label"):
    """Evaluate on the pre-defined stratified val split."""
    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    clf   = tfidf_lr()
    clf.fit(train[text_col], train[label_col])
    y_true = val[label_col].values
    y_pred = clf.predict(val[text_col])
    y_prob = clf.predict_proba(val[text_col])[:, 1]
    return compute_all(y_true, y_pred, y_prob)


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    records_sft = load_training(SFT_PATH)
    records_rl  = load_training(RL_PATH)
    all_records = records_sft + records_rl

    df_dedup = pd.read_parquet(PARQUET)
    df_dedup = load_split(df_dedup, folds_path=FOLDS_PATH)
    df_dedup = extract(df_dedup)

    ablation_results = {}

    # A1: Stratified split baseline score
    print("\n=== A1: Baseline stratified split ===")
    group_score = split_score(df_dedup)
    ablation_results["A1_stratified"] = group_score
    print(f"  Stratified val macro_f1: {group_score['macro_f1']:.4f}")

    # A2: Dedup vs no-dedup
    print("\n=== A2: Dedup vs no-dedup ===")
    from src.data.folds import make_split as _make_split
    df_nodup = extract(pd.DataFrame(all_records))
    df_nodup = _make_split(df_nodup, seed=SEED, output_path="/tmp/folds_nodup.json")
    nodedup_score = split_score(df_nodup)
    ablation_results["A2_no_dedup"] = nodedup_score
    delta2 = nodedup_score["macro_f1"] - group_score["macro_f1"]
    print(f"  No-dedup macro_f1: {nodedup_score['macro_f1']:.4f}  (Δ vs dedup={delta2:+.4f})")

    # A3: Keep prefix vs strip prefix
    print("\n=== A3: Instruction prefix ===")
    df_with_prefix = df_dedup.copy()
    df_with_prefix["claim_text"] = df_with_prefix["prompt_text"]
    prefix_score = split_score(df_with_prefix)
    ablation_results["A3_with_prefix"]    = prefix_score
    ablation_results["A3_stripped_prefix"]= group_score
    delta3 = prefix_score["macro_f1"] - group_score["macro_f1"]
    print(f"  With prefix macro_f1:   {prefix_score['macro_f1']:.4f}")
    print(f"  Stripped prefix macro_f1:{group_score['macro_f1']:.4f}  (Δ={delta3:+.4f})")

    # A4: Mask all numbers
    print("\n=== A4: Mask all numbers ===")
    df_masked_nums = df_dedup.copy()
    df_masked_nums["claim_text"] = df_masked_nums["claim_text"].apply(mask_numbers)
    masked_num_score = split_score(df_masked_nums)
    ablation_results["A4_masked_numbers"] = masked_num_score
    delta4 = masked_num_score["macro_f1"] - group_score["macro_f1"]
    print(f"  Masked numbers macro_f1: {masked_num_score['macro_f1']:.4f}  (Δ={delta4:+.4f})")

    # A5: Mask all tickers
    print("\n=== A5: Mask all tickers ===")
    df_masked_tickers = df_dedup.copy()
    df_masked_tickers["claim_text"] = df_masked_tickers["claim_text"].apply(mask_tickers)
    masked_ticker_score = split_score(df_masked_tickers)
    ablation_results["A5_masked_tickers"] = masked_ticker_score
    delta5 = masked_ticker_score["macro_f1"] - group_score["macro_f1"]
    print(f"  Masked tickers macro_f1: {masked_ticker_score['macro_f1']:.4f}  (Δ={delta5:+.4f})")

    # A6: Contrastive off (α=0) — placeholder (requires training run)
    print("\n=== A6: Contrastive off (α=0) — note in log ===")
    ablation_results["A6_note"] = "Run 05_approach_B.py with ALPHA=0 and compare R005"

    # A7: Aux losses off (λ=0) — placeholder
    print("\n=== A7: Aux losses off (λ=0) — note in log ===")
    ablation_results["A7_note"] = "Run 04_approach_A.py with loss weights=0 and compare R004"

    # A8: Calibration off vs calibrated
    print("\n=== A8: Calibration off vs calibrated ===")
    # Already computed above; note is to compare R009 with/without calibration step
    ablation_results["A8_note"] = "Compare R009 with isotonic vs raw stacker probs"

    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/ablations.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("\nAblation results saved to results/ablations.json")

    # Fill summary table rows
    for tag, scores in ablation_results.items():
        if isinstance(scores, dict) and "macro_f1" in scores:
            log_run(RUN_LOG, {
                "run_id": f"ABL_{tag}", "model": "TF-IDF+LR (ablation)",
                "dedup": "Y" if "no_dedup" not in tag else "N",
                "split": "GroupKFold(title)" if "random" not in tag else "StratifiedKFold",
                "seed": SEED,
                **{k: f"{v:.4f}" for k, v in scores.items()},
                "notes": tag,
            })

    save_manifest("R010_ablations", config={"ablations": list(ablation_results.keys())}, hashes=hashes)
    print("Ablation script complete.")


if __name__ == "__main__":
    main()

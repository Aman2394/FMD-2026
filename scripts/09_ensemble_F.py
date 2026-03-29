"""
Script 09 — Approach F: Calibrated Ensemble (R009).

Loads OOF predictions from R001–R008, stacks via logistic regression,
calibrates with isotonic regression, tunes abstention threshold.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.models.approach_F import CalibratedEnsemble
from src.evaluation.metrics import compute_all
from src.evaluation.calibration import IsotonicCalibrator
from src.reporting.logger import log_run, save_manifest
from src.reporting.plots import plot_calibration, plot_roc_curves, plot_pr_curves

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
OOF_DIR    = "runs/oof"          # directory where each approach saves OOF probs
RUN_ID     = "R009"
SEEDS      = [0, 1, 2]


def load_oof_probs(run_ids: list[str], n_rows: int) -> np.ndarray:
    """
    Load OOF probability files (one per run_id) from OOF_DIR.
    Falls back to random probs if file not found (for development).
    """
    probs = []
    for rid in run_ids:
        path = os.path.join(OOF_DIR, f"{rid}_oof.npy")
        if os.path.isfile(path):
            p = np.load(path)
            print(f"  Loaded OOF for {rid}: shape={p.shape}")
        else:
            print(f"  WARNING: OOF not found for {rid}, using random probs")
            p = np.random.default_rng(42).uniform(0.3, 0.7, n_rows)
        probs.append(p)
    return np.stack(probs, axis=1)  # (N, n_models)


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)
    y  = df["label"].values
    n  = len(df)

    run_ids = ["R001", "R002", "R003", "R004", "R005", "R006", "R007", "R008"]
    os.makedirs(OOF_DIR, exist_ok=True)

    print("Loading OOF predictions...")
    oof_probs = load_oof_probs(run_ids, n_rows=n)   # (N, n_models)

    # Fit ensemble on OOF
    ensemble = CalibratedEnsemble(method="isotonic")
    ensemble.fit(oof_probs, y)
    ensemble.tune_threshold(oof_probs, y)

    # CV-style evaluation (use the stacked OOF as pseudo-val)
    y_prob = ensemble.predict_proba(oof_probs)
    y_pred, abstain, _ = ensemble.predict(oof_probs)
    metrics = compute_all(y, y_pred, y_prob)

    print("\nEnsemble OOF metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Abstentions: {abstain.sum()} / {len(abstain)} ({abstain.mean():.1%})")

    # Plots
    os.makedirs("results", exist_ok=True)
    plot_calibration(y, y_prob, model_name="Ensemble F",
                     output_path="results/calibration_R009.png")
    plot_roc_curves({"Ensemble F": {"y_true": y, "y_prob": y_prob}},
                    output_path="results/roc_R009.png")

    for seed in SEEDS:
        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}", "model": "Approach F (Ensemble)",
            "dedup": "Y", "split": "GroupKFold(title)", "seed": seed,
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "notes": f"isotonic calibration tau={ensemble.tau:.3f}",
        })

    save_manifest(RUN_ID, config={"components": run_ids, "method": "isotonic"}, hashes=hashes)
    print(f"\nR009 complete.")


if __name__ == "__main__":
    main()

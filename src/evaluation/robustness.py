"""
Robustness evaluation — perturbation battery.

Each perturbation is applied to the validation set; delta-F1 relative
to the clean baseline is reported per perturbation type.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.features import NUMBER_RE, TICKER_RE, TIME_RE


# ── Perturbation functions ──────────────────────────────────────────────────

def mask_numbers(text: str, mask: str = "[NUM]") -> str:
    return NUMBER_RE.sub(mask, text)


def mask_tickers(text: str, mask: str = "[TICKER]") -> str:
    return TICKER_RE.sub(mask, text)


def mask_time(text: str, mask: str = "[TIME]") -> str:
    return TIME_RE.sub(mask, text)


PERTURBATIONS = {
    "mask_numbers": mask_numbers,
    "mask_tickers": mask_tickers,
    "mask_time":    mask_time,
}


# ── Evaluation harness ──────────────────────────────────────────────────────

def run_perturbation_battery(
    clf,
    val_df: pd.DataFrame,
    text_col: str = "claim_text",
    label_col: str = "label",
    metric: str = "macro_f1",
    seed: int = 42,
) -> dict:
    """
    Evaluate `clf` on clean + perturbed validation texts.

    Args:
        clf:      A fitted sklearn-compatible pipeline with predict / predict_proba.
        val_df:   Validation DataFrame.
        text_col: Column containing text to perturb.
        label_col: Column containing integer labels.
        metric:   "macro_f1" or "f1_false".
        seed:     Random seed for stochastic perturbations.
    Returns:
        Dict mapping perturbation name → delta score vs. clean baseline.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    y_true = val_df[label_col].values
    texts  = val_df[text_col].tolist()

    # Baseline
    y_pred_clean = clf.predict(texts)
    if metric == "macro_f1":
        baseline = float(f1_score(y_true, y_pred_clean, average="macro"))
    else:
        baseline = float(f1_score(y_true, y_pred_clean, pos_label=1))

    results = {"clean": baseline}

    for name, perturb_fn in PERTURBATIONS.items():
        perturbed = [perturb_fn(t) for t in texts]
        y_pred    = clf.predict(perturbed)
        if metric == "macro_f1":
            score = float(f1_score(y_true, y_pred, average="macro"))
        else:
            score = float(f1_score(y_true, y_pred, pos_label=1))
        results[name] = score
        delta = score - baseline
        print(f"  {name:20s}: {score:.4f}  (Δ={delta:+.4f})")

    deltas = {k: v - baseline for k, v in results.items() if k != "clean"}
    deltas["clean"] = baseline
    return deltas

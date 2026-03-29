"""
Statistical testing utilities.

  - paired_bootstrap_ci   : 95% CI and p-value for metric delta between two models
  - mcnemar_test          : McNemar's test on paired binary predictions
  - delong_auc_test       : DeLong AUC comparison (approximation via bootstrap)
  - holm_bonferroni       : Holm-Bonferroni correction for multiple comparisons
  - benjamini_hochberg    : Benjamini-Hochberg FDR correction
"""
import numpy as np
from scipy.stats import chi2


# ── Paired bootstrap ────────────────────────────────────────────────────────

def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn,
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap CI for the difference metric_fn(a) − metric_fn(b).

    Args:
        metric_fn: Callable(y_true, y_prob) → float.
    """
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx = np.random.choice(n, n, replace=True)
        da  = metric_fn(y_true[idx], y_prob_a[idx])
        db  = metric_fn(y_true[idx], y_prob_b[idx])
        deltas.append(da - db)
    deltas = np.array(deltas)
    alpha  = (1 - ci) / 2
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low":     float(np.percentile(deltas, 100 * alpha)),
        "ci_high":    float(np.percentile(deltas, 100 * (1 - alpha))),
        "p_value":    float(np.mean(deltas <= 0)),
    }


# ── McNemar test ─────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test for two classifiers on the same test set.

    Returns p-value (continuity-corrected).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong,  B correct
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / n     # continuity correction
    p_value   = float(1 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


# ── DeLong AUC (bootstrap approximation) ────────────────────────────────────

def delong_auc_test(y_true: np.ndarray,
                    y_prob_a: np.ndarray,
                    y_prob_b: np.ndarray,
                    n_resamples: int = 2000) -> dict:
    """Bootstrap approximation to DeLong AUC comparison."""
    from sklearn.metrics import roc_auc_score
    return paired_bootstrap_ci(
        y_true, y_prob_a, y_prob_b,
        metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
        n_resamples=n_resamples,
    )


# ── Multiple comparison corrections ─────────────────────────────────────────

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the original order.
    """
    n      = len(p_values)
    order  = np.argsort(p_values)
    sorted_p = [p_values[i] for i in order]
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        adj = sorted_p[rank] * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def benjamini_hochberg(p_values: list[float],
                       alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (adjusted_p_values, reject_mask).
    """
    n     = len(p_values)
    order = np.argsort(p_values)
    adj   = [0.0] * n
    for rank in range(n - 1, -1, -1):
        orig_idx = order[rank]
        adj[orig_idx] = min(
            p_values[orig_idx] * n / (rank + 1),
            adj[order[rank + 1]] if rank < n - 1 else 1.0,
        )
        adj[orig_idx] = min(adj[orig_idx], 1.0)
    reject = [a < alpha for a in adj]
    return adj, reject

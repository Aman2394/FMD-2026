"""
Plotting utilities — ROC/PR curves, calibration diagrams, robustness curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_roc_curves(results: dict, output_path: str = "results/roc_curves.png"):
    """
    Args:
        results: {model_name: {"y_true": ..., "y_prob": ...}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d in results.items():
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_pr_curves(results: dict, output_path: str = "results/pr_curves.png"):
    """
    Args:
        results: {model_name: {"y_true": ..., "y_prob": ...}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d in results.items():
        prec, rec, _ = precision_recall_curve(d["y_true"], d["y_prob"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (class=False/misinfo)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     model_name: str = "",
                     n_bins: int = 10,
                     output_path: str = "results/calibration.png"):
    """Reliability diagram."""
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_means.append(y_prob[mask].mean())
        bin_accs.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bin_means, bin_accs, "o-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration diagram — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_robustness_deltas(deltas: dict,
                           model_name: str = "",
                           output_path: str = "results/robustness.png"):
    """
    Bar chart of delta-F1 for each perturbation type.

    Args:
        deltas: {perturbation_name: delta_value}  (excluding "clean" key)
    """
    names  = [k for k in deltas if k != "clean"]
    values = [deltas[k] for k in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#d62728" if v < 0 else "#2ca02c" for v in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔMacro-F1 vs. clean baseline")
    ax.set_title(f"Robustness — {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")

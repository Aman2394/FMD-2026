import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, auc,
    brier_score_loss,
)


def compute_all(y_true, y_pred, y_prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    macro_f1        = f1_score(y_true, y_pred, average="macro")
    f1_false        = f1_score(y_true, y_pred, pos_label=1)  # 1 = False (misinfo)
    roc_auc         = roc_auc_score(y_true, y_prob)
    prec, rec, _    = precision_recall_curve(y_true, y_prob)
    pr_auc          = auc(rec, prec)
    brier           = brier_score_loss(y_true, y_prob)
    ece             = _ece(y_true, y_prob, n_bins)

    return dict(
        macro_f1=macro_f1,
        f1_false=f1_false,
        roc_auc=roc_auc,
        pr_auc_false=pr_auc,
        brier=brier,
        ece=ece,
    )


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)

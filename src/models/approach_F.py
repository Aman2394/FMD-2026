"""
Approach F — Uncertainty-Aware Calibrated Ensemble.

Stacks out-of-fold (OOF) predictions from component models A–E
via a logistic regression meta-learner, then calibrates with
isotonic regression. Supports abstention when confidence is low.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class CalibratedEnsemble:
    """Stacks OOF predictions from components A–E; calibrates via isotonic regression."""

    def __init__(self, temperature_init: float = 1.0,
                 method: str = "isotonic"):
        self.method  = method
        self.stacker = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.cal     = (IsotonicRegression(out_of_bounds="clip")
                        if method == "isotonic" else None)
        self.tau     = 0.5      # abstention threshold (tune on CV)

    def fit(self, oof_probs: np.ndarray, y: np.ndarray):
        """
        Args:
            oof_probs: (N, n_models) — OOF probabilities from each component.
            y:         (N,) ground-truth labels.
        """
        self.stacker.fit(oof_probs, y)
        stacked = self.stacker.predict_proba(oof_probs)[:, 1]
        if self.cal:
            self.cal.fit(stacked, y)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Args:
            probs: (N, n_models) prediction probabilities.
        Returns:
            (N,) calibrated probability of class 1 (misinfo).
        """
        stacked = self.stacker.predict_proba(probs)[:, 1]
        if self.cal:
            stacked = self.cal.transform(stacked)
        return stacked

    def predict(self, probs: np.ndarray):
        """
        Returns:
            pred:    (N,) predicted labels (0 or 1).
            abstain: (N,) boolean mask — True when confidence below tau.
            p:       (N,) calibrated probabilities.
        """
        p       = self.predict_proba(probs)
        pred    = np.where(p >= 0.5, 1, 0)
        abstain = np.abs(p - 0.5) < (0.5 - self.tau)
        return pred, abstain, p

    def tune_threshold(self, oof_probs: np.ndarray, y: np.ndarray,
                       tau_grid: np.ndarray = None) -> float:
        """
        Select abstention threshold tau that maximises macro-F1 on OOF data.

        Args:
            oof_probs: (N, n_models) OOF probabilities.
            y:         (N,) ground-truth labels.
            tau_grid:  Thresholds to try. Defaults to linspace(0.3, 0.5, 20).
        Returns:
            Best tau value.
        """
        from sklearn.metrics import f1_score

        if tau_grid is None:
            tau_grid = np.linspace(0.3, 0.5, 20)

        p = self.predict_proba(oof_probs)
        best_tau, best_f1 = 0.5, -1.0
        for tau in tau_grid:
            pred = np.where(p >= 0.5, 1, 0)
            f1   = f1_score(y, pred, average="macro")
            if f1 > best_f1:
                best_f1  = f1
                best_tau = tau

        self.tau = best_tau
        print(f"Best tau={best_tau:.3f} → macro_f1={best_f1:.4f}")
        return best_tau

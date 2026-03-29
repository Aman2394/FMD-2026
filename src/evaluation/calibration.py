"""
Calibration utilities — temperature scaling and isotonic regression.

All calibration fitting must use CV splits only (never the blind set).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression


class TemperatureScaling(nn.Module):
    """Scalar temperature applied to logits before softmax."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: np.ndarray, y: np.ndarray,
            lr: float = 0.01, n_iter: int = 100) -> float:
        """
        Optimise temperature to minimise NLL on held-out logits.

        Args:
            logits: (N, 2) raw logits.
            y:      (N,) integer labels.
        Returns:
            Fitted temperature value.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t      = torch.tensor(y, dtype=torch.long)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature
            loss   = nn.CrossEntropyLoss()(scaled, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


class IsotonicCalibrator:
    """Isotonic regression calibrator (CV-only fitting)."""

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        self.ir.fit(y_prob, y_true)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return self.ir.transform(y_prob)

    def fit_transform(self, y_prob: np.ndarray,
                      y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)

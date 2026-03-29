"""
Approach C — Fold-Safe kNN Memory + Meta-Calibrator.

A NearestNeighbors index is built exclusively on the training fold.
Retrieval features (vote, entropy, margin, max_sim, mean_sim) are fed
into a meta logistic-regression calibrator.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FoldSafeMemory:
    """kNN retrieval restricted to the training fold only."""
    def __init__(self, k: int = 15):
        self.k     = k
        self.index = None
        self.y     = None

    def build(self, X_train: np.ndarray, y_train: np.ndarray):
        self.index = NearestNeighbors(
            n_neighbors=self.k, metric="cosine"
        ).fit(X_train)
        self.y = y_train

    def neighbour_features(self, X_query: np.ndarray) -> np.ndarray:
        distances, indices = self.index.kneighbors(X_query)
        feats = []
        for dist, idx in zip(distances, indices):
            sim    = 1 - dist
            nbr_y  = self.y[idx]
            vote   = nbr_y.mean()
            entropy= -(vote * np.log(vote + 1e-9) + (1 - vote) * np.log(1 - vote + 1e-9))
            margin = (
                sim[nbr_y == 1].mean() - sim[nbr_y == 0].mean()
                if len(np.unique(nbr_y)) > 1
                else 0.0
            )
            feats.append([vote, entropy, margin, sim.max(), sim.mean()])
        return np.array(feats)


class KNNCalibrator:
    def __init__(self, k: int = 15):
        self.memory = FoldSafeMemory(k)
        self.scaler = StandardScaler()
        self.meta   = LogisticRegression(class_weight="balanced", max_iter=1000)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        self.memory.build(X_train, y_train)
        feats_val = self.memory.neighbour_features(X_val)
        feats_val = self.scaler.fit_transform(feats_val)
        self.meta.fit(feats_val, y_val)

    def predict_proba(self, X_query: np.ndarray,
                      X_train: np.ndarray = None,
                      y_train: np.ndarray = None) -> np.ndarray:
        if X_train is not None:
            self.memory.build(X_train, y_train)
        feats = self.memory.neighbour_features(X_query)
        feats = self.scaler.transform(feats)
        return self.meta.predict_proba(feats)

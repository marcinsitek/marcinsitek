import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class FractionalDifferentiation(BaseEstimator, TransformerMixin):
    def __init__(self, d: float):
        self.d: float = d

    def _calculate_weights(self, d: float, K: int) -> np.ndarray:
        weights = [1.0]
        k = 1
        while k < K:
            w_k = -weights[-1] * (d - k + 1) / k
            weights.append(w_k)
            k += 1
        weights = np.array(weights).flatten()
        return weights

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=False)
        return self

    def transform(self, X: np.ndarray):
        y = check_array(X, ensure_2d=False)
        y = np.flip(y[1:])
        weights = self._calculate_weights(self.d, len(y))
        T = len(y)
        y_diff = []
        for t in range(0, T):
            y_before_t = y[t:]
            w_before_t = weights[: T - t]
            y_t = np.dot(w_before_t, y_before_t.T)
            y_diff.append(y_t)
        y_diff = np.array(y_diff, dtype=float)
        y_diff = np.append(y_diff, np.nan)
        y_diff = np.flip(y_diff)
        return y_diff

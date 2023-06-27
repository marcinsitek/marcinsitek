import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class Saturation(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        col_0_b: float = None,
        col_0_a: float = None,
        col_1_b: float = None,
        col_1_a: float = None,
        col_2_a: float = None,
        col_2_b: float = None,
        col_3_a: float = None,
        col_3_b: float = None,
        col_4_a: float = None,
        col_4_b: float = None,
        col_5_a: float = None,
        col_5_b: float = None,
    ):
        self.col_0_a = col_0_a
        self.col_0_b = col_0_b
        self.col_1_a = col_1_a
        self.col_1_b = col_1_b
        self.col_2_a = col_2_a
        self.col_2_b = col_2_b
        self.col_3_a = col_3_a
        self.col_3_b = col_3_b
        self.col_4_a = col_4_a
        self.col_4_b = col_4_b
        self.col_5_a = col_5_a
        self.col_5_b = col_5_b

    def _hill(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return x**a / (x**a + b**a)

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        X_saturated = np.zeros_like(X)

        for j in range(0, 6):
            X_saturated[:, j] = self._hill(
                X[:, j],
                a=getattr(self, f"col_{j}_a"),
                b=getattr(self, f"col_{j}_b"),
            )
        return X_saturated

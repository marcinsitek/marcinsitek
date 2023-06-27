import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class Saturation(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        col_0_k: float = None,
        col_0_n: float = None,
        col_1_k: float = None,
        col_1_n: float = None,
        col_2_k: float = None,
        col_2_n: float = None,
        col_3_k: float = None,
        col_3_n: float = None,
        col_4_k: float = None,
        col_4_n: float = None,
        col_5_k: float = None,
        col_5_n: float = None,
    ):
        self.col_0_k = col_0_k
        self.col_0_n = col_0_n
        self.col_1_k = col_1_k
        self.col_1_n = col_1_n
        self.col_2_k = col_2_k
        self.col_2_n = col_2_n
        self.col_3_k = col_3_k
        self.col_3_n = col_3_n
        self.col_4_k = col_4_k
        self.col_4_n = col_4_n
        self.col_5_k = col_5_k
        self.col_5_n = col_5_n

    def _hill(self, x: np.ndarray, k: float, n: float) -> np.ndarray:
        return 1 / (1 + (x / k) ** -n)

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
                k=getattr(self, f"col_{j}_k"),
                n=getattr(self, f"col_{j}_n"),
            )
        return X_saturated

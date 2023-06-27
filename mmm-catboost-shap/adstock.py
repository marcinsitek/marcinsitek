import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class Adstock(BaseEstimator, TransformerMixin):
    def __init__(self, theta=0.5):
        self.theta = theta

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        x_decayed = np.zeros_like(X)
        x_decayed[0] = X[0]

        for xi in range(1, len(x_decayed)):
            # x_decayed[xi] = X[xi] + self.theta * x_decayed[xi - 1]
            x_decayed[xi] = X[xi] + self.theta * X[xi - 1]
        return x_decayed

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import statsmodels.api as sm


class Detrender(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.coef_ = None

    def get_summary(self):
        check_is_fitted(self)
        return self.results.summary()

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=False)
        time = np.arange(1, len(X)+1)
        time_squared = time ** 2
        Z = np.stack([time, time_squared], axis=1)
        Z = sm.add_constant(Z)
        model = sm.OLS(X, Z)
        self.results = model.fit()
        self.coef_ = self.results.params
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        time = np.arange(1, len(X)+1)
        time_squared = time ** 2
        return X - (
            self.coef_[0] 
            + self.coef_[1] * time 
            + self.coef_[2] * time_squared
        )


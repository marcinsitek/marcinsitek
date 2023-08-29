import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel(BaseEstimator):
    def __init__(self, p: int, d: int, q: int) -> None:
        self.p = p
        self.d = d
        self.q = q

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None) -> "ARIMAModel":
        if exog is not None:
            mod = ARIMA(y, exog, order=(self.p, self.d, self.q), trend="n")
        else:
            mod = ARIMA(y, order=(self.p, self.d, self.q), trend="n")
        self.res = mod.fit()
        return self
    
    def predict(self, exog: Optional[np.ndarray] = None) -> float:
        if exog is not None:
            return self.res.forecast(steps=1, exog=exog).item()
        else:
            return self.res.forecast(steps=1).item()
    
    def get_aic(self) -> float:
        assert self.res is not None
        return self.res.aic
    
    def get_summary(self) -> None:
        assert self.res is not None
        return self.res.summary()
    
    def get_residuals(self) -> np.ndarray:
        assert self.res is not None
        return self.res.resid
    
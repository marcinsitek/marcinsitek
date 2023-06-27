from typing import Dict, List

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.backends.base import MultiTrace
from sklearn.base import BaseEstimator


class BayesianModel:
    def __init__(
        self,
        media_variables: List[str] = None,
        control_variables: List[str] = None,
        fit_intercept: bool = True,
    ):
        self.media_variables = media_variables
        self.control_variables = control_variables
        self.variables = self.media_variables + self.control_variables
        self.fit_intercept = fit_intercept
        self.model = pm.Model()
        self.trace: MultiTrace = None
        self.coefficients = {}

    def _adstock(self, x, theta: float = 0.0, l: int = 12):
        cycles = [tt.concatenate([tt.zeros(i), x[: x.shape[0] - i]]) for i in range(l)]
        x_cycle = tt.stack(cycles)
        w = tt.as_tensor_variable([tt.power(theta, i) for i in range(l)])
        return tt.dot(w, x_cycle)

    @staticmethod
    def hill(x, k, n):
        return 1 / (1 + (x / k) ** -n)

    def _get_coefficients(self, trace):
        coefficients = {}
        for var in self.variables:
            coefficients[var] = trace.posterior[f"beta_{var}"].values.mean()
        if self.fit_intercept:
            coefficients["intercept"] = trace.posterior["intercept"].values.mean()
        return coefficients

    def _get_posterior_params(self, trace):
        posterior_params = {}
        for var in self.media_variables:
            posterior_params[f"{var}_k"] = trace.posterior[f"k_{var}"].values.mean()
            posterior_params[f"{var}_n"] = trace.posterior[f"n_{var}"].values.mean()
            posterior_params[f"{var}_theta"] = trace.posterior[
                f"theta_{var}"
            ].values.mean()
        return posterior_params

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y = y.copy()
        mu = []
        with self.model:
            # mu - intercept
            if self.fit_intercept:
                intercept = pm.Beta("intercept", alpha=2, beta=5)
                mu.append(intercept)

            # mu - media variables
            for var in self.media_variables:
                # Data
                X_vector = pm.Data(f"{var}", X.loc[:, var].values)
                # Regression coefficient
                beta = pm.Beta(f"beta_{var}", alpha=2, beta=5)
                # Adstock coefficient
                theta = pm.Beta(f"theta_{var}", alpha=1, beta=3)
                # Saturation coefficients
                k = pm.Gamma(f"k_{var}", alpha=3, beta=2)
                n = pm.Gamma(f"n_{var}", alpha=2, beta=2)
                mu.append(
                    beta
                    * BayesianModel.hill(
                        self._adstock(x=X_vector, theta=theta, l=1),
                        k,
                        n,
                    )
                )

            # mu - control variables
            for var in self.control_variables:
                # Data
                X_vector = pm.Data(f"{var}", X.loc[:, var].values)
                # Regression coefficient
                beta = pm.Beta(f"beta_{var}", alpha=2, beta=5)
                mu.append(beta * X_vector)

            # Noise
            sigma = pm.Exponential("sigma", lam=0.0001)

            # Likelihood of observations
            pm.Normal("target", mu=sum(mu), sigma=sigma, observed=y.values)

            self.trace = pm.sample(
                tune=500,
                draws=500,
                chains=2,
                target_accept=0.9,
                return_inferencedata=True,
            )
            self.coefficients = self._get_coefficients(self.trace)
            self.posterior_params = self._get_posterior_params(self.trace)

    def predict(self, X: pd.DataFrame):
        with self.model:
            # mu - media variables
            for var in self.media_variables:
                pm.set_data({f"{var}": X.loc[:, var].values})

            # mu - control variables
            for var in self.control_variables:
                pm.set_data({f"{var}": X.loc[:, var].values})

            posterior = pm.sample_posterior_predictive(self.trace)
        y_hat = posterior["target"].mean(axis=0)
        return y_hat

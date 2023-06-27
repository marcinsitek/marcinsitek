import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from catboost import CatBoostRegressor
from typing import List


class SHAPAttribution(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model: CatBoostRegressor,
        media_variables: List[str],
        control_variables: List[str],
    ):
        self.model = model
        self.media_variables = media_variables
        self.control_variables = control_variables

    def _transform_shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy().add(df.min(axis=0).abs(), axis=1).round()
        df_transformed["Organic"] = df_transformed[self.control_variables].sum(axis=1)
        for c in self.media_variables:
            # shift SHAP values by mean when spend == 0
            if not (self.mask_zero_spend[c] == False).all():
                shift = df_transformed.loc[self.mask_zero_spend[c], c].mean()
                df_transformed[c] = df_transformed[c] - shift
                # realocate SHAP values to Organic when spend == 0
                df_transformed["Organic"] = np.where(
                    self.mask_zero_spend[c],
                    df_transformed["Organic"] + df_transformed[c],
                    df_transformed["Organic"],
                )
            # set SHAP values to 0 when spend == 0
            df_transformed[c] = np.where(
                self.mask_zero_spend[c],
                0,
                df_transformed[c],
            )
            # realocate SHAP values to Organic when shap_value < 0
            df_transformed["Organic"] = np.where(
                df_transformed[c] < 0,
                df_transformed["Organic"] + df_transformed[c],
                df_transformed["Organic"],
            )
            df_transformed[c] = np.where(
                df_transformed[c] < 0,
                0,
                df_transformed[c],
            )
        # set SHAP values to 0 for Organic if shap_value < 0
        df_transformed["Organic"] = np.where(
            df_transformed["Organic"] < 0,
            0,
            df_transformed["Organic"],
        )
        df_transformed.drop(self.control_variables, axis=1, inplace=True)
        return df_transformed

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X):
        check_is_fitted(self.model)
        explainer = shap.TreeExplainer(self.model)
        self.mask_zero_spend = X[self.media_variables] == 0
        shap_values = explainer.shap_values(X)
        shap_values_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        return self._transform_shap_values(shap_values_df)

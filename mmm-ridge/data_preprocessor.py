from pathlib import Path
from typing import Tuple, List

# import numpy as np
import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(
        self,
        data: pd.DataFrame = None,
        media_vars: List[str] = None,
        control_vars: List[str] = None,
        target: str = None,
    ) -> None:
        self.data = data
        self.media_vars = media_vars
        self.control_vars = control_vars
        self.target = target

    def _add_binary_months(
        self, df: pd.DataFrame = None, dt_column: str = "dt"
    ) -> pd.DataFrame:
        df[dt_column] = pd.to_datetime(df[dt_column], format="%Y-%m-%d")
        df["month"] = df[dt_column].dt.month
        df = df.join(pd.get_dummies(df["month"], prefix="month"))
        df.drop("month", axis=1, inplace=True)
        return df

    def _add_binary_events(
        self, df: pd.DataFrame = None, events_column: str = "events"
    ) -> pd.DataFrame:
        df["event1"] = np.where(df[events_column] == "event1", 1, 0)
        df["event2"] = np.where(df[events_column] == "event2", 1, 0)
        df.drop("events", axis=1, inplace=True)
        return df

    def _preprocess_data(self) -> pd.DataFrame:
        assert self.data is not None
        # df = self._add_binary_months(self.data)
        df = self._add_binary_events(self.data)
        df.set_index("dt", inplace=True)
        return df

    def compute_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._preprocess_data()
        assert df is not None
        X = df.loc[:, self.media_vars + self.control_vars].copy()
        y = df.loc[:, self.target].copy()
        return X, y

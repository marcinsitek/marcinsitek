import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        df['value'] = df['value'].astype(float)
        return df.pivot(columns=['series'], index='date', values='value')
    
    def _add_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            df[f"ln_{c}"] = np.log(df[f"{c}"])
            df[f"d_ln_{c}"] = df[f"{c}"].diff()
            df[f"l_d_ln_{c}"] = df[f"d_ln_{c}"].shift(1)
        # df = df.reset_index().set_index('date').asfreq('M')
        # df = df.reset_index()
        df.index = pd.DatetimeIndex(df.index).to_period('M')
        return df

    def preprocess(self) -> pd.DataFrame:
        df = self._pivot(self.df)
        df = self._add_transformations(df)
        df = df.dropna()
        return df
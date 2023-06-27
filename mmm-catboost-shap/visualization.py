import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def attribution_plot(df: pd.DataFrame, target: str, actual: pd.Series = None) -> None:
    COLORS = ["r", "b", "y", "g", "k", "m", "b"]

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(20, 5))

    x = df.index.values
    bottom = np.zeros(len(df))
    for column, color in zip(df.columns, COLORS):
        y = df[column].values
        plt.bar(x, y, bottom=bottom, color=color, width=0.8, label=column)
        bottom += y
    plt.legend()
    if actual is not None:
        plt.plot(x, actual.values, "k-", label="actual")
    step = int(math.ceil(len(df) / 80))

    plt.xlabel("dates")
    plt.xticks(x[::step], rotation=45)
    plt.ylabel(f"{target}")
    plt.legend()
    plt.title("Attribution")
    plt.show()

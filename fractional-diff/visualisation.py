import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

sns.set_theme(style="darkgrid")

def plot_two_series(
        x: np.ndarray, y1: np.ndarray, y2: np.ndarray, y1_label: str, y2_label: str
) -> None:
    _, ax1 = plt.subplots(figsize=(20, 8), dpi=300)
    step = int(math.ceil(len(x) / 40))
    line1, = ax1.plot(x, y1, '-', color='darkorange', label=y1_label)
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, y2, '--', color='dimgray', label=y2_label)
    plt.xticks(x[::step])
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)
    ax1.set_xlabel("dt")
    plt.legend(handles=[line1, line2])
    plt.show()

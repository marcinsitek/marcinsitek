import numpy as np
import math


# Performance metrics
# R2
def r_squared(y: np.array, y_hat: np.array) -> float:
    total_ss = ((y - y.mean()) ** 2).sum()
    residual_ss = ((y - y_hat) ** 2).sum()
    return 1 - (residual_ss / total_ss)


# RMSE
def rmse(y: np.array, y_hat: np.array) -> float:
    mse = ((y - y_hat) ** 2).mean()
    return math.sqrt(mse)


# NRMSE
def nrmse(y: np.array, y_hat: np.array) -> float:
    mse = ((y - y_hat) ** 2).mean()
    return math.sqrt(mse) / (y.max() - y.min())


# MAPE
def mape(y: np.array, y_hat: np.array) -> float:
    return np.mean(np.abs((y - y_hat) / y))


# SMAPE
def smape(y: np.array, y_hat: np.array) -> float:
    return np.mean(np.abs(y_hat - y) / ((np.abs(y) + np.abs(y_hat)) / 2))

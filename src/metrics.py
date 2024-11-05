"""
Metrics for project.
"""

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
)

from src.utils import ShapeError


def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    if not 1 == len(y_true.shape) == len(y_pred.shape):
        raise ShapeError(y_true.shape, ("N",))
    return (y_pred - y_true).squared().mean().sqrt() / abs(y_true).mean()


def normalized_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    if not 1 == len(y_true.shape) == len(y_pred.shape):
        raise ShapeError(y_true.shape, ("N",))
    return abs(y_pred - y_true).sum() / abs(y_pred).sum()


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "msle": mean_squared_log_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "merr": max_error(y_true, y_pred),
        "evar": explained_variance_score(y_true, y_pred),
        "ndev": normalized_deviation(y_true, y_pred),
        "nrmse": normalized_root_mean_squared_error(y_true, y_pred),
    }

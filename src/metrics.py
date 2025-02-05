"""
Metrics for project.
"""

from typing import Callable, Literal
import warnings

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
    return np.sqrt(np.square((y_pred - y_true)).mean()) / abs(y_true).mean()


def normalized_deviation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    if not 1 == len(y_true.shape) == len(y_pred.shape):
        raise ShapeError(y_true.shape, ("N",))
    return abs(y_pred - y_true).sum() / abs(y_pred).sum()


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, errors: Literal["raise", "warn", "nan", "ignore"] = "nan") -> dict[str, float]:

    SENTINAL = "IGNORE"

    def f(func: Callable[[np.ndarray, np.ndarray], float]) -> float:
        try:
            return func(y_true, y_pred)
        except Exception as e:  # pylint: disable=broad-except
            if errors == "raise":
                raise e
            if errors == "warn":
                warnings.warn(str(e))
                return np.nan
            if errors == "nan":
                return np.nan
            if errors == "ignore":
                return SENTINAL
            raise ValueError(f"Invalid value for 'errors': {errors}") from e

    d = {
        "r2": r2_score,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "msle": mean_squared_log_error,
        "mape": mean_absolute_percentage_error,
        "medae": median_absolute_error,
        "merr": max_error,
        "evar": explained_variance_score,
        "ndev": normalized_deviation,
        "nrmse": normalized_root_mean_squared_error,
    }
    d = {k: f(func) for k, func in d.items()}
    d = {k: float(v) for k, v in d.items() if v != SENTINAL}
    return d

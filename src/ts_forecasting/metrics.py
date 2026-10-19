"""Point and interval accuracy metrics for one series and one forecast window."""

from __future__ import annotations

import numpy as np


def mase_scale(y_train: np.ndarray, season: int) -> float:
    """In-sample seasonal-naive MAE (falls back to naive MAE, then 1, for flat histories)."""
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale == 0:
        scale = np.mean(np.abs(np.diff(y_train))) or 1.0
    return float(scale)


def mase(y: np.ndarray, y_hat: np.ndarray, y_train: np.ndarray, season: int) -> float:
    """Mean absolute scaled error; the scale is the in-sample seasonal-naive MAE."""
    return float(np.mean(np.abs(y - y_hat)) / mase_scale(y_train, season))


def smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Symmetric MAPE in percent (M4 definition, 0-200)."""
    denom = np.abs(y) + np.abs(y_hat)
    ratio = np.divide(2 * np.abs(y - y_hat), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(100 * np.mean(ratio))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of actuals inside the prediction interval."""
    return float(np.mean((y >= lo) & (y <= hi)))


def msis(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, scale: float, level: int) -> float:
    """Mean scaled interval score (M4): interval width plus 2/alpha times each miss, divided
    by the MASE scale. Rewards narrow intervals that still cover; lower is better."""
    alpha = 1 - level / 100
    penalty = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(hi - lo + penalty) / scale)

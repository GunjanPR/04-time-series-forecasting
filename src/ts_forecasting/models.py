"""Forecasting methods behind one interface.

Each model takes the training rows of many series (long format) and returns, per series, the
next `h` point forecasts and prediction intervals: columns `unique_id, step, mean,
lo-<level>, hi-<level>` for each requested level.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm


class Forecaster(Protocol):
    name: str

    def forecast(
        self, train: pd.DataFrame, h: int, season: int, levels: tuple[int, ...]
    ) -> pd.DataFrame: ...


def _with_intervals(
    uid: str, mean: np.ndarray, sigma_h: np.ndarray, levels: tuple[int, ...]
) -> pd.DataFrame:
    out = {"unique_id": uid, "step": np.arange(1, len(mean) + 1), "mean": mean}
    for level in levels:
        z = norm.ppf(0.5 + level / 200)
        out[f"lo-{level}"] = mean - z * sigma_h
        out[f"hi-{level}"] = mean + z * sigma_h
    return pd.DataFrame(out)


class SeasonalNaive:
    """Repeat the last observed season. Intervals follow Hyndman & Athanasopoulos (FPP3, 5.5):
    sigma_h = sigma * sqrt(k + 1), k = floor((h - 1) / m), sigma from seasonal-naive residuals."""

    name = "snaive"

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = []
        for uid, group in train.groupby("unique_id", sort=False):
            y = group["y"].to_numpy(float)
            last = y[-season:]
            mean = np.array([last[i % season] for i in range(h)])
            resid = y[season:] - y[:-season]
            sigma = np.sqrt(np.mean(resid**2))
            k = np.floor(np.arange(h) / season)
            frames.append(_with_intervals(uid, mean, sigma * np.sqrt(k + 1), levels))
        return pd.concat(frames, ignore_index=True)


class Naive:
    """Repeat the last value. sigma_h = sigma * sqrt(h)."""

    name = "naive"

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = []
        for uid, group in train.groupby("unique_id", sort=False):
            y = group["y"].to_numpy(float)
            sigma = np.sqrt(np.mean(np.diff(y) ** 2))
            mean = np.full(h, y[-1])
            frames.append(_with_intervals(uid, mean, sigma * np.sqrt(np.arange(1, h + 1)), levels))
        return pd.concat(frames, ignore_index=True)


MODELS: dict[str, type] = {m.name: m for m in (SeasonalNaive, Naive)}

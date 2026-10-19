"""Rolling-origin cross-validation with the same cutoffs for every model.

With `n_windows` windows of horizon `h`, window `w` (0 = oldest) trains on everything before
the last `(n_windows - w) * h` points of each series and tests on the next `h` points. The
last window therefore tests on the final `h` points of every series.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ts_forecasting.metrics import coverage, mase, rmse, smape
from ts_forecasting.models import Forecaster


def rolling_origin_splits(
    df: pd.DataFrame, h: int, n_windows: int
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield `(window, train, test)`. `df` must be sorted by `unique_id, ds`."""
    idx = df.groupby("unique_id", sort=False).cumcount()
    n = df.groupby("unique_id", sort=False)["y"].transform("size")
    for w in range(n_windows):
        cutoff = n - (n_windows - w) * h
        train = df[idx < cutoff]
        test = df[(idx >= cutoff) & (idx < cutoff + h)]
        yield w, train, test


def run_backtest(
    df: pd.DataFrame,
    model: Forecaster,
    h: int,
    n_windows: int,
    season: int,
    levels: tuple[int, ...] = (80, 95),
) -> pd.DataFrame:
    """Score one model on every series and window. One row per (series, window)."""
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rows = []
    for window, train, test in rolling_origin_splits(df, h, n_windows):
        start = time.perf_counter()
        fc = model.forecast(train, h, season, levels)
        seconds = time.perf_counter() - start
        n_series = train["unique_id"].nunique()
        fc = fc.set_index(["unique_id", "step"])
        train_y = {uid: g.to_numpy(float) for uid, g in train.groupby("unique_id")["y"]}
        for uid, group in test.groupby("unique_id", sort=False):
            y = group["y"].to_numpy(float)
            f = fc.loc[uid]
            row = {
                "model": model.name,
                "unique_id": uid,
                "window": window,
                "mase": mase(y, f["mean"].to_numpy(), train_y[uid], season),
                "smape": smape(y, f["mean"].to_numpy()),
                "rmse": rmse(y, f["mean"].to_numpy()),
                "seconds_per_series": seconds / n_series,
            }
            for level in levels:
                row[f"coverage_{level}"] = coverage(
                    y, f[f"lo-{level}"].to_numpy(), f[f"hi-{level}"].to_numpy()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def leaderboard(scores: pd.DataFrame, baseline: str = "snaive") -> pd.DataFrame:
    """Mean metrics per model, plus MASE relative to the baseline (lower is better)."""
    board = scores.groupby("model").agg(
        mase=("mase", "mean"),
        smape=("smape", "mean"),
        rmse=("rmse", "mean"),
        **{c: (c, "mean") for c in scores.columns if c.startswith("coverage_")},
        ms_per_series=("seconds_per_series", lambda s: 1000 * np.mean(s)),
        n=("mase", "size"),
    )
    if baseline in board.index:
        board.insert(1, "rel_mase", board["mase"] / board.loc[baseline, "mase"])
    return board.sort_values("mase")

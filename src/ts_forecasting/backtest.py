"""Rolling-origin cross-validation with the same cutoffs for every model.

With `n_windows` windows of horizon `h`, window `w` (0 = oldest) trains on everything before
the last `(n_windows - w) * h` points of each series and tests on the next `h` points. The
last window therefore tests on the final `h` points of every series.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ts_forecasting.metrics import coverage, mase, mase_scale, msis, rmse, smape
from ts_forecasting.models import Forecaster

log = logging.getLogger(__name__)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest one model on every series and window.

    Returns `(scores, forecasts)`: `scores` has one row per (series, window) with MASE, sMAPE,
    RMSE, interval coverage, MSIS and runtime; `forecasts` has one row per (series, window,
    step) with the actual, the forecast, its intervals and the series' MASE scale, which the
    significance tests use.
    """
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    rows, paths = [], []
    for window, train, test in rolling_origin_splits(df, h, n_windows):
        start = time.perf_counter()
        fc = model.forecast(train, h, season, levels)
        seconds = time.perf_counter() - start
        log.info("%s window %d/%d: %.1fs", model.name, window + 1, n_windows, seconds)
        n_series = train["unique_id"].nunique()
        fc = fc.set_index(["unique_id", "step"])
        train_y = {uid: g.to_numpy(float) for uid, g in train.groupby("unique_id")["y"]}
        for uid, group in test.groupby("unique_id", sort=False):
            y = group["y"].to_numpy(float)
            f = fc.loc[uid]
            y_hat = f["mean"].to_numpy()
            scale = mase_scale(train_y[uid], season)
            row = {
                "model": model.name,
                "unique_id": uid,
                "window": window,
                "mase": mase(y, y_hat, train_y[uid], season),
                "smape": smape(y, y_hat),
                "rmse": rmse(y, y_hat),
                "seconds_per_series": seconds / n_series,
            }
            for level in levels:
                lo, hi = f[f"lo-{level}"].to_numpy(), f[f"hi-{level}"].to_numpy()
                row[f"coverage_{level}"] = coverage(y, lo, hi)
                row[f"msis_{level}"] = msis(y, lo, hi, scale, level)
            rows.append(row)
            path = f.reset_index().assign(model=model.name, unique_id=uid, window=window)
            paths.append(path.assign(y=y, scale=scale))
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True)


def leaderboard(
    scores: pd.DataFrame, baseline: str = "snaive", owa_reference: str = "naive2"
) -> pd.DataFrame:
    """Mean metrics per model. `rel_mase` is MASE relative to the baseline; `owa` is the M4
    Overall Weighted Average, (sMAPE/sMAPE_naive2 + MASE/MASE_naive2) / 2. Lower is better."""
    interval_cols = [c for c in scores.columns if c.startswith(("coverage_", "msis_"))]
    board = scores.groupby("model").agg(
        mase=("mase", "mean"),
        smape=("smape", "mean"),
        rmse=("rmse", "mean"),
        **{c: (c, "mean") for c in interval_cols},
        ms_per_series=("seconds_per_series", lambda s: 1000 * np.mean(s)),
        n=("mase", "size"),
    )
    if owa_reference in board.index:
        ref = board.loc[owa_reference]
        board.insert(1, "owa", (board["smape"] / ref["smape"] + board["mase"] / ref["mase"]) / 2)
    if baseline in board.index:
        board.insert(1, "rel_mase", board["mase"] / board.loc[baseline, "mase"])
    return board.sort_values("mase")


COMBINATIONS = {"comb": ("ets", "arima", "theta")}


def attribute_combination_cost(scores: pd.DataFrame) -> pd.DataFrame:
    """A combination reuses its members' cached forecasts, so its measured runtime is near
    zero. Charge it the sum of its members' runtimes instead (when they were run)."""
    scores = scores.copy()
    for combo, members in COMBINATIONS.items():
        if combo not in set(scores["model"]) or not set(members) <= set(scores["model"]):
            continue
        member_cost = (
            scores[scores["model"].isin(members)]
            .groupby(["unique_id", "window"])["seconds_per_series"]
            .sum()
        )
        mask = scores["model"] == combo
        keys = pd.MultiIndex.from_frame(scores.loc[mask, ["unique_id", "window"]])
        scores.loc[mask, "seconds_per_series"] = member_cost.reindex(keys).to_numpy()
    return scores

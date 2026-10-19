import numpy as np
import pandas as pd
import pytest

from conftest import seasonal_frame
from ts_forecasting.backtest import (
    attribute_combination_cost,
    leaderboard,
    rolling_origin_splits,
    run_backtest,
)
from ts_forecasting.models import MODEL_NAMES, Naive, SeasonalNaive, build


def test_splits_never_leak_future_into_train():
    df = seasonal_frame()
    for _, train, test in rolling_origin_splits(df, h=6, n_windows=3):
        last_train = train.groupby("unique_id")["ds"].max()
        first_test = test.groupby("unique_id")["ds"].min()
        assert (last_train < first_test).all()
        assert (test.groupby("unique_id").size() == 6).all()


def test_last_window_tests_on_final_points_and_windows_step_by_h():
    df = seasonal_frame(length=60)
    splits = list(rolling_origin_splits(df, h=6, n_windows=3))
    assert [len(train) // 3 for _, train, _ in splits] == [42, 48, 54]
    final_test = splits[-1][2]
    assert final_test.groupby("unique_id")["ds"].max().eq(df["ds"].max()).all()


def test_seasonal_naive_repeats_last_season():
    df = seasonal_frame(n_series=1)
    fc = SeasonalNaive().forecast(df, h=15, season=12, levels=(80,))
    y = df["y"].to_numpy()
    np.testing.assert_allclose(fc["mean"][:12], y[-12:])
    np.testing.assert_allclose(fc["mean"][12:], y[-12:-9])
    width = fc["hi-80"] - fc["lo-80"]
    assert width.iloc[12] > width.iloc[11]  # intervals widen after one full season


def test_seasonal_naive_beats_naive_on_seasonal_data():
    df = seasonal_frame()
    results = [run_backtest(df, m(), h=12, n_windows=2, season=12) for m in (SeasonalNaive, Naive)]
    board = leaderboard(pd.concat([scores for scores, _ in results]))
    assert board.index[0] == "snaive"
    assert board.loc["naive", "rel_mase"] > 1
    assert board.loc["snaive", "coverage_95"] > 0.8


def test_every_registered_model_builds():
    for name in MODEL_NAMES:
        assert build(name, n_jobs=1).name == name


def test_forecast_paths_line_up_with_scores():
    df = seasonal_frame()
    scores, forecasts = run_backtest(df, SeasonalNaive(), h=6, n_windows=2, season=12)
    assert len(forecasts) == len(scores) * 6
    first = scores.iloc[0]
    path = forecasts[
        (forecasts["unique_id"] == first["unique_id"]) & (forecasts["window"] == first["window"])
    ]
    recomputed = np.mean(np.abs(path["y"] - path["mean"])) / path["scale"].iloc[0]
    assert recomputed == pytest.approx(first["mase"])


def test_combination_is_charged_its_members_runtime():
    rows = [
        {"model": m, "unique_id": "A", "window": 0, "seconds_per_series": t}
        for m, t in [("ets", 0.1), ("arima", 0.5), ("theta", 0.2), ("comb", 0.0)]
    ]
    scores = attribute_combination_cost(pd.DataFrame(rows))
    assert scores.loc[scores["model"] == "comb", "seconds_per_series"].item() == pytest.approx(0.8)

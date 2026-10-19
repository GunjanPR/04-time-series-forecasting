import numpy as np
import pandas as pd

from conftest import seasonal_frame
from ts_forecasting.backtest import leaderboard, rolling_origin_splits, run_backtest
from ts_forecasting.models import MODELS, Naive, SeasonalNaive


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
    scores = [run_backtest(df, m(), h=12, n_windows=2, season=12) for m in (SeasonalNaive, Naive)]
    board = leaderboard(pd.concat(scores))
    assert board.index[0] == "snaive"
    assert board.loc["naive", "rel_mase"] > 1
    assert board.loc["snaive", "coverage_95"] > 0.8


def test_registry_names_match():
    assert set(MODELS) == {"snaive", "naive"}

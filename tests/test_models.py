"""Every model honours the forecast contract; method-specific behaviour is checked too."""

import numpy as np
import pandas as pd
import pytest

from conftest import seasonal_frame
from ts_forecasting.models import (
    MODEL_NAMES,
    Combination,
    Naive,
    Naive2,
    SeasonalNaive,
    build,
    is_seasonal,
    seasonal_indices,
)

H, SEASON, LEVELS = 12, 12, (80, 95)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return seasonal_frame(n_series=3, length=72)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forecast_contract(frame, name):
    fc = build(name, n_jobs=1).forecast(frame, H, SEASON, LEVELS)
    assert list(fc.columns[:3]) == ["unique_id", "step", "mean"]
    assert len(fc) == 3 * H
    assert set(fc["unique_id"]) == set(frame["unique_id"])
    for _, steps in fc.groupby("unique_id")["step"]:
        assert steps.tolist() == list(range(1, H + 1))
    assert np.isfinite(fc.drop(columns="unique_id").to_numpy(float)).all()
    assert (fc["lo-95"] <= fc["lo-80"] + 1e-9).all()
    assert (fc["lo-80"] <= fc["mean"] + 1e-9).all()
    assert (fc["mean"] <= fc["hi-80"] + 1e-9).all()
    assert (fc["hi-80"] <= fc["hi-95"] + 1e-9).all()


@pytest.mark.parametrize("name", ["ets", "arima", "theta", "sarima", "lgbm", "naive2"])
def test_seasonal_models_beat_naive_on_clean_seasonal_data(name):
    frame = seasonal_frame(n_series=20, length=72)  # enough rows for the global LightGBM
    train = frame.groupby("unique_id").head(60)
    actual = frame.groupby("unique_id").tail(12).copy()
    actual["step"] = actual.groupby("unique_id").cumcount() + 1

    def mae(model) -> float:
        fc = model.forecast(train, H, SEASON, LEVELS).merge(actual, on=["unique_id", "step"])
        assert len(fc) == len(actual)
        return float(np.mean(np.abs(fc["mean"] - fc["y"])))

    assert mae(build(name, n_jobs=1)) < 0.5 * mae(Naive())


def test_seasonal_naive_repeats_last_season(frame):
    one = frame[frame["unique_id"] == "S0"]
    fc = SeasonalNaive().forecast(one, 15, SEASON, (80,))
    y = one["y"].to_numpy()
    np.testing.assert_allclose(fc["mean"][:12], y[-12:])
    np.testing.assert_allclose(fc["mean"][12:], y[-12:-9])
    width = fc["hi-80"] - fc["lo-80"]
    assert width.iloc[12] > width.iloc[11]  # intervals widen after one full season


def test_seasonality_test_and_indices():
    t = np.arange(96)
    seasonal = 100 + 30 * np.sin(2 * np.pi * t / 12)
    assert is_seasonal(seasonal, 12)
    assert not is_seasonal(np.linspace(1, 50, 96), 12)
    idx = seasonal_indices(seasonal, 12)
    assert idx.mean() == pytest.approx(1)
    assert idx.argmax() == 3  # sin peaks at t = 3 within each season


def test_naive2_on_non_seasonal_data_equals_naive():
    df = pd.DataFrame(
        {"unique_id": "A", "ds": pd.date_range("2000-01-01", periods=48, freq="MS"),
         "y": np.linspace(10, 60, 48)}
    )  # fmt: skip
    a = Naive2().forecast(df, 6, SEASON, LEVELS)
    b = Naive().forecast(df, 6, SEASON, LEVELS)
    pd.testing.assert_frame_equal(a, b)


def test_combination_is_member_mean(frame):
    combo = Combination("c", [Naive(), SeasonalNaive()]).forecast(frame, H, SEASON, LEVELS)
    a = Naive().forecast(frame, H, SEASON, LEVELS)
    b = SeasonalNaive().forecast(frame, H, SEASON, LEVELS)
    np.testing.assert_allclose(combo["mean"], (a["mean"] + b["mean"]) / 2)


def test_lgbm_conformal_interval_widths_are_per_step_and_positive(frame):
    fc = build("lgbm", n_jobs=1).forecast(frame, H, SEASON, LEVELS)
    width = (fc["hi-95"] - fc["lo-95"]).to_numpy()
    assert (width > 0).all()

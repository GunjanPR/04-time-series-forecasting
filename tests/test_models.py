"""Every model honours the forecast contract; method-specific behaviour is checked too."""

import numpy as np
import pandas as pd
import pytest

from conftest import seasonal_frame
from ts_forecasting.models import (
    MODEL_NAMES,
    Combination,
    LightGBMGlobal,
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


def test_adaptive_intervals_follow_series_volatility():
    rng = np.random.default_rng(3)
    t = np.arange(72)
    frames = []
    for i in range(20):
        noise = 1.0 if i % 2 == 0 else 15.0  # calm and volatile series, same level
        y = 200 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, noise, 72)
        frames.append(pd.DataFrame({"unique_id": f"S{i}",
                                    "ds": pd.date_range("2010-01-01", periods=72, freq="MS"),
                                    "y": y}))  # fmt: skip
    df = pd.concat(frames, ignore_index=True)
    width = {}
    for name in ("lgbm", "lgbm_global"):
        fc = build(name, n_jobs=1).forecast(df, 6, 12, (95,))
        fc["width"] = fc["hi-95"] - fc["lo-95"]
        calm = fc[fc["unique_id"].str[1:].astype(int) % 2 == 0]["width"].mean()
        volatile = fc[fc["unique_id"].str[1:].astype(int) % 2 == 1]["width"].mean()
        width[name] = volatile / calm
    assert width["lgbm"] > 3 * width["lgbm_global"]
    assert build("lgbm_global").name == "lgbm_global"


def test_multi_origin_calibration_collects_errors_from_every_origin(frame):
    one = LightGBMGlobal(n_jobs=1)._calibration_errors(frame, H, SEASON)
    three = LightGBMGlobal(n_jobs=1, calibration_origins=3)._calibration_errors(frame, H, SEASON)
    assert one.shape == (3, H)  # 3 series x 1 origin
    assert three.shape == (9, H)  # 3 series x 3 origins (72 points leave > 1 season each)
    assert (three >= 0).all()


def test_origins_without_a_season_of_history_are_skipped():
    long = seasonal_frame(n_series=1, length=72)
    short = seasonal_frame(n_series=1, length=40).assign(unique_id="short")
    df = pd.concat([long, short], ignore_index=True)
    # 40 - 3 x 12 leaves 4 points (less than a season) before the oldest origin
    errors = LightGBMGlobal(n_jobs=1, calibration_origins=3)._calibration_errors(df, H, SEASON)
    assert errors.shape == (5, H)  # long: 3 origins, short: the 2 recent ones

    with pytest.raises(ValueError, match="calibration_origins"):
        LightGBMGlobal(n_jobs=1, calibration_origins=3)._calibration_errors(short, H, SEASON)


def test_calibration_origins_is_validated_and_named():
    assert build("lgbm_cal4").name == "lgbm_cal4"
    assert build("lgbm_cal8").calibration_origins == 8
    assert build("lgbm", calibration_origins=26).name == "lgbm"  # benchmark name is stable
    with pytest.raises(ValueError):
        LightGBMGlobal(calibration_origins=0)


def _daily_frame(n_series=8, length=140):
    rng = np.random.default_rng(5)
    t = np.arange(length)
    frames = []
    for i in range(n_series):
        weekly = np.array([1.0, 1.1, 1.1, 1.05, 1.0, 0.6, 0.5])[t % 7]
        y = (100 + i) * weekly + rng.normal(0, 2, length)
        ds = pd.date_range("2013-01-07", periods=length, freq="D")  # a Monday
        frames.append(pd.DataFrame({"unique_id": f"D{i}", "ds": ds, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_frequency_and_season_position_helpers():
    from ts_forecasting.models import infer_freq, season_position

    daily = _daily_frame(1, 14)
    assert infer_freq(daily) == "D"
    assert infer_freq(seasonal_frame(n_series=1)) == "MS"
    assert season_position(daily["ds"], 7).tolist()[:8] == [1, 2, 3, 4, 5, 6, 7, 1]
    months = pd.date_range("2015-11-01", periods=3, freq="MS")
    assert season_position(months, 12).tolist() == [11, 12, 1]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_runs_on_daily_data(name):
    df = _daily_frame()
    fc = build(name, n_jobs=1).forecast(df, 14, 7, LEVELS)
    assert len(fc) == 8 * 14
    assert np.isfinite(fc["mean"]).all()


def test_calibration_spans_about_one_year_per_dataset():
    from ts_forecasting.data import DATASETS, calibration_origins

    assert calibration_origins(DATASETS["m4-monthly"], 18) == 1  # one origin already > a year
    assert calibration_origins(DATASETS["electricity-daily"], 14) == 26

import numpy as np
import pytest

from ts_forecasting.metrics import coverage, mase, rmse, smape


def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mase(y, y, np.arange(30.0), season=12) == 0
    assert smape(y, y) == 0
    assert rmse(y, y) == 0


def test_mase_scale_is_in_sample_seasonal_naive_mae():
    y_train = np.array([0.0, 2.0, 4.0, 6.0])  # seasonal-naive (m=2) errors are all 4
    assert mase(np.array([10.0]), np.array([6.0]), y_train, season=2) == pytest.approx(1.0)


def test_mase_handles_constant_history():
    assert np.isfinite(mase(np.array([5.0]), np.array([4.0]), np.ones(30), season=12))


def test_smape_bounds_and_zero_denominator():
    assert smape(np.array([1.0]), np.array([-1.0])) == pytest.approx(200)
    assert smape(np.array([0.0]), np.array([0.0])) == 0


def test_coverage():
    y = np.array([1.0, 5.0, 10.0, 20.0])
    assert coverage(y, np.full(4, 0.0), np.full(4, 10.0)) == 0.75

import numpy as np
import pandas as pd

from ts_forecasting.analysis import FEATURES, series_features, win_table


def test_strength_features_separate_seasonal_and_trending_series():
    t = np.arange(120)
    rng = np.random.default_rng(0)
    seasonal = series_features(100 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 120), 12)
    trending = series_features(10 + 2.0 * t + rng.normal(0, 1, 120), 12)
    assert seasonal["seasonal_strength"] > 0.9 > trending["seasonal_strength"]
    assert trending["trend_strength"] > 0.9
    assert seasonal["length"] == 120


def test_win_table_has_five_quintiles_per_feature():
    rng = np.random.default_rng(1)
    ids = [f"S{i}" for i in range(50)]
    features = pd.DataFrame({f: rng.random(50) for f in FEATURES}, index=ids)
    per_series = pd.DataFrame({"naive2": rng.random(50), "comb": rng.random(50)}, index=ids)
    table = win_table(features, per_series, "naive2", "comb")
    assert len(table) == 5 * len(FEATURES)
    assert (table.groupby("feature")["series"].sum() == 50).all()
    assert table["simple_win_pct"].between(0, 100).all()

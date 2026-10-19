import pandas as pd

from ts_forecasting.data import sample_ids, wide_to_long


def test_wide_to_long_joins_train_and_test_in_order():
    train = pd.DataFrame({"V1": ["M1", "M2"], "V2": [1.0, 10.0], "V3": [2.0, None]})
    test = pd.DataFrame({"V1": ["M1", "M2"], "V2": [3.0, 11.0]})
    df = wide_to_long(train, test)
    assert df[df["unique_id"] == "M1"]["y"].tolist() == [1.0, 2.0, 3.0]
    assert df[df["unique_id"] == "M2"]["y"].tolist() == [10.0, 11.0]
    assert df.groupby("unique_id")["ds"].max().nunique() == 1  # every series ends together


def test_sample_ids_filters_short_series_and_is_deterministic():
    lengths = pd.Series({f"M{i}": 50 + i for i in range(20)})
    ids = sample_ids(lengths, min_length=60, max_series=5, seed=1)
    assert len(ids) == 5
    assert all(lengths[i] >= 60 for i in ids)
    assert list(ids) == list(sample_ids(lengths, min_length=60, max_series=5, seed=1))


def test_daily_kwh_sums_quarter_hours_into_complete_days():
    import numpy as np

    from ts_forecasting.data import daily_kwh

    # timestamps mark the END of each 15-minute interval, like the UCI file
    index = pd.date_range("2012-01-01 00:15", "2012-01-03 00:00", freq="15min")
    raw = pd.DataFrame({"MT_001": np.full(len(index), 4.0)}, index=index)
    daily = daily_kwh(raw)
    assert list(daily.index) == list(pd.to_datetime(["2012-01-01", "2012-01-02"]))
    assert daily["MT_001"].tolist() == [96 * 4.0 * 0.25] * 2  # 96 intervals x 1 kWh


def test_daily_to_long_trims_leading_zeros_and_drops_gappy_clients():
    import numpy as np

    from ts_forecasting.data import daily_to_long

    days = pd.date_range("2012-01-01", periods=400, freq="D")
    late = np.r_[np.zeros(30), np.ones(370)]
    gappy = np.ones(400)
    gappy[::20] = 0  # 5% zero days
    short = np.r_[np.zeros(300), np.ones(100)]
    daily = pd.DataFrame({"late": late, "gappy": gappy, "short": short}, index=days)
    long, lengths = daily_to_long(daily, min_len=200)
    assert set(long["unique_id"]) == {"late"}
    assert lengths["late"] == 370
    assert long["ds"].min() == days[30]

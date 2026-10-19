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

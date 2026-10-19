"""Dataset loaders. Every dataset is returned in long format: `unique_id, ds, y`.

M4-Monthly: the official train and test files are joined into one series each, so the last
rolling-origin window is exactly the official M4 test period. M4 start dates are known to be
unreliable, so every series gets a synthetic month-start index ending in 2015-12; models only
see the monthly frequency, never the calendar year.

    python -m ts_forecasting.data        # download (if missing) and cache the sample
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ts_forecasting.config import Settings, get_settings

log = logging.getLogger(__name__)

M4_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{part}/Monthly-{split}.csv"
M4_END = pd.Timestamp("2015-12-01")
SEASON_LENGTH = {"m4-monthly": 12}
# Shortest history allowed in the first backtest window: three full seasons.
MIN_TRAIN_SEASONS = 3


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, path)
    return path


def wide_to_long(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Join M4 wide train/test rows (first column = id) into one long frame."""
    train = train.set_index(train.columns[0])
    test = test.set_index(test.columns[0])
    rows = []
    for uid, values in train.iterrows():
        y = np.concatenate(
            [values.dropna().to_numpy(float), test.loc[uid].dropna().to_numpy(float)]
        )
        ds = pd.date_range(end=M4_END, periods=len(y), freq="MS")
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def sample_ids(
    lengths: pd.Series, min_length: int, max_series: int | None, seed: int
) -> np.ndarray:
    """Ids (index of `lengths`) with at least `min_length` points, sampled down to `max_series`."""
    ids = np.sort(lengths[lengths >= min_length].index.to_numpy())
    if max_series is not None and max_series < len(ids):
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_series, replace=False))
    return ids


def min_length(settings: Settings) -> int:
    season = SEASON_LENGTH[settings.dataset]
    return settings.horizon * settings.n_windows + MIN_TRAIN_SEASONS * season


def load_m4_monthly(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    n = settings.max_series or "all"
    cache = (
        settings.data_dir
        / "processed"
        / (f"m4-monthly_n{n}_seed{settings.random_seed}_min{min_length(settings)}.parquet")
    )
    if cache.exists():
        return pd.read_parquet(cache)
    raw = settings.data_dir / "raw"
    train = pd.read_csv(
        _download(M4_URL.format(part="Train", split="train"), raw / "Monthly-train.csv")
    )
    test = pd.read_csv(
        _download(M4_URL.format(part="Test", split="test"), raw / "Monthly-test.csv")
    )
    # Sample on ids first so only the chosen series are converted to long format.
    first = train.columns[0]
    lengths = train.set_index(first).notna().sum(axis=1) + test.set_index(first).notna().sum(axis=1)
    chosen = set(
        sample_ids(lengths, min_length(settings), settings.max_series, settings.random_seed)
    )
    df = wide_to_long(train[train[first].isin(chosen)], test[test[first].isin(chosen)])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    log.info("cached %d series (%d rows) -> %s", df["unique_id"].nunique(), len(df), cache)
    return df


LOADERS = {"m4-monthly": load_m4_monthly}


def load(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return LOADERS[settings.dataset](settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load()

"""Dataset loaders. Every dataset is returned in long format: `unique_id, ds, y`.

* `m4-monthly` - the official M4 train and test files joined into one series each, so the
  last rolling-origin window is exactly the official M4 test period. M4 start dates are known
  to be unreliable, so every series gets a synthetic month-start index ending in 2015-12;
  models only see the monthly frequency, never the calendar year.
* `electricity-daily` - UCI ElectricityLoadDiagrams20112014: 15-minute kW readings of 370
  Portuguese clients, 2011-2014, summed to daily kWh. Leading zeros (clients that joined
  later) are trimmed, and clients with more than 1% zero days after that are dropped as
  metering gaps.

    python -m ts_forecasting.data        # download (if missing) and cache the sample
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ts_forecasting.config import Settings, get_settings

log = logging.getLogger(__name__)

M4_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{part}/Monthly-{split}.csv"
M4_END = pd.Timestamp("2015-12-01")
ELECTRICITY_URL = (
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)
MAX_ZERO_DAY_SHARE = 0.01


@dataclass(frozen=True)
class DatasetSpec:
    freq: str  # pandas frequency of the series
    season: int  # seasonal period used by every model and by MASE
    horizon: int  # default forecast horizon
    n_windows: int  # default number of rolling-origin windows
    min_train: int  # shortest history allowed in the first backtest window
    unit: str  # one period, for labels


DATASETS = {
    "m4-monthly": DatasetSpec("MS", 12, 18, 3, min_train=36, unit="month"),
    "electricity-daily": DatasetSpec("D", 7, 14, 3, min_train=365, unit="day"),
}
SEASON_LENGTH = {name: spec.season for name, spec in DATASETS.items()}


def resolve(settings: Settings) -> Settings:
    """Fill a blank horizon / number of windows with the dataset's defaults."""
    spec = DATASETS[settings.dataset]
    return settings.model_copy(
        update={
            "horizon": settings.horizon or spec.horizon,
            "n_windows": settings.n_windows or spec.n_windows,
        }
    )


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
    settings = resolve(settings)
    return settings.horizon * settings.n_windows + DATASETS[settings.dataset].min_train


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


def daily_kwh(raw: pd.DataFrame) -> pd.DataFrame:
    """15-minute kW readings (index = end of each interval) -> daily kWh per client."""
    day = (raw.index - pd.Timedelta(minutes=15)).normalize()
    daily = (raw * 0.25).groupby(day).sum()
    complete = daily.index[daily.index < raw.index.max().normalize()]
    return daily.loc[complete]


def daily_to_long(daily: pd.DataFrame, min_len: int) -> tuple[pd.DataFrame, pd.Series]:
    """Trim leading zeros, drop clients with metering gaps; returns (long frame, lengths)."""
    rows, lengths = [], {}
    for client in daily.columns:
        y = daily[client]
        nonzero = np.flatnonzero(y.to_numpy() > 0)
        if len(nonzero) == 0:
            continue
        y = y.iloc[nonzero[0] :]
        if (y == 0).mean() > MAX_ZERO_DAY_SHARE or len(y) < min_len:
            continue
        lengths[client] = len(y)
        rows.append(pd.DataFrame({"unique_id": client, "ds": y.index, "y": y.to_numpy()}))
    return pd.concat(rows, ignore_index=True), pd.Series(lengths)


def load_electricity(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    n = settings.max_series or "all"
    cache = (
        settings.data_dir
        / "processed"
        / f"electricity-daily_n{n}_seed{settings.random_seed}_min{min_length(settings)}.parquet"
    )
    if cache.exists():
        return pd.read_parquet(cache)
    archive = _download(ELECTRICITY_URL, settings.data_dir / "raw" / "electricity.zip")
    with zipfile.ZipFile(archive) as zf, zf.open("LD2011_2014.txt") as fh:
        raw = pd.read_csv(fh, sep=";", decimal=",", index_col=0, parse_dates=True)
    df, lengths = daily_to_long(daily_kwh(raw), min_length(settings))
    chosen = set(sample_ids(lengths, 0, settings.max_series, settings.random_seed))
    df = df[df["unique_id"].isin(chosen)].reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    log.info("cached %d series (%d rows) -> %s", df["unique_id"].nunique(), len(df), cache)
    return df


LOADERS = {"m4-monthly": load_m4_monthly, "electricity-daily": load_electricity}


def load(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return LOADERS[settings.dataset](settings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load()

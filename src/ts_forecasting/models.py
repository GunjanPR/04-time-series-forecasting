"""Forecasting methods behind one interface.

Every model takes the training rows of many series (long format, `unique_id, ds, y`) and
returns the next `h` point forecasts and prediction intervals for each series, as columns
`unique_id, step, mean, lo-<level>, hi-<level>` (step = 1..h).

| name    | method                                                             | library        |
|---------|--------------------------------------------------------------------|----------------|
| naive   | last value                                                         | numpy          |
| snaive  | last season (the benchmark baseline)                               | numpy          |
| naive2  | naive on seasonally adjusted data (official M4 benchmark)          | numpy          |
| ets     | AutoETS (error/trend/season chosen by AICc)                        | statsforecast  |
| arima   | AutoARIMA (Hyndman-Khandakar stepwise search)                      | statsforecast  |
| theta   | AutoTheta (best of standard/optimised/dynamic theta)               | statsforecast  |
| sarima  | fixed SARIMA(0,1,1)(0,1,1)_m, the "airline" model                  | statsmodels    |
| prophet | additive trend + yearly seasonality                                | prophet        |
| lgbm    | one global LightGBM over all series, lag + calendar features       | lightgbm       |
| comb    | mean of ets, arima and theta (the M4 "Comb"-style combination)     | -              |
"""

from __future__ import annotations

import logging
import shutil
import sys
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

Levels = tuple[int, ...]


class Forecaster(Protocol):
    name: str

    def forecast(
        self, train: pd.DataFrame, h: int, season: int, levels: Levels
    ) -> pd.DataFrame: ...


def _gaussian(uid, mean: np.ndarray, sigma_h: np.ndarray, levels: Levels) -> pd.DataFrame:
    out = {"unique_id": uid, "step": np.arange(1, len(mean) + 1), "mean": mean}
    for level in levels:
        z = norm.ppf(0.5 + level / 200)
        out[f"lo-{level}"] = mean - z * sigma_h
        out[f"hi-{level}"] = mean + z * sigma_h
    return pd.DataFrame(out)


def _series(train: pd.DataFrame):
    for uid, group in train.groupby("unique_id", sort=False):
        yield uid, group["y"].to_numpy(float)


class Naive:
    """Repeat the last value. sigma_h = sigma * sqrt(h) (FPP3, 5.5)."""

    name = "naive"

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = []
        for uid, y in _series(train):
            sigma = np.sqrt(np.mean(np.diff(y) ** 2))
            mean = np.full(h, y[-1])
            frames.append(_gaussian(uid, mean, sigma * np.sqrt(np.arange(1, h + 1)), levels))
        return pd.concat(frames, ignore_index=True)


class SeasonalNaive:
    """Repeat the last observed season. sigma_h = sigma * sqrt(k + 1), k = floor((h-1)/m),
    sigma from in-sample seasonal-naive residuals (FPP3, 5.5)."""

    name = "snaive"

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = []
        for uid, y in _series(train):
            mean = np.resize(y[-season:], h)
            sigma = np.sqrt(np.mean((y[season:] - y[:-season]) ** 2))
            k = np.floor(np.arange(h) / season)
            frames.append(_gaussian(uid, mean, sigma * np.sqrt(k + 1), levels))
        return pd.concat(frames, ignore_index=True)


def is_seasonal(y: np.ndarray, season: int) -> bool:
    """M4 seasonality test: lag-m autocorrelation significant at 90% (Bartlett's formula)."""
    if season <= 1 or len(y) < 3 * season:
        return False
    y = y - y.mean()
    denom = np.sum(y**2)
    if denom == 0:
        return False
    acf = np.array([np.sum(y[k:] * y[:-k]) / denom for k in range(1, season + 1)])
    limit = 1.645 * np.sqrt((1 + 2 * np.sum(acf[:-1] ** 2)) / len(y))
    return bool(abs(acf[-1]) > limit)


def seasonal_indices(y: np.ndarray, season: int) -> np.ndarray:
    """Classical multiplicative decomposition: one index per position in the season,
    aligned so that index[i] belongs to y[i % season]. Indices average to 1."""
    if season % 2 == 0:
        weights = np.r_[0.5, np.ones(season - 1), 0.5] / season
    else:
        weights = np.ones(season) / season
    trend = np.convolve(y, weights, mode="valid")
    offset = (len(weights) - 1) // 2
    ratio = y[offset : offset + len(trend)] / trend
    positions = (np.arange(len(ratio)) + offset) % season
    idx = np.array([np.mean(ratio[positions == p]) for p in range(season)])
    return idx / idx.mean()


class Naive2:
    """Official M4 benchmark: seasonally adjust (if the ACF test says seasonal), forecast with
    naive, re-seasonalise. Intervals: naive intervals on the adjusted scale, re-seasonalised."""

    name = "naive2"

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = []
        for uid, y in _series(train):
            if np.all(y > 0) and is_seasonal(y, season):
                idx = seasonal_indices(y, season)
                in_sample = idx[np.arange(len(y)) % season]
                future = idx[np.arange(len(y), len(y) + h) % season]
            else:
                in_sample, future = np.ones(len(y)), np.ones(h)
            adjusted = y / in_sample
            sigma = np.sqrt(np.mean(np.diff(adjusted) ** 2))
            fc = _gaussian(
                uid, np.full(h, adjusted[-1]), sigma * np.sqrt(np.arange(1, h + 1)), levels
            )
            for col in fc.columns[2:]:
                fc[col] = fc[col] * future
            frames.append(fc)
        return pd.concat(frames, ignore_index=True)


class StatsForecastModel:
    """Wrap a statsforecast model; its native prediction intervals are used.

    Forecasts are memoised on the training data, so `comb` reuses the member forecasts that
    the benchmark has already computed instead of refitting them."""

    _cache: dict[tuple, pd.DataFrame] = {}

    def __init__(self, name: str, n_jobs: int = -1):
        self.name = name
        self.n_jobs = n_jobs

    def _model(self, season: int):
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

        return {
            "ets": AutoETS(season_length=season),
            "arima": AutoARIMA(season_length=season),
            "theta": AutoTheta(season_length=season),
        }[self.name]

    def forecast(self, train, h, season, levels=(80, 95)):
        key = (self.name, h, season, levels, int(pd.util.hash_pandas_object(train).sum()))
        if key not in self._cache:
            self._cache[key] = self._forecast(train, h, season, levels)
        return self._cache[key].copy()

    def _forecast(self, train, h, season, levels):
        from statsforecast import StatsForecast

        model = self._model(season)
        sf = StatsForecast(models=[model], freq="MS", n_jobs=self.n_jobs)
        fc = sf.forecast(df=train[["unique_id", "ds", "y"]], h=h, level=list(levels))
        alias = str(model)
        out = pd.DataFrame(
            {
                "unique_id": fc["unique_id"].to_numpy(),
                "step": fc.groupby("unique_id", sort=False).cumcount().to_numpy() + 1,
                "mean": fc[alias].to_numpy(),
            }
        )
        for level in levels:
            out[f"lo-{level}"] = fc[f"{alias}-lo-{level}"].to_numpy()
            out[f"hi-{level}"] = fc[f"{alias}-hi-{level}"].to_numpy()
        return out


def _sarima_one(uid, y: np.ndarray, h: int, season: int, levels: Levels) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, season)).fit(disp=False)
            pred = fit.get_forecast(h)
            mean = np.asarray(pred.predicted_mean)
            se = np.asarray(pred.se_mean)
        except Exception:  # noqa: BLE001 - fall back to seasonal naive if the fit fails
            return SeasonalNaive().forecast(
                pd.DataFrame({"unique_id": uid, "y": y}), h, season, levels
            )
    return _gaussian(uid, mean, se, levels)


class Sarima:
    name = "sarima"

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = Parallel(n_jobs=self.n_jobs)(
            delayed(_sarima_one)(uid, y, h, season, levels) for uid, y in _series(train)
        )
        return pd.concat(frames, ignore_index=True)


def ensure_prophet_tbb() -> None:
    """Windows only: put cmdstan's tbb.dll next to Prophet's Stan binary.

    Windows searches the executable's folder before System32 but System32 before PATH, so an
    unrelated System32\tbb.dll (shipped by some drivers) shadows the one cmdstan needs and
    the fit dies with 0xC0000139 (entry point not found)."""
    if sys.platform != "win32":
        return
    import prophet

    stan_dir = Path(prophet.__file__).parent / "stan_model"
    target = stan_dir / "tbb.dll"
    if not target.exists():
        for source in stan_dir.glob("cmdstan-*/stan/lib/stan_math/lib/tbb/tbb.dll"):
            shutil.copyfile(source, target)
            break


def _prophet_one(uid, ds: pd.Series, y: np.ndarray, h: int, levels: Levels) -> pd.DataFrame:
    from prophet import Prophet

    ensure_prophet_tbb()

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=500,
    )
    model.fit(pd.DataFrame({"ds": ds.to_numpy(), "y": y}))
    future = model.make_future_dataframe(periods=h, freq="MS", include_history=False)
    mean = model.predict(future)["yhat"].to_numpy()
    samples = model.predictive_samples(future)["yhat"]  # shape (h, n_samples)
    out = {"unique_id": uid, "step": np.arange(1, h + 1), "mean": mean}
    for level in levels:
        out[f"lo-{level}"] = np.quantile(samples, 0.5 - level / 200, axis=1)
        out[f"hi-{level}"] = np.quantile(samples, 0.5 + level / 200, axis=1)
    return pd.DataFrame(out)


class ProphetModel:
    name = "prophet"

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def forecast(self, train, h, season, levels=(80, 95)):
        groups = train.groupby("unique_id", sort=False)
        frames = Parallel(n_jobs=self.n_jobs)(
            delayed(_prophet_one)(uid, g["ds"], g["y"].to_numpy(float), h, levels)
            for uid, g in groups
        )
        return pd.concat(frames, ignore_index=True)


class LightGBMGlobal:
    """One LightGBM model across all series, forecasting recursively.

    Each series is divided by the mean of its last `season` training values, so series on very
    different scales share one model. Features: lags 1..2m, rolling means of the last 3 and m
    values, and month of year; training rows start one season into each series.

    Prediction intervals are split-conformal per horizon step: the model is first fitted with
    the last `h` points of every series held out, and the errors on those points give a
    quantile per step; the final model is then refitted on all the data. With
    `intervals="adaptive"` (default) each error is first divided by its series' in-sample
    seasonal-naive MAE, so interval width follows each series' volatility; `"global"` uses one
    error distribution for every series.
    """

    name = "lgbm"

    def __init__(self, seed: int = 42, n_jobs: int = -1, intervals: str = "adaptive", **params):
        if intervals not in ("adaptive", "global"):
            raise ValueError(f"intervals must be 'adaptive' or 'global', not {intervals!r}")
        self.intervals = intervals
        self.name = "lgbm" if intervals == "adaptive" else "lgbm_global"
        self.params = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "objective": "l1",
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
            **params,
        }

    @staticmethod
    def _features(window: np.ndarray, month: int, season: int) -> np.ndarray:
        """Features for the next value, from the last 2m scaled values (oldest first).
        Histories shorter than 2m (early backtest windows) are left-padded with their first
        value."""
        if len(window) < 2 * season:
            window = np.r_[np.full(2 * season - len(window), window[0]), window]
        lags = window[::-1][: 2 * season]
        return np.r_[lags, window[-3:].mean(), window[-season:].mean(), month]

    def _design(self, train: pd.DataFrame, season: int):
        rows, targets = [], []
        for _, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(float)
            y = y / self._scale(y, season)
            months = g["ds"].dt.month.to_numpy()
            for t in range(season, len(y)):
                rows.append(self._features(y[max(0, t - 2 * season) : t], months[t], season))
                targets.append(y[t])
        return np.array(rows), np.array(targets)

    @staticmethod
    def _scale(y: np.ndarray, season: int) -> float:
        scale = np.mean(np.abs(y[-season:]))
        return scale if scale > 0 else 1.0

    def _fit(self, train: pd.DataFrame, season: int):
        import lightgbm as lgb

        x, y = self._design(train, season)
        return lgb.LGBMRegressor(**self.params).fit(x, y)

    def _predict(self, model, train: pd.DataFrame, h: int, season: int):
        """Recursive h-step forecasts on the scaled level. Returns ({uid: (scaled, scale)})."""
        histories, scales, next_month = {}, {}, {}
        for uid, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(float)
            scales[uid] = self._scale(y, season)
            histories[uid] = list(y[-2 * season :] / scales[uid])
            next_month[uid] = g["ds"].iloc[-1].month % 12 + 1
        uids = list(histories)
        preds = {uid: [] for uid in uids}
        for step in range(h):
            x = np.array(
                [
                    self._features(
                        np.array(histories[u][-2 * season :]),
                        (next_month[u] + step - 1) % 12 + 1,
                        season,
                    )
                    for u in uids
                ]
            )
            for u, p in zip(uids, model.predict(x), strict=True):
                preds[u].append(p)
                histories[u].append(p)
        return {u: (np.array(preds[u]), scales[u]) for u in uids}

    def _spreads(self, train: pd.DataFrame, fitted: dict, season: int) -> dict:
        """Per-series error unit for the conformal step. `global`: 1 for every series (one
        error distribution for all). `adaptive`: the series' in-sample seasonal-naive MAE on
        the scaled level, so volatile series get wider intervals and calm ones narrower."""
        if self.intervals == "global":
            return dict.fromkeys(fitted, 1.0)
        spreads = {}
        for uid, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(float) / fitted[uid][1]
            lag = season if len(y) > season else 1
            mae = np.mean(np.abs(y[lag:] - y[:-lag]))
            spreads[uid] = mae if mae > 0 else 1.0
        return spreads

    def forecast(self, train, h, season, levels=(80, 95)):
        train = train.sort_values(["unique_id", "ds"])
        position = train.groupby("unique_id", sort=False).cumcount(ascending=False)
        calib_train, calib_test = train[position >= h], train[position < h]
        calib = self._predict(self._fit(calib_train, season), calib_train, h, season)
        calib_spread = self._spreads(calib_train, calib, season)
        errors = np.array(
            [
                np.abs(g["y"].to_numpy(float) / calib[uid][1] - calib[uid][0]) / calib_spread[uid]
                for uid, g in calib_test.groupby("unique_id", sort=False)
            ]
        )  # (n_series, h), in units of each series' spread
        quantiles = {level: np.quantile(errors, level / 100, axis=0) for level in levels}

        final = self._predict(self._fit(train, season), train, h, season)
        spread = self._spreads(train, final, season)
        frames = []
        for uid, (scaled, scale) in final.items():
            out = {"unique_id": uid, "step": np.arange(1, h + 1), "mean": scaled * scale}
            for level in levels:
                half_width = quantiles[level] * spread[uid]
                out[f"lo-{level}"] = (scaled - half_width) * scale
                out[f"hi-{level}"] = (scaled + half_width) * scale
            frames.append(pd.DataFrame(out))
        return pd.concat(frames, ignore_index=True)


class Combination:
    """Equal-weight mean of member forecasts (points and interval bounds)."""

    def __init__(self, name: str, members: list[Forecaster]):
        self.name = name
        self.members = members

    def forecast(self, train, h, season, levels=(80, 95)):
        frames = [
            m.forecast(train, h, season, levels).set_index(["unique_id", "step"])
            for m in self.members
        ]
        return (sum(frames) / len(frames)).reset_index()


def build(name: str, n_jobs: int = -1, seed: int = 42) -> Forecaster:
    """Instantiate a model by name."""
    if name in ("ets", "arima", "theta"):
        return StatsForecastModel(name, n_jobs)
    if name == "comb":
        return Combination(
            "comb", [StatsForecastModel(n, n_jobs) for n in ("ets", "arima", "theta")]
        )
    factories = {
        "naive": Naive,
        "snaive": SeasonalNaive,
        "naive2": Naive2,
        "sarima": lambda: Sarima(n_jobs),
        "prophet": lambda: ProphetModel(n_jobs),
        "lgbm": lambda: LightGBMGlobal(seed=seed, n_jobs=n_jobs),
        # ablation: one conformal error distribution for all series (not in MODEL_NAMES)
        "lgbm_global": lambda: LightGBMGlobal(seed=seed, n_jobs=n_jobs, intervals="global"),
    }
    return factories[name]()


MODEL_NAMES = (
    "snaive",
    "naive",
    "naive2",
    "ets",
    "arima",
    "theta",
    "comb",
    "sarima",
    "prophet",
    "lgbm",
)

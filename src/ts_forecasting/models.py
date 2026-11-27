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
| lgbm_tuned | lgbm with hyperparameters chosen per window on an inner validation split | lightgbm |
| chronos | Chronos-2 (120M) zero-shot foundation model, quantile forecasts    | chronos        |
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


def infer_freq(train: pd.DataFrame) -> str:
    """Pandas frequency of the series (taken from the first one); monthly if unclear."""
    first = train[train["unique_id"] == train["unique_id"].iloc[0]]["ds"].sort_values()
    return pd.infer_freq(first) or "MS"


def season_position(ds: pd.Series | pd.DatetimeIndex, season: int) -> np.ndarray:
    """1-based position inside the season: month for monthly data, weekday for daily data."""
    ds = pd.DatetimeIndex(ds)
    if season == 12:
        return ds.month.to_numpy()
    if season == 7:
        return ds.dayofweek.to_numpy() + 1
    return (np.arange(len(ds)) % season) + 1


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
        sf = StatsForecast(models=[model], freq=infer_freq(train), n_jobs=self.n_jobs)
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


def _prophet_one(
    uid, ds: pd.Series, y: np.ndarray, h: int, levels: Levels, freq: str = "MS"
) -> pd.DataFrame:
    from prophet import Prophet

    ensure_prophet_tbb()

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=freq == "D",
        daily_seasonality=False,
        uncertainty_samples=500,
    )
    model.fit(pd.DataFrame({"ds": ds.to_numpy(), "y": y}))
    future = model.make_future_dataframe(periods=h, freq=freq, include_history=False)
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
        freq = infer_freq(train)
        frames = Parallel(n_jobs=self.n_jobs)(
            delayed(_prophet_one)(uid, g["ds"], g["y"].to_numpy(float), h, levels, freq)
            for uid, g in groups
        )
        return pd.concat(frames, ignore_index=True)


class LightGBMGlobal:
    """One LightGBM model across all series, forecasting recursively.

    Each series is divided by the mean of its last `season` training values, so series on very
    different scales share one model. Features: lags 1..2m, rolling means of the last 3 and m
    values, and the position in the season (month of year for monthly data, day of week
    for daily data); training rows start one season into each series.

    Prediction intervals are split-conformal per horizon step: the model is first fitted with
    the last `h` points of every series held out, and the errors on those points give a
    quantile per step; the final model is then refitted on all the data. With
    `intervals="adaptive"` (default) each error is first divided by its series' in-sample
    seasonal-naive MAE, so interval width follows each series' volatility; `"global"` uses one
    error distribution for every series.

    `calibration_origins=k` (default 1) collects the conformal errors from k forecast origins
    instead of one: a model fitted with the last k*h points held out forecasts from each of
    the k origins h, 2h, ... kh before the end. One origin calibrates on only the last h
    points of every series, which misses seasonal swings longer than h.
    """

    name = "lgbm"

    def __init__(
        self,
        seed: int = 42,
        n_jobs: int = -1,
        intervals: str = "adaptive",
        calibration_origins: int = 1,
        **params,
    ):
        if intervals not in ("adaptive", "global"):
            raise ValueError(f"intervals must be 'adaptive' or 'global', not {intervals!r}")
        if calibration_origins < 1:
            raise ValueError("calibration_origins must be at least 1")
        self.intervals = intervals
        self.calibration_origins = calibration_origins
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
    def _features(window: np.ndarray, position: int, season: int) -> np.ndarray:
        """Features for the next value, from the last 2m scaled values (oldest first).
        Histories shorter than 2m (early backtest windows) are left-padded with their first
        value."""
        if len(window) < 2 * season:
            window = np.r_[np.full(2 * season - len(window), window[0]), window]
        lags = window[::-1][: 2 * season]
        return np.r_[lags, window[-3:].mean(), window[-season:].mean(), position]

    def _design(self, train: pd.DataFrame, season: int):
        rows, targets = [], []
        for _, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(float)
            y = y / self._scale(y, season)
            positions = season_position(g["ds"], season)
            for t in range(season, len(y)):
                rows.append(self._features(y[max(0, t - 2 * season) : t], positions[t], season))
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
        histories, scales, last_position = {}, {}, {}
        for uid, g in train.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(float)
            scales[uid] = self._scale(y, season)
            histories[uid] = list(y[-2 * season :] / scales[uid])
            last_position[uid] = int(season_position(g["ds"].iloc[-1:], season)[0])
        uids = list(histories)
        preds = {uid: [] for uid in uids}
        for step in range(h):
            x = np.array(
                [
                    self._features(
                        np.array(histories[u][-2 * season :]),
                        (last_position[u] + step) % season + 1,
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

    def _calibration_errors(self, train: pd.DataFrame, h: int, season: int) -> np.ndarray:
        """Absolute errors in units of each series' spread, one row per (series, origin)."""
        k = self.calibration_origins
        position = train.groupby("unique_id", sort=False).cumcount(ascending=False)
        fit = train[position >= k * h]
        if not (fit.groupby("unique_id").size() > season).any():
            raise ValueError(
                f"calibration_origins={k} leaves no series with more than one season of "
                f"history before the oldest origin; use fewer origins"
            )
        model = self._fit(fit, season)
        errors = []
        for held_out in range(k * h, 0, -h):  # origins from the oldest to the latest
            history = train[position >= held_out]
            future = train[(position < held_out) & (position >= held_out - h)]
            # an origin needs more than one season of history for the seasonal spread
            lengths = history.groupby("unique_id", sort=False).size()
            keep = lengths.index[lengths > season]
            history = history[history["unique_id"].isin(keep)]
            pred = self._predict(model, history, h, season)
            spread = self._spreads(history, pred, season)
            for uid, g in future[future["unique_id"].isin(keep)].groupby("unique_id", sort=False):
                errors.append(np.abs(g["y"].to_numpy(float) / pred[uid][1] - pred[uid][0]))
                errors[-1] /= spread[uid]
        return np.array(errors)  # (n_series x origins, h)

    def forecast(self, train, h, season, levels=(80, 95)):
        train = train.sort_values(["unique_id", "ds"])
        errors = self._calibration_errors(train, h, season)
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


class LightGBMTuned(LightGBMGlobal):
    """`lgbm` with its hyperparameters chosen inside each backtest window.

    Nested validation, so the test window is never seen: within the window's training data,
    the last `h` points of every series are held out, each configuration in `GRID` is fitted
    on the rest and scored by mean MASE on the held-out points, and the best one is then used
    for the window's forecast (refitted on all its training data, with the same conformal
    intervals as `lgbm`).
    """

    GRID = [
        {"num_leaves": leaves, "learning_rate": lr, "n_estimators": trees,
         "min_child_samples": child}
        for leaves in (15, 63)
        for lr, trees in ((0.03, 500), (0.1, 200))
        for child in (20, 100)
    ]  # fmt: skip

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "lgbm_tuned"
        self.base_params = dict(self.params)
        self.chosen: list[dict] = []

    def _select(self, train: pd.DataFrame, h: int, season: int) -> dict:
        from ts_forecasting.metrics import mase

        position = train.groupby("unique_id", sort=False).cumcount(ascending=False)
        inner, held = train[position >= h], train[position < h]
        lengths = inner.groupby("unique_id", sort=False).size()
        keep = lengths.index[lengths > season]
        inner, held = inner[inner["unique_id"].isin(keep)], held[held["unique_id"].isin(keep)]
        actual = {u: g["y"].to_numpy(float) for u, g in held.groupby("unique_id", sort=False)}
        history = {u: g["y"].to_numpy(float) for u, g in inner.groupby("unique_id", sort=False)}
        scores = []
        for config in self.GRID:
            self.params = {**self.base_params, **config}
            pred = self._predict(self._fit(inner, season), inner, h, season)
            errs = [mase(actual[u], s * sc, history[u], season) for u, (s, sc) in pred.items()]
            scores.append(float(np.nanmean(errs)))
        best = self.GRID[int(np.argmin(scores))]
        self.chosen.append({**best, "inner_mase": min(scores)})
        log = logging.getLogger(__name__)
        log.info("lgbm_tuned picked %s (inner MASE %.3f)", best, min(scores))
        return best

    def forecast(self, train, h, season, levels=(80, 95)):
        train = train.sort_values(["unique_id", "ds"])
        best = self._select(train, h, season)
        self.params = {**self.base_params, **best}
        return super().forecast(train, h, season, levels)


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


class ChronosModel:
    """Zero-shot forecasts from a pretrained time-series foundation model (Chronos-2).

    No training on the benchmark: each series' history is the model's context, and the model
    returns quantiles directly. The point forecast is the median; the 80% and 95% intervals are
    the 10-90% and 2.5-97.5% quantiles (Chronos-2 is trained on 21 quantiles from 1% to 99%
    and interpolates between them). Each series is forecast on its own (no cross-learning), so
    the comparison with the univariate statistical models is like for like.

    Caveat: Chronos-2 was pretrained on large public corpora (`autogluon/chronos_datasets`,
    GIFT-Eval pretraining data) that include M4 and electricity series, so some test periods
    may have been seen in pretraining. Treat its scores here as an optimistic estimate.
    """

    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str | None = None,
                 batch_size: int = 256):  # fmt: skip
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=device)
        return self._pipeline

    def forecast(self, train, h, season, levels=(80, 95)):
        import torch

        pipeline = self._load()
        train = train.sort_values(["unique_id", "ds"])
        uids, contexts = [], []
        for uid, g in train.groupby("unique_id", sort=False):
            uids.append(uid)
            contexts.append(torch.tensor(g["y"].to_numpy(float), dtype=torch.float32))
        bounds = [0.5 - lv / 200 for lv in levels] + [0.5 + lv / 200 for lv in levels]
        q_levels = sorted({0.5, *bounds})
        frames = []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start : start + self.batch_size]
            quantiles, _mean = pipeline.predict_quantiles(
                batch, prediction_length=h, quantile_levels=q_levels
            )
            for i, uid in enumerate(uids[start : start + self.batch_size]):
                q = quantiles[i].reshape(h, len(q_levels)).numpy()
                col = {lv: q[:, j] for j, lv in enumerate(q_levels)}
                out = {"unique_id": uid, "step": np.arange(1, h + 1), "mean": col[0.5]}
                for level in levels:
                    out[f"lo-{level}"] = col[0.5 - level / 200]
                    out[f"hi-{level}"] = col[0.5 + level / 200]
                frames.append(pd.DataFrame(out))
        return pd.concat(frames, ignore_index=True)


def build(name: str, n_jobs: int = -1, seed: int = 42, calibration_origins: int = 1) -> Forecaster:
    """Instantiate a model by name. `calibration_origins` applies to `lgbm` (see LightGBMGlobal);
    the benchmark sets it per dataset."""
    if name in ("ets", "arima", "theta"):
        return StatsForecastModel(name, n_jobs)
    if name == "comb":
        return Combination(
            "comb", [StatsForecastModel(n, n_jobs) for n in ("ets", "arima", "theta")]
        )
    factories = {
        "chronos": ChronosModel,
        "lgbm_tuned": lambda: LightGBMTuned(
            seed=seed, n_jobs=n_jobs, calibration_origins=calibration_origins
        ),
        "naive": Naive,
        "snaive": SeasonalNaive,
        "naive2": Naive2,
        "sarima": lambda: Sarima(n_jobs),
        "prophet": lambda: ProphetModel(n_jobs),
        "lgbm": lambda: LightGBMGlobal(
            seed=seed, n_jobs=n_jobs, calibration_origins=calibration_origins
        ),
        # ablation: one conformal error distribution for all series (not in MODEL_NAMES)
        "lgbm_global": lambda: LightGBMGlobal(seed=seed, n_jobs=n_jobs, intervals="global"),
    }
    if name.startswith("lgbm_cal") and name[len("lgbm_cal") :].isdigit():
        # ablation: conformal errors from k origins (e.g. lgbm_cal4; not in MODEL_NAMES)
        model = LightGBMGlobal(seed=seed, n_jobs=n_jobs, calibration_origins=int(name[8:]))
        model.name = name
        return model
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
    "lgbm_tuned",
)
# Foundation models need the optional `fm` extra (torch + chronos-forecasting); run them with
# `--models ...,chronos`.
FM_MODEL_NAMES = ("chronos",)

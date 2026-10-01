# Time Series Forecasting Benchmark

A careful, reproducible comparison of classical, statistical, and ML forecasting methods. It uses rolling-origin evaluation, measured accuracy and runtime, and significance tests.

> Part of the Live Projects portfolio. See [SHARED_STANDARDS.md](../SHARED_STANDARDS.md) for conventions and the milestone definition of done.

## Overview

Forecasting papers and blog posts often skip strong simple baselines. This benchmark
compares **seasonal naive, ETS, ARIMA/SARIMA (statsmodels / statsforecast), Prophet, and
LightGBM with lag features** on M4/M5 subsets and public energy or retail series. Every
method goes through the same **rolling-origin cross-validation**.

Each method is scored on accuracy (MASE, sMAPE, RMSE), prediction-interval coverage,
and runtime per series. Diebold-Mariano tests check whether differences between methods
are significant. The README will document the cases where simple baselines win.

## Tech Stack

| Layer | Tools |
|---|---|
| Classical / statistical | statsmodels (ARIMA/SARIMA, ETS), statsforecast (AutoARIMA, AutoETS, SeasonalNaive) |
| Additive models | Prophet |
| ML | LightGBM with lag/calendar features |
| Evaluation | SciPy (Diebold-Mariano), custom MASE/sMAPE/coverage metrics |
| Data | pandas, pyarrow (parquet cache) |
| Reporting | matplotlib; leaderboard CSV/Markdown (Streamlit dashboard optional at M7) |

## Architecture

| Component | Responsibility |
|---|---|
| **Data loaders** | Download and cache M4/M5 subsets and energy/retail series as parquet in a standard long format (`unique_id, ds, y`) |
| **Model registry** | One wrapper per method with a common `fit / predict(h, level)` interface |
| **Backtester** | Rolling-origin CV: fixed horizon `h`, `n_windows` cutoffs, the same splits for every model |
| **Metrics** | MASE, sMAPE, RMSE, interval coverage, runtime per series |
| **Stat tests** | Diebold-Mariano pairwise tests against the seasonal-naive baseline |
| **Reporter** | Leaderboard tables, plots, and a Markdown report under `results/` |

```
 M4 / M5 / energy / retail ──► loaders ──► parquet cache (long format)
                                                 │
                                                 ▼
                           rolling-origin backtester (same cutoffs for all)
         ┌───────────────┬──────────────┬──────────┴─────┬──────────────┐
         ▼               ▼              ▼                ▼              ▼
  seasonal naive     ETS / ARIMA     statsforecast     Prophet       LightGBM
                    (statsmodels)    Auto* models                  (lag feats)
         └───────────────┴──────────────┴────────┬───────┴──────────────┘
                                                 ▼
                           metrics (MASE, sMAPE, RMSE, coverage, runtime)
                                                 ▼
                           Diebold-Mariano tests ──► leaderboard + report
```

## Setup

### Prerequisites
- Python 3.11
- A C/C++ build toolchain is usually **not** needed: Prophet, LightGBM, and statsforecast
  ship wheels for Python 3.11 on Windows, macOS, and Linux
- About 2 GB of free disk space for M4/M5 subsets
- Optional: GNU Make (see the Windows note below)

### Install
```bash
git clone <repo-url> 04-time-series-forecasting && cd 04-time-series-forecasting
py -3.11 -m venv .venv            # Windows; on macOS/Linux use: python3.11 -m venv .venv
source .venv/Scripts/activate     # Git Bash; PowerShell: .venv\Scripts\Activate.ps1; macOS/Linux: source .venv/bin/activate
make install                      # = pip install -e ".[dev]"
```

### Environment config
```bash
cp .env.example .env              # then fill in values; .env is git-ignored
```
The settings cover data and results folders, the forecast horizon, the number of
backtest windows, the random seed, and parallelism. No secrets are needed.
`KAGGLE_*` is needed only to download M5 from Kaggle.

## How to Run

> **Windows note:** `make` isn't installed on Windows by default. Install it with `choco install make`, or run the
> targets from **Git Bash**. You can also run the commands in the `Makefile` yourself
> (for example, `python -m pytest` instead of `make test`).

### Run the benchmark
```bash
make run                              # = python -m ts_forecasting.benchmark
```
> `make run` starts working at **M2**, once `src/ts_forecasting/benchmark.py` is added.
> Results are written to `results/` (git-ignored).

Planned CLI options: `--dataset m4-monthly --horizon 18 --windows 3 --models snaive,ets,arima,prophet,lgbm`.

### Lint and test
```bash
make lint     # ruff check + ruff format --check
make test     # pytest (unit tests only; integration tests are marked `integration`)
make format   # auto-fix lint and formatting
```

## Evaluation & Results

- **Baseline:** seasonal naive. Every method is reported relative to it (relative MASE).
- **Metrics:** MASE, sMAPE, RMSE, 80%/95% prediction-interval coverage, and fit+predict
  runtime per series.
- **Method:** rolling-origin CV with the same cutoffs for all models, plus pairwise
  Diebold-Mariano tests against the baseline.
- **Results:** *targets until measured*: a reproducible leaderboard and documented cases
  where simple baselines win.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [ ] Baseline: seasonal naive forecast + MASE on one dataset

### M2: Core functionality
- [ ] Common model interface; ETS, ARIMA/SARIMA, statsforecast Auto*, Prophet, LightGBM wrappers
- [ ] Rolling-origin backtester
- [ ] `python -m ts_forecasting.benchmark` CLI

### M3: Data / storage layer
- [ ] Data loaders for M4/M5 subsets and an energy/retail series
- [ ] Parquet cache in long format; dataset versions documented

### M4: Evaluation & testing
- [ ] Metrics: MASE, sMAPE, RMSE, interval coverage, runtime
- [ ] Diebold-Mariano significance tests vs seasonal naive
- [ ] Unit tests for metrics and backtest splits (no leakage)

### M5: Observability / monitoring
- [ ] Runtime/memory profiling per model; experiment logging (MLflow optional)
- [ ] Leaderboard plots

### M6: Containerization & CI/CD
- [ ] Dockerfile for reproducible runs
- [ ] GitHub Actions: lint + test + small benchmark smoke run

### M7: Docs, benchmarks & demo
- [ ] README with leaderboard and "where baselines win" analysis
- [ ] Benchmark report
- [ ] Streamlit results dashboard or recorded demo

## Project Layout
```
04-time-series-forecasting/
├── src/ts_forecasting/__init__.py
├── tests/test_smoke.py
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

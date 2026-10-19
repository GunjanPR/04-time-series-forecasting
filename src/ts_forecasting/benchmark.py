"""Run the rolling-origin benchmark and write a leaderboard.

    python -m ts_forecasting.benchmark
    python -m ts_forecasting.benchmark --models snaive,naive --horizon 18 --windows 3

Writes to `results/<dataset>/n<series>_h<h>_w<windows>_seed<seed>/`: `scores/<model>.csv`
(one row per series and window, reused on re-runs), `scores.csv`, `leaderboard.csv` and
`leaderboard.md`.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ts_forecasting.backtest import attribute_combination_cost, leaderboard, run_backtest
from ts_forecasting.config import Settings, get_settings
from ts_forecasting.data import DATASETS, calibration_origins, load, resolve
from ts_forecasting.models import MODEL_NAMES, build

log = logging.getLogger(__name__)


def run_dir(settings: Settings):
    settings = resolve(settings)
    n = settings.max_series or "all"
    jobs = "" if settings.n_jobs == -1 else f"_jobs{settings.n_jobs}"
    return (
        settings.results_dir
        / settings.dataset
        / f"n{n}_h{settings.horizon}_w{settings.n_windows}_seed{settings.random_seed}{jobs}"
    )


def benchmark(
    models: list[str], settings: Settings | None = None, force: bool = False
) -> pd.DataFrame:
    """Backtest each model, reusing per-model scores already saved for the same configuration
    (unless `force`), so an interrupted run resumes instead of starting over."""
    settings = resolve(settings or get_settings())
    out = run_dir(settings)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    df = load(settings)
    season = DATASETS[settings.dataset].season
    log.info(
        "%s: %d series, h=%d, %d windows",
        settings.dataset,
        df["unique_id"].nunique(),
        settings.horizon,
        settings.n_windows,
    )
    frames = []
    for name in models:
        path = out / "scores" / f"{name}.csv"
        fc_path = out / "forecasts" / f"{name}.parquet"
        if path.exists() and fc_path.exists() and not force:
            log.info("%s: reusing %s", name, path)
            frames.append(pd.read_csv(path))
            continue
        model = build(
            name,
            n_jobs=settings.n_jobs,
            seed=settings.random_seed,
            calibration_origins=calibration_origins(DATASETS[settings.dataset], settings.horizon),
        )
        scores, forecasts = run_backtest(df, model, settings.horizon, settings.n_windows, season)
        fc_path.parent.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(fc_path, index=False)
        scores.to_csv(path, index=False)
        frames.append(scores)
    scores = attribute_combination_cost(pd.concat(frames, ignore_index=True))
    board = leaderboard(scores)
    scores.to_csv(out / "scores.csv", index=False)
    board.to_csv(out / "leaderboard.csv")
    (out / "leaderboard.md").write_text(board.to_markdown(floatfmt=".3f"))
    log.info("leaderboard -> %s\n%s", out, board.to_string(float_format="%.3f"))
    if settings.mlflow_tracking_uri:
        log_to_mlflow(board, settings, out)
    return board


def log_to_mlflow(board: pd.DataFrame, settings: Settings, out) -> None:
    """One MLflow run per model (params = benchmark config, metrics = leaderboard row)."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"forecasting-{settings.dataset}")
    config = {
        "dataset": settings.dataset,
        "horizon": settings.horizon,
        "n_windows": settings.n_windows,
        "max_series": settings.max_series,
        "seed": settings.random_seed,
    }
    for model, row in board.iterrows():
        with mlflow.start_run(run_name=f"{model}-{out.name}"):
            mlflow.log_params({"model": model, **config})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if pd.notna(v)})
            mlflow.log_artifact(str(out / "scores" / f"{model}.csv"))
    log.info("logged %d runs to MLflow at %s", len(board), settings.mlflow_tracking_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--windows", type=int, dest="n_windows")
    parser.add_argument("--max-series", type=int)
    parser.add_argument(
        "--n-jobs", type=int, help="worker processes (-1 = all cores; 1 = single-core cost)"
    )
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--force", action="store_true", help="recompute saved model scores")
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "models" and v is not None}
    settings = get_settings().model_copy(update=overrides)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    benchmark(args.models.split(","), settings, force=args.force)


if __name__ == "__main__":
    main()

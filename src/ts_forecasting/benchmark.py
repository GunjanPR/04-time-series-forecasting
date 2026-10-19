"""Run the rolling-origin benchmark and write a leaderboard.

    python -m ts_forecasting.benchmark
    python -m ts_forecasting.benchmark --models snaive,naive --horizon 18 --windows 3

Writes `results/<dataset>/scores.csv` (one row per model, series and window),
`leaderboard.csv` and `leaderboard.md`.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ts_forecasting.backtest import leaderboard, run_backtest
from ts_forecasting.config import Settings, get_settings
from ts_forecasting.data import SEASON_LENGTH, load
from ts_forecasting.models import MODELS

log = logging.getLogger(__name__)


def benchmark(models: list[str], settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    df = load(settings)
    season = SEASON_LENGTH[settings.dataset]
    log.info(
        "%s: %d series, h=%d, %d windows",
        settings.dataset,
        df["unique_id"].nunique(),
        settings.horizon,
        settings.n_windows,
    )
    scores = pd.concat(
        [
            run_backtest(df, MODELS[name](), settings.horizon, settings.n_windows, season)
            for name in models
        ],
        ignore_index=True,
    )
    board = leaderboard(scores)
    out = settings.results_dir / settings.dataset
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    board.to_csv(out / "leaderboard.csv")
    (out / "leaderboard.md").write_text(board.to_markdown(floatfmt=".3f"))
    log.info("leaderboard -> %s\n%s", out, board.to_string(float_format="%.3f"))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(SEASON_LENGTH))
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--windows", type=int, dest="n_windows")
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "models" and v is not None}
    settings = get_settings().model_copy(update=overrides)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    benchmark(args.models.split(","), settings)


if __name__ == "__main__":
    main()

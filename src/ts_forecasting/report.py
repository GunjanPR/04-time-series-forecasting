"""Build the benchmark report (tables + plots) from a finished benchmark run.

    python -m ts_forecasting.report                  # same settings as the benchmark
    python -m ts_forecasting.report --publish docs/results

Reads `scores/*.csv` and `forecasts/*.parquet` from the run directory and writes
`report.md`, `significance.csv` and PNG plots next to them. `--publish` also copies the
Markdown and plots to a folder that is committed to git.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ts_forecasting.backtest import attribute_combination_cost, leaderboard  # noqa: E402
from ts_forecasting.config import Settings, get_settings  # noqa: E402
from ts_forecasting.stats import (  # noqa: E402
    dm_summary,
    friedman_nemenyi,
    wilcoxon_summary,
)

log = logging.getLogger(__name__)

BASELINE = "snaive"
LABELS = {
    "snaive": "Seasonal naive",
    "naive": "Naive",
    "naive2": "Naive2 (M4)",
    "ets": "AutoETS",
    "arima": "AutoARIMA",
    "theta": "AutoTheta",
    "comb": "Comb (ETS+ARIMA+Theta)",
    "sarima": "SARIMA airline",
    "prophet": "Prophet",
    "lgbm": "LightGBM global",
}


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = attribute_combination_cost(
        pd.concat([pd.read_csv(p) for p in sorted((run_dir / "scores").glob("*.csv"))])
    )
    forecasts = pd.concat(
        [pd.read_parquet(p) for p in sorted((run_dir / "forecasts").glob("*.parquet"))]
    )
    return scores, forecasts


def plot_relative_mase(table: pd.DataFrame, path: Path) -> None:
    order = table.sort_values("mase_diff")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1))
    y = np.arange(len(order))
    err = [
        order["mase_diff"] - order["mase_diff_ci_low"],
        order["mase_diff_ci_high"] - order["mase_diff"],
    ]
    low, high = order["mase_diff_ci_low"], order["mase_diff_ci_high"]
    # blue = significantly better, red = significantly worse, grey = CI spans zero
    colors = np.where(high < 0, "#2a7ab9", np.where(low > 0, "#c0504d", "#999999"))
    ax.barh(y, order["mase_diff"], xerr=err, color=colors, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [LABELS.get(m, m) for m in order.index])
    ax.invert_yaxis()
    ax.set_xlabel("Mean MASE difference vs seasonal naive (95% bootstrap CI; < 0 is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ranks(ranks: pd.Series, cd: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(ranks) + 1))
    y = np.arange(len(ranks))
    ax.errorbar(ranks, y, xerr=cd / 2, fmt="o", color="#2a7ab9", capsize=4)
    ax.set_yticks(y, [LABELS.get(m, m) for m in ranks.index])
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean rank on per-series MASE (bars = Nemenyi CD/2; CD = {cd:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coverage(board: pd.DataFrame, path: Path) -> None:
    levels = sorted(int(c.split("_")[1]) for c in board.columns if c.startswith("coverage_"))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(board))
    width = 0.8 / len(levels)
    for i, level in enumerate(levels):
        ax.bar(x + i * width, 100 * board[f"coverage_{level}"], width, label=f"{level}% interval")
        ax.axhline(level, color="black", lw=0.6, ls="--")
    ax.set_xticks(x + width * (len(levels) - 1) / 2, [LABELS.get(m, m) for m in board.index],
                  rotation=35, ha="right")  # fmt: skip
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_ylim(40, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_accuracy_vs_cost(board: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(board["ms_per_series"], board["mase"], color="#2a7ab9")
    for model, row in board.iterrows():
        ax.annotate(LABELS.get(model, model), (row["ms_per_series"], row["mase"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Fit + predict time per series (ms, log scale)")
    ax.set_ylabel("Mean MASE")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mase_by_step(forecasts: pd.DataFrame, path: Path) -> None:
    loss = forecasts.assign(err=np.abs(forecasts["y"] - forecasts["mean"]) / forecasts["scale"])
    curve = loss.groupby(["model", "step"])["err"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in curve.columns:
        ax.plot(curve.index, curve[model], label=LABELS.get(model, model),
                lw=2.2 if model == BASELINE else 1.2)  # fmt: skip
    ax.set_xlabel("Forecast step (months ahead)")
    ax.set_ylabel("Mean absolute scaled error")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(settings: Settings | None = None, publish: Path | None = None) -> Path:
    from ts_forecasting.benchmark import run_dir

    settings = settings or get_settings()
    out = run_dir(settings)
    scores, forecasts = load_run(out)
    board = leaderboard(scores, baseline=BASELINE)
    significance = wilcoxon_summary(scores, baseline=BASELINE, seed=settings.random_seed).join(
        dm_summary(forecasts, baseline=BASELINE)
    )
    ranks, friedman_p, cd = friedman_nemenyi(scores)
    significance["mean_rank"] = ranks
    significance.to_csv(out / "significance.csv")

    plots = {
        "mase_vs_baseline.png": lambda p: plot_relative_mase(significance, p),
        "ranks.png": lambda p: plot_ranks(ranks, cd, p),
        "coverage.png": lambda p: plot_coverage(board, p),
        "accuracy_vs_cost.png": lambda p: plot_accuracy_vs_cost(board, p),
        "mase_by_step.png": lambda p: plot_mase_by_step(forecasts, p),
    }
    for name, draw in plots.items():
        draw(out / name)

    n_series = scores["unique_id"].nunique()
    n_windows = scores["window"].nunique()
    show = board.rename(index=LABELS)
    lines = [
        f"# Benchmark report: {settings.dataset}",
        "",
        f"{n_series} series, horizon {settings.horizon}, {n_windows} rolling-origin windows "
        f"({len(scores) // board.shape[0]} series-windows per model), seed {settings.random_seed}. "
        "Baseline: seasonal naive. Lower is better for every error metric.",
        "",
        "## Leaderboard",
        "",
        show.to_markdown(floatfmt=".3f"),
        "",
        "`owa` = M4 Overall Weighted Average vs Naive2; `msis_95` = M4 mean scaled interval score.",
        "",
        "## Significance vs seasonal naive",
        "",
        significance.rename(index=LABELS).to_markdown(floatfmt=".3g"),
        "",
        f"Friedman test across all models: p = {friedman_p:.2g}. Nemenyi critical difference "
        f"at 5%: {cd:.3f} mean ranks.",
        "",
        "`mase_diff` = mean per-series MASE difference (model - baseline) with a 95% series "
        "bootstrap CI; `win_rate_pct` = share of series where the model's MASE is lower; "
        "`dm_better_pct` / `dm_worse_pct` = share of series where a Diebold-Mariano test "
        "(absolute scaled error, HAC variance, HLN correction) finds the model significantly "
        "better / worse at 5%.",
        "",
        *[f"![{name}]({name})" for name in plots],
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("report -> %s", report)

    if publish:
        publish.mkdir(parents=True, exist_ok=True)
        for name in ["report.md", "significance.csv", *plots]:
            shutil.copyfile(out / name, publish / name)
        board.to_csv(publish / "leaderboard.csv")
        log.info("published -> %s", publish)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--publish", type=Path, help="also copy report + plots here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_report(publish=args.publish)


if __name__ == "__main__":
    main()

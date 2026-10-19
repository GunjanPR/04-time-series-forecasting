"""Where do simple baselines win? Relate per-series results to series characteristics.

    python -m ts_forecasting.analysis --challenger comb --simple naive2 --publish docs/results

For every series, features are computed on its history before the first backtest window
(so they use no test data): STL trend and seasonal strength (Wang, Smith & Hyndman, 2006),
length, and the volatility of month-on-month changes. Series are grouped into quintiles of
each feature and, per quintile, we report how often the simple method beats the challenger
on per-series MASE and the mean MASE of both.
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

from ts_forecasting.config import Settings, get_settings  # noqa: E402
from ts_forecasting.data import SEASON_LENGTH, load  # noqa: E402

log = logging.getLogger(__name__)

FEATURES = {
    "seasonal_strength": "Seasonal strength (STL)",
    "trend_strength": "Trend strength (STL)",
    "length": "History length (months)",
    "volatility": "Volatility of monthly % changes",
}


def series_features(y: np.ndarray, season: int) -> dict[str, float]:
    from statsmodels.tsa.seasonal import STL

    fit = STL(y, period=season, robust=True).fit()
    remainder = np.var(fit.resid)

    def strength(component: np.ndarray) -> float:
        total = np.var(component + fit.resid)
        return float(max(0.0, 1 - remainder / total)) if total > 0 else 0.0

    changes = np.diff(y) / np.where(y[:-1] == 0, 1, np.abs(y[:-1]))
    return {
        "seasonal_strength": strength(fit.seasonal),
        "trend_strength": strength(fit.trend),
        "length": float(len(y)),
        "volatility": float(np.std(changes)),
    }


def feature_table(df: pd.DataFrame, holdout: int, season: int) -> pd.DataFrame:
    rows = {}
    for uid, group in df.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        history = group["y"].to_numpy(float)[:-holdout]
        rows[uid] = series_features(history, season)
    return pd.DataFrame.from_dict(rows, orient="index")


def win_table(
    features: pd.DataFrame, per_series: pd.DataFrame, simple: str, challenger: str
) -> pd.DataFrame:
    """Per feature quintile: share of series where `simple` beats `challenger`, and MASEs."""
    joined = features.join(per_series[[simple, challenger]], how="inner")
    joined["simple_wins"] = joined[simple] < joined[challenger]
    parts = []
    for feature in FEATURES:
        bins = pd.qcut(joined[feature].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        grouped = joined.groupby(bins, observed=True).agg(
            low=(feature, "min"),
            high=(feature, "max"),
            series=(feature, "size"),
            simple_win_pct=("simple_wins", lambda s: 100 * s.mean()),
            simple_mase=(simple, "mean"),
            challenger_mase=(challenger, "mean"),
        )
        parts.append(grouped.assign(feature=feature).reset_index(names="quintile"))
    return pd.concat(parts, ignore_index=True)


def plot_wins(table: pd.DataFrame, simple: str, challenger: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(13, 3.2), sharey=True)
    for ax, (feature, label) in zip(axes, FEATURES.items(), strict=True):
        part = table[table["feature"] == feature]
        ax.bar(part["quintile"].astype(int), part["simple_win_pct"], color="#2a7ab9")
        ax.axhline(50, color="black", lw=0.6, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("quintile (1 = lowest)", fontsize=8)
    axes[0].set_ylabel(f"% series where {simple} beats {challenger}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyse(
    settings: Settings | None = None,
    simple: str = "naive2",
    challenger: str = "comb",
    publish: Path | None = None,
) -> pd.DataFrame:
    from ts_forecasting.benchmark import run_dir
    from ts_forecasting.report import load_run
    from ts_forecasting.stats import per_series_mase

    settings = settings or get_settings()
    out = run_dir(settings)
    scores, _ = load_run(out)
    per_series = per_series_mase(scores)
    holdout = settings.horizon * settings.n_windows
    features = feature_table(load(settings), holdout, SEASON_LENGTH[settings.dataset])
    table = win_table(features, per_series, simple, challenger)

    stem = f"baseline_wins_{simple}_vs_{challenger}"
    table.to_csv(out / f"{stem}.csv", index=False)
    plot_wins(table, simple, challenger, out / f"{stem}.png")
    (out / f"{stem}.md").write_text(
        table.to_markdown(index=False, floatfmt=".3g"), encoding="utf-8"
    )
    if publish:
        publish.mkdir(parents=True, exist_ok=True)
        for suffix in (".csv", ".png", ".md"):
            shutil.copyfile(out / f"{stem}{suffix}", publish / f"{stem}{suffix}")
    log.info("\n%s", table.to_string(float_format="%.3f"))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simple", default="naive2")
    parser.add_argument("--challenger", default="comb")
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    analyse(simple=args.simple, challenger=args.challenger, publish=args.publish)


if __name__ == "__main__":
    main()

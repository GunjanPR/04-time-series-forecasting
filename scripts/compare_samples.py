"""Compare the M4-Monthly leaderboard on two independent 500-series samples.

    # second sample (writes results_seed7/, then the published report)
    RANDOM_SEED=7 RESULTS_DIR=./results_seed7 python -m ts_forecasting.benchmark \
        --dataset m4-monthly
    RANDOM_SEED=7 RESULTS_DIR=./results_seed7 python -m ts_forecasting.report \
        --dataset m4-monthly --publish docs/results/m4-monthly-seed7
    python scripts/compare_samples.py   # -> docs/results/m4-monthly-seed7/sample_comparison.csv
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

FIRST = Path("docs/results/m4-monthly")
SECOND = Path("docs/results/m4-monthly-seed7")
SAMPLES = {
    42: Path("data/processed/m4-monthly_n500_seed42_min90.parquet"),
    7: Path("data/processed/m4-monthly_n500_seed7_min90.parquet"),
}


def main() -> None:
    a = pd.read_csv(FIRST / "leaderboard.csv").set_index("model")
    b = pd.read_csv(SECOND / "leaderboard.csv").set_index("model")
    if all(p.exists() for p in SAMPLES.values()):
        ids = [set(pd.read_parquet(p)["unique_id"].unique()) for p in SAMPLES.values()]
        print(f"series in both samples: {len(ids[0] & ids[1])} of {len(ids[0])}")
    models = [m for m in a.index if m in b.index and m != "lgbm_global"]
    table = pd.DataFrame(
        {
            "mase_seed42": a.loc[models, "mase"],
            "mase_seed7": b.loc[models, "mase"],
            "vs_snaive_seed42": a.loc[models, "mase"] / a.loc["snaive", "mase"] - 1,
            "vs_snaive_seed7": b.loc[models, "mase"] / b.loc["snaive", "mase"] - 1,
            "coverage95_seed42": a.loc[models, "coverage_95"],
            "coverage95_seed7": b.loc[models, "coverage_95"],
            "msis95_seed42": a.loc[models, "msis_95"],
            "msis95_seed7": b.loc[models, "msis_95"],
        }
    )
    table["rank_seed42"] = table["mase_seed42"].rank().astype(int)
    table["rank_seed7"] = table["mase_seed7"].rank().astype(int)
    table = table.sort_values("mase_seed42")
    print(table.round(3).to_string())
    rho, p = spearmanr(table["mase_seed42"], table["mase_seed7"])
    print(f"Spearman rank correlation of MASE across samples: {rho:.3f} (p = {p:.2g})")
    table.round(4).to_csv(SECOND / "sample_comparison.csv")


if __name__ == "__main__":
    main()

"""Significance tests for forecast comparisons.

Three complementary views, all against the same baseline:

* **Diebold-Mariano, per series.** Loss = absolute scaled error over every backtest step of a
  series (n_windows * h points). Errors from one forecast origin are autocorrelated, so the
  variance uses a Bartlett (Newey-West) HAC estimate, with the Harvey-Leybourne-Newbold
  small-sample correction and a t(n-1) reference. Reported as the share of series where a
  model is significantly better or worse than the baseline.
* **Wilcoxon signed-rank, across series.** Paired test on per-series mean MASE, Holm-corrected
  across models, with a series-bootstrap CI for the mean MASE difference.
* **Friedman + Nemenyi, all models together.** Mean rank of each model on per-series MASE and
  the Nemenyi critical difference: two models whose mean ranks differ by less than the CD are
  not significantly different (Demsar, 2006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano test of equal expected loss. Returns `(statistic, two-sided p)`.

    A negative statistic means `loss_a` is smaller (model A is better)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if lag is None:
        lag = int(np.floor(n ** (1 / 3)))
    d_mean = d.mean()
    centered = d - d_mean
    gamma0 = np.dot(centered, centered) / n
    variance = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.dot(centered[k:], centered[:-k]) / n
        variance += 2 * (1 - k / (lag + 1)) * gamma_k
    if variance <= 0 or not np.isfinite(variance):
        return 0.0, 1.0
    dm = d_mean / np.sqrt(variance / n)
    # Harvey, Leybourne & Newbold (1997) correction, with lag + 1 playing the role of h.
    h = lag + 1
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = float(dm * correction)
    p_value = float(2 * stats.t.sf(abs(statistic), df=n - 1))
    return statistic, p_value


def scaled_abs_errors(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, series, window, step) with the absolute scaled error."""
    out = forecasts[["model", "unique_id", "window", "step"]].copy()
    out["loss"] = np.abs(forecasts["y"] - forecasts["mean"]) / forecasts["scale"]
    return out


def dm_summary(
    forecasts: pd.DataFrame, baseline: str = "snaive", alpha: float = 0.05
) -> pd.DataFrame:
    """Per model: share of series significantly better / worse than the baseline (DM test)."""
    losses = scaled_abs_errors(forecasts).sort_values(["unique_id", "window", "step"])
    wide = losses.pivot_table(index=["unique_id", "window", "step"], columns="model", values="loss")
    rows = []
    for model in wide.columns.drop(baseline):
        better = worse = 0
        n_series = 0
        for _, group in wide[[model, baseline]].groupby(level="unique_id"):
            statistic, p = dm_test(group[model].to_numpy(), group[baseline].to_numpy())
            n_series += 1
            if p < alpha:
                better += statistic < 0
                worse += statistic > 0
        rows.append(
            {
                "model": model,
                "dm_better_pct": 100 * better / n_series,
                "dm_worse_pct": 100 * worse / n_series,
                "dm_series": n_series,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = p_values.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p_values.index)


def per_series_mase(scores: pd.DataFrame) -> pd.DataFrame:
    """Series x model table of MASE averaged over backtest windows."""
    return scores.pivot_table(index="unique_id", columns="model", values="mase", aggfunc="mean")


def wilcoxon_summary(
    scores: pd.DataFrame, baseline: str = "snaive", n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per model vs baseline: mean MASE difference with a series-bootstrap 95% CI, share of
    series won, and a Holm-corrected Wilcoxon signed-rank p-value."""
    table = per_series_mase(scores)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(table), (n_boot, len(table)))
    rows = []
    for model in table.columns.drop(baseline):
        diff = (table[model] - table[baseline]).to_numpy()
        boot = diff[idx].mean(axis=1)
        p = stats.wilcoxon(table[model], table[baseline]).pvalue if np.any(diff) else 1.0
        rows.append(
            {
                "model": model,
                "mase_diff": diff.mean(),
                "mase_diff_ci_low": np.percentile(boot, 2.5),
                "mase_diff_ci_high": np.percentile(boot, 97.5),
                "win_rate_pct": 100 * np.mean(diff < 0),
                "wilcoxon_p": p,
            }
        )
    out = pd.DataFrame(rows).set_index("model")
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    return out


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.Series, float, float]:
    """Mean rank per model (1 = best), Friedman p-value and the Nemenyi critical difference."""
    table = per_series_mase(scores).dropna()
    n, k = table.shape
    ranks = table.rank(axis=1).mean().sort_values()
    p_value = float(stats.friedmanchisquare(*[table[c] for c in table.columns]).pvalue)
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6 * n)))
    return ranks, p_value, cd

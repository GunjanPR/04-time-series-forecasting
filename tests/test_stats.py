import numpy as np
import pandas as pd
import pytest

from ts_forecasting.stats import (
    dm_summary,
    dm_test,
    friedman_nemenyi,
    holm,
    wilcoxon_summary,
)


def test_dm_detects_clearly_smaller_loss():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 2, 60)
    statistic, p = dm_test(base * 0.5, base)
    assert statistic < 0 and p < 0.001


def test_dm_equal_losses_is_not_significant():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 0.3, 60), rng.normal(1, 0.3, 60)
    _, p = dm_test(a, b)
    assert p > 0.05
    assert dm_test(a, a) == (0.0, 1.0)


def test_dm_false_positive_rate_is_near_alpha():
    rng = np.random.default_rng(2)
    rejections = [dm_test(rng.normal(size=54), rng.normal(size=54))[1] < 0.05 for _ in range(400)]
    assert 0.02 < np.mean(rejections) < 0.09


def test_holm_matches_hand_computation():
    adjusted = holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03}))
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # monotone: never below a smaller p's value


def _scores_and_forecasts(n_series=40, h=6, windows=2, seed=3):
    """'good' has half the error of 'snaive'; 'same' matches it on average."""
    rng = np.random.default_rng(seed)
    score_rows, fc_rows = [], []
    for s in range(n_series):
        for w in range(windows):
            base = rng.uniform(0.5, 1.5, h)
            errors = {"snaive": base, "good": base * 0.5, "same": base * rng.uniform(0.8, 1.2, h)}
            for model, err in errors.items():
                score_rows.append(
                    {"model": model, "unique_id": f"S{s}", "window": w, "mase": err.mean()}
                )
                for step, e in enumerate(err, 1):
                    fc_rows.append(
                        {"model": model, "unique_id": f"S{s}", "window": w, "step": step,
                         "y": 10.0, "mean": 10.0 + e, "scale": 1.0}
                    )  # fmt: skip
    return pd.DataFrame(score_rows), pd.DataFrame(fc_rows)


def test_summaries_rank_the_better_model_first():
    scores, forecasts = _scores_and_forecasts()
    dm = dm_summary(forecasts)
    assert dm.loc["good", "dm_better_pct"] > 90
    assert dm.loc["same", "dm_better_pct"] < 20

    wil = wilcoxon_summary(scores, n_boot=300)
    assert wil.loc["good", "mase_diff_ci_high"] < 0
    assert wil.loc["good", "wilcoxon_p_holm"] < 0.001
    assert wil.loc["same", "mase_diff_ci_low"] < 0 < wil.loc["same", "mase_diff_ci_high"]

    ranks, p, cd = friedman_nemenyi(scores)
    assert ranks.index[0] == "good"
    assert p < 0.001
    assert ranks["snaive"] - ranks["good"] > cd

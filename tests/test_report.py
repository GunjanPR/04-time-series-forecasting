"""End to end: benchmark (on synthetic series) -> resume -> report."""

import pandas as pd

import ts_forecasting.benchmark as bench
from conftest import seasonal_frame
from ts_forecasting.config import Settings
from ts_forecasting.report import build_report


def test_benchmark_resumes_and_report_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load", lambda settings: seasonal_frame(n_series=12, length=72))
    settings = Settings(
        results_dir=tmp_path / "results", horizon=6, n_windows=2, n_jobs=1, _env_file=None
    )
    models = ["snaive", "naive", "naive2", "theta"]

    board = bench.benchmark(models, settings)
    assert set(board.index) == set(models)
    assert {"rel_mase", "owa", "msis_95"} <= set(board.columns)

    calls = []
    monkeypatch.setattr(bench, "run_backtest", lambda *a, **k: calls.append(a))
    again = bench.benchmark(models, settings)
    assert not calls  # every model was reused from disk
    pd.testing.assert_frame_equal(board, again)

    publish = tmp_path / "docs"
    report = build_report(settings, publish=publish)
    text = report.read_text(encoding="utf-8")
    assert "## Leaderboard" in text and "Nemenyi" in text
    for name in ["report.md", "significance.csv", "ranks.png", "mase_vs_baseline.png"]:
        assert (publish / name).exists()
    significance = pd.read_csv(publish / "significance.csv", index_col=0)
    assert "snaive" not in significance.index
    assert {"wilcoxon_p_holm", "dm_better_pct", "mean_rank"} <= set(significance.columns)

"""Streamlit results dashboard.

    streamlit run src/ts_forecasting/dashboard.py          # or: make dashboard

Shows the published results in `docs/results/<dataset>/` (committed, so the dashboard works
on a fresh clone) and, when a local benchmark run exists under `results/`, an interactive
explorer of every series' backtest forecasts and intervals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ts_forecasting.report import LABELS

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "docs" / "results"
RESULTS = ROOT / "results"


def label(model: str) -> str:
    return LABELS.get(model, model)


@st.cache_data
def read_csv(path: Path, index_col=None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


@st.cache_data
def load_forecasts(run: Path) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(p) for p in sorted((run / "forecasts").glob("*.parquet")))


@st.cache_data
def load_history(pattern: str) -> pd.DataFrame | None:
    files = sorted((ROOT / "data" / "processed").glob(pattern))
    return pd.read_parquet(files[-1]) if files else None


def leaderboard_tab(published: Path) -> None:
    board = read_csv(published / "leaderboard.csv", index_col=0)
    st.subheader("Leaderboard")
    st.caption("Means over every series and backtest window. Lower is better.")
    st.dataframe(board.rename(index=label).style.format(precision=3), use_container_width=True)
    left, right = st.columns(2)
    left.image(str(published / "mase_vs_baseline.png"))
    right.image(str(published / "accuracy_vs_cost.png"))
    left.image(str(published / "coverage.png"))
    right.image(str(published / "mase_by_step.png"))


def significance_tab(published: Path) -> None:
    sig = read_csv(published / "significance.csv", index_col=0)
    st.subheader("Is the difference real?")
    st.dataframe(sig.rename(index=label).style.format(precision=3), use_container_width=True)
    st.image(str(published / "ranks.png"))


def baselines_tab(published: Path) -> None:
    st.subheader("Where simple baselines win")
    for path in sorted(published.glob("baseline_wins_*.png")):
        simple, challenger = path.stem.removeprefix("baseline_wins_").split("_vs_")
        st.markdown(f"**{label(simple)}** vs **{label(challenger)}**")
        st.image(str(path))


def explorer_tab(dataset: str) -> None:
    runs = sorted(p for p in (RESULTS / dataset).glob("*") if (p / "forecasts").exists())
    if not runs:
        st.info("Run `make run` to create a local benchmark run; its forecasts appear here.")
        return
    run = st.selectbox("Benchmark run", runs, format_func=lambda p: p.name)
    forecasts = load_forecasts(run)
    history = load_history(f"{run.parent.name}_*.parquet")
    models = sorted(forecasts["model"].unique())
    left, mid, right = st.columns([2, 1, 3])
    uid = left.selectbox("Series", sorted(forecasts["unique_id"].unique()))
    window = mid.selectbox("Window", sorted(forecasts["window"].unique()), index=0)
    chosen = right.multiselect(
        "Models", models, default=[m for m in ("snaive", "comb", "lgbm") if m in models],
        format_func=label,
    )  # fmt: skip
    level = st.radio("Interval", [80, 95], horizontal=True)
    part = forecasts[(forecasts["unique_id"] == uid) & (forecasts["window"] == window)]
    chart = part[part["model"] == models[0]][["step", "y"]].rename(columns={"y": "actual"})
    for model in chosen:
        m = part[part["model"] == model].set_index("step")
        chart[label(model)] = m["mean"].reindex(chart["step"]).to_numpy()
    st.line_chart(chart.set_index("step"))
    if history is not None:
        st.caption("Full history of the series (the backtest covers the last points).")
        st.line_chart(history[history["unique_id"] == uid].set_index("ds")["y"])
    rows = []
    for model in chosen:
        m = part[part["model"] == model]
        inside = (m["y"] >= m[f"lo-{level}"]) & (m["y"] <= m[f"hi-{level}"])
        mae = (m["y"] - m["mean"]).abs().mean()
        rows.append({"model": label(model), "MASE": mae / m["scale"].iloc[0],
                     f"{level}% coverage": inside.mean()})  # fmt: skip
    st.dataframe(pd.DataFrame(rows).set_index("model").style.format(precision=3))


def main() -> None:
    st.set_page_config(page_title="Forecasting benchmark", layout="wide")
    st.title("Time series forecasting benchmark")
    datasets = sorted(p.name for p in PUBLISHED.iterdir() if (p / "leaderboard.csv").exists())
    dataset = st.sidebar.radio("Dataset", datasets)
    published = PUBLISHED / dataset
    report = published / "report.md"
    if report.exists():
        st.caption(report.read_text(encoding="utf-8").splitlines()[2])
    tabs = st.tabs(["Leaderboard", "Significance", "Where baselines win", "Series explorer"])
    with tabs[0]:
        leaderboard_tab(published)
    with tabs[1]:
        significance_tab(published)
    with tabs[2]:
        baselines_tab(published)
    with tabs[3]:
        explorer_tab(dataset)


main()

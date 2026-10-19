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
def load_history(dataset: str) -> pd.DataFrame | None:
    """Every cached sample of the dataset, so any benchmarked series can be found."""
    files = sorted((ROOT / "data" / "processed").glob(f"{dataset}_*.parquet"))
    if not files:
        return None
    frames = pd.concat(pd.read_parquet(f) for f in files)
    return frames.drop_duplicates(["unique_id", "ds"])


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
    import altair as alt

    runs = sorted(
        (p for p in (RESULTS / dataset).glob("*") if (p / "forecasts").exists()),
        key=lambda p: ("jobs" in p.name, p.name),  # the main run first, cost profiles last
    )
    if not runs:
        st.info("Run `make run` to create a local benchmark run; its forecasts appear here.")
        return
    run = st.selectbox("Benchmark run", runs, format_func=lambda p: p.name)
    forecasts = load_forecasts(run)
    history = load_history(dataset)
    models = sorted(forecasts["model"].unique())
    left, mid, right = st.columns([2, 1, 3])
    uid = left.selectbox("Series", sorted(forecasts["unique_id"].unique()))
    windows = sorted(forecasts["window"].unique())
    window = mid.selectbox("Window", windows, index=len(windows) - 1)
    chosen = right.multiselect(
        "Models", models, default=[m for m in ("snaive", "comb", "lgbm") if m in models],
        format_func=label,
    )  # fmt: skip
    level = st.radio("Interval", [80, 95], horizontal=True)
    part = forecasts[(forecasts["unique_id"] == uid) & (forecasts["window"] == window)]
    first = part[part["model"] == part["model"].iloc[0]].sort_values("step")
    h = len(first)

    # Dates for the forecast steps: the last h points before the cutoff of this window.
    series = history[history["unique_id"] == uid].sort_values("ds") if history is not None else None
    if series is not None and len(series):
        n_after = (len(windows) - 1 - windows.index(window)) * h
        end = len(series) - n_after
        dates = series["ds"].iloc[end - h : end].to_numpy()
        past = series.iloc[max(0, end - h - 4 * h) : end - h]
    else:
        dates, past = first["step"].to_numpy(), None

    layers = []
    if past is not None:
        layers.append(
            alt.Chart(past.assign(kind="history"))
            .mark_line(color="#888")
            .encode(x=alt.X("ds:T", title=None), y=alt.Y("y:Q", scale=alt.Scale(zero=False)))
        )
    actual = pd.DataFrame({"ds": dates, "y": first["y"].to_numpy()})
    lines = []
    for model in chosen:
        m = part[part["model"] == model].sort_values("step")
        lines.append(pd.DataFrame({"ds": dates, "y": m["mean"].to_numpy(), "series": label(model)}))
        band = pd.DataFrame(
            {
                "ds": dates,
                "lo": m[f"lo-{level}"].to_numpy(),
                "hi": m[f"hi-{level}"].to_numpy(),
                "series": label(model),
            }
        )
        layers.append(
            alt.Chart(band)
            .mark_area(opacity=0.12)
            .encode(
                x="ds:T",
                y=alt.Y("lo:Q", scale=alt.Scale(zero=False)),
                y2="hi:Q",
                color=alt.Color("series:N", legend=None),
            )  # fmt: skip
        )
    if lines:
        layers.append(
            alt.Chart(pd.concat(lines))
            .mark_line(point=True, strokeDash=[5, 3])
            .encode(
                x="ds:T",
                y=alt.Y("y:Q", title="value", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None),
            )
        )
    layers.append(
        alt.Chart(actual)
        .mark_line(point=True, color="black")
        .encode(x="ds:T", y=alt.Y("y:Q", scale=alt.Scale(zero=False)))
    )
    st.altair_chart(alt.layer(*layers).properties(height=420), use_container_width=True)
    st.caption(
        f"Grey: history before the cutoff. Bands: {level}% prediction intervals. "
        "Dashed: forecasts; black: what actually happened."
    )

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
    default = datasets.index("m4-monthly") if "m4-monthly" in datasets else 0
    dataset = st.sidebar.radio("Dataset", datasets, index=default)
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

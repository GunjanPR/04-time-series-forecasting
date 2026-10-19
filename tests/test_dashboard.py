from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "ts_forecasting" / "dashboard.py"


def test_dashboard_renders_published_results_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Time series forecasting benchmark"
    assert len(app.tabs) == 4
    assert len(app.dataframe) >= 2  # leaderboard + significance tables

"""Streamlit app smoke test (AppTest): boots, loads the real bundle, runs one analysis, renders every tab."""
from pathlib import Path

import pytest

from nexus_alpha.inference.pipeline import latest_bundle

from .conftest import source_data

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@source_data
@pytest.mark.slow
@pytest.mark.skipif(latest_bundle() is None, reason="no inference bundle built")
def test_app_runs_an_analysis():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].set_value("MSFT")
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert "res" in at.session_state
    res = at.session_state["res"]
    assert set(res.forecasts) == {"1D", "2D", "5D", "10D", "20D", "1M"}
    assert len(at.tabs) == 12

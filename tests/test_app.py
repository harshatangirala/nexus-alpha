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
    # at.tabs also lists nested tabs (Desk 1-5 has one inputs tab per stock): check the 19 top-level tabs in order
    nested = {"Macro and market (shared)"}
    top = [t.label for t in at.tabs if not t.label.endswith(" inputs") and t.label not in nested]
    assert top == ["Mission Control", "Trade Desk", "Compare 1-5", "Desk 1-5", "A Forecast", "B Charts", "C News", "D SEC",
                   "E Indicators", "F Explain", "G Trade planner", "H Model diagnostics", "I Backtesting",
                   "J Knowledge base", "K Configuration", "L Audit", "M LLM committee", "N Options", "O Vision + NLP"]



@source_data
@pytest.mark.slow
@pytest.mark.skipif(latest_bundle() is None, reason="no inference bundle built")
def test_desk_1_to_5_reads_dated_news():
    """Desk 1-5: one stock, one article stamped in New York time, a historical as-of date -> board, plan, chronology."""
    import datetime as dt
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.multiselect(key="dk_picks").set_value(["MSFT"])
    at.radio(key="dk_mode").set_value("historical")
    at.run()                                       # the as-of date is enabled only in historical mode
    at.date_input(key="dk_asof").set_value(dt.date(2021, 6, 15))
    at.run()
    at.text_input(key="dk_MSFT_n0_head").set_value("Microsoft wins a large cloud contract")
    at.date_input(key="dk_MSFT_n0_date").set_value(dt.date(2021, 6, 15))
    at.time_input(key="dk_MSFT_n0_time").set_value(dt.time(17, 30))
    at.selectbox(key="dk_MSFT_n0_tz").set_value("ET (New York)")
    at.button(key="dk_run").click().run()
    assert not at.exception, at.exception
    rec = at.session_state["dk_result"]["out"]["MSFT"]
    assert "res" in rec, rec.get("error")
    ch = rec["chrono"]
    assert len(ch) == 1 and ch.status.iloc[0] == "model input (news branch)"      # 17:30 ET, before 09:00 next day
    assert ch["published (UTC)"].iloc[0] == __import__("pandas").Timestamp("2021-06-15 21:30", tz="UTC")

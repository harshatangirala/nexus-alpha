"""The nightly done marker is keyed by the as-of session (A-64), so an extra trigger can never skip a session."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _preflight():
    spec = importlib.util.spec_from_file_location("preflight_under_test", REPO / "scripts" / "shadow" / "preflight.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_marker_is_keyed_by_the_session_not_the_date(tmp_path, monkeypatch):
    pf = _preflight()
    monkeypatch.setattr(pf, "LOGS", tmp_path)
    monkeypatch.setattr(pf, "expected_session", lambda: "2026-11-11")
    assert pf.marker().name == "shadow_done_session_2026-11-11.txt"
    # an old date-keyed marker written on 2026-11-12 must never read as session 2026-11-12's marker
    (tmp_path / "shadow_ok_2026-11-12.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(pf, "expected_session", lambda: "2026-11-12")
    assert not pf.marker().exists()


def test_a_run_after_midnight_cannot_block_the_next_session(tmp_path, monkeypatch):
    # 01:00 IST on 2026-11-13: the US session of 2026-11-12 has not closed, so the expected session is still 2026-11-11.
    # Marking it complete must leave the 07:00 run (expected session 2026-11-12) free to run.
    pf = _preflight()
    monkeypatch.setattr(pf, "LOGS", tmp_path)
    pf.marker("2026-11-11").write_text("done", encoding="utf-8")
    assert pf.marker("2026-11-11").exists()
    assert not pf.marker("2026-11-12").exists()


def test_reachable_but_stale_yahoo_is_not_a_go():
    # A-80: at 07:00 IST on 2026-11-13 SPY had 2026-11-12 but stock downloads ended on 2026-11-11
    import pandas as pd
    pf = _preflight()
    ok, why = pf.fresh("2026-11-12", last_bar=lambda t: pd.Timestamp("2026-11-11"))
    assert not ok and "0/8" in why and "AAPL" in why
    assert pf.fresh("2026-11-12", last_bar=lambda t: pd.Timestamp("2026-11-12"))[0]


def test_one_failed_or_lagging_stock_does_not_block():
    import pandas as pd
    pf = _preflight()

    def mixed(t):
        if t == "XOM":
            raise ValueError("no data")
        return pd.Timestamp("2026-11-11") if t == "CAT" else pd.Timestamp("2026-11-12")
    ok, why = pf.fresh("2026-11-12", last_bar=mixed)
    assert ok and why.startswith("6/8")
    # three behind is too many
    def three(t):
        return pd.Timestamp("2026-11-11") if t in ("XOM", "CAT", "HD") else pd.Timestamp("2026-11-12")
    assert not pf.fresh("2026-11-12", last_bar=three)[0]

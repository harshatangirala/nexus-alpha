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

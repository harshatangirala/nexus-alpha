"""Process tools adapted from MIT trading skills (D-034)."""
import pandas as pd

from nexus_alpha import trade_tools as TT


def test_breaker_empty_allows():
    assert TT.circuit_breaker(pd.DataFrame(), 100_000)["recommendation"] == "TRADING_ALLOWED"


def test_breaker_daily_loss_halts():
    now = pd.Timestamp("2026-11-04 20:00", tz="UTC")                 # Wednesday 15:00 New York
    j = pd.DataFrame({"exit_utc": ["2026-11-04 15:00", "2026-11-04 17:00"], "pnl": [-1200.0, -900.0]})
    r = TT.circuit_breaker(j, 100_000, now)
    assert r["recommendation"] == "HALTED" and "daily loss" in r["reasons"][0]


def test_breaker_losing_streak_cools_down():
    now = pd.Timestamp("2026-11-04 20:00", tz="UTC")
    j = pd.DataFrame({"exit_utc": ["2026-11-03 15:00", "2026-11-04 15:00"], "pnl": [-100.0, -100.0]})
    assert TT.circuit_breaker(j, 100_000, now)["recommendation"] == "COOLDOWN"
    assert TT.circuit_breaker(j, 100_000, now + pd.Timedelta(days=2))["recommendation"] == "TRADING_ALLOWED"


def test_gate_rules():
    ok = TT.Checklist(True, True, True, 500, 500)
    assert TT.discipline_gate(ok, {"recommendation": "TRADING_ALLOWED"})["decision"] == "GO"
    assert TT.discipline_gate(TT.Checklist(True, False, True, 500, 500), {"recommendation": "TRADING_ALLOWED"})["decision"] == "NO_GO"
    assert TT.discipline_gate(TT.Checklist(True, True, True, 500, 600), {"recommendation": "TRADING_ALLOWED"})["decision"] == "NO_GO"
    assert TT.discipline_gate(TT.Checklist(True, True, True, None, 500), {"recommendation": "TRADING_ALLOWED"})["decision"] == "REVIEW_REQUIRED"
    assert TT.discipline_gate(ok, {"recommendation": "COOLDOWN", "reasons": ["x"]})["decision"] == "NO_GO"
    assert TT.discipline_gate(ok, {"recommendation": "TRADING_ALLOWED"}, recent_loss_hours=3)["decision"] == "NO_GO"


def test_position_size():
    r = TT.position_size(500, 100.0, 95.0)
    assert r["shares"] == 100 and r["risk_dollars"] == 500
    assert TT.position_size(500, 100.0, 95.0, max_position_dollars=5_000)["shares"] == 50

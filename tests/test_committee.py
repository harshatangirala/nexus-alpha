"""The LLM committee (E27): rating map, one vote per stock per window, scoring summary, and the quota guard."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "shadow"))

import committee_score as CS  # noqa: E402

from nexus_alpha.llm import committee as C  # noqa: E402


def test_rating_map_follows_the_preregistration():
    assert C.DIRECTION == {"Buy": 2, "Overweight": 2, "Hold": 1, "Underweight": 0, "Sell": 0}
    assert "REVIEW" not in C.DIRECTION                       # a failure, never a Hold


def test_one_vote_per_stock_per_five_sessions():
    runs = pd.DataFrame({"ticker": ["A", "A", "A", "B"],
                         "asof_session": ["2026-11-09", "2026-11-11", "2026-11-17", "2026-11-11"]})
    kept = CS.one_vote_per_window(runs)
    assert sorted(zip(kept.ticker, kept.asof_session, strict=True)) == [
        ("A", "2026-11-09"), ("A", "2026-11-17"), ("B", "2026-11-11")]


def test_summary_counts_hits_only_on_moves_and_never_claims_early():
    rows = pd.DataFrame({"asof_session": ["d1"] * 6, "ticker": list("abcdef"),
                         "direction_class": [2, 2, 0, 1, 0, 2], "real_y": [2, 0, 0, 2, 1, 1], "v11_lean": [0.1] * 6})
    s = CS.summarize(rows)
    assert s["n_directional_calls_on_moves"] == 3 and s["hit_rate_up_vs_down"] == pytest.approx(2 / 3)
    assert s["hold_share"] == pytest.approx(1 / 6) and s["sideways_share"] == pytest.approx(2 / 6)
    assert s["claim"].startswith("none yet")


def test_transient_failure_falls_over_to_the_next_model(monkeypatch, tmp_path):
    from nexus_alpha import paths, shadow as SH
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")       # never the real public log
    monkeypatch.setattr(SH, "SHADOW_ROOT", tmp_path / "shadow")
    monkeypatch.setattr(C, "available", lambda: (True, ""))
    monkeypatch.setattr(C, "model", lambda: "m1")
    monkeypatch.setattr(C.GM, "candidate_models", lambda: ["m1", "m2", "m3"])
    monkeypatch.setattr(C, "quota_left", lambda m: 20)
    monkeypatch.setattr(C.envs, "env_python", lambda name: "python")
    tried = []

    def fake(py, work, run_id, ticker, asof, analysts, m, extra, timeout_s):
        tried.append(m)
        if m == "m1":
            return {"error": "GoogleAPIError: 503 UNAVAILABLE (high demand)", "llm_calls": 1}
        out = work / "runs" / f"{run_id}_{m}.json"
        out.write_text("{}", encoding="utf-8")
        return {"rating": "Overweight", "llm_calls": 12, "seconds": 61.0, "reports": {"final_trade_decision": "x"},
                "_out": out}

    monkeypatch.setattr(C, "_attempt", fake)
    r = C.run("AAPL", ["market"])
    assert tried == ["m1", "m2"] and r["quick_model"] == "m2" and r["direction_class"] == 2 and r["chain_ok"]
    assert len(list((tmp_path / "shadow" / "committee" / "runs").glob("*.json"))) == 1


def test_a_failure_mid_run_does_not_spend_another_model(monkeypatch, tmp_path):
    from nexus_alpha import paths, shadow as SH
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(SH, "SHADOW_ROOT", tmp_path / "shadow")
    monkeypatch.setattr(C, "available", lambda: (True, ""))
    monkeypatch.setattr(C, "model", lambda: "m1")
    monkeypatch.setattr(C.GM, "candidate_models", lambda: ["m1", "m2"])
    monkeypatch.setattr(C, "quota_left", lambda m: 20)
    monkeypatch.setattr(C.envs, "env_python", lambda name: "python")
    tried = []
    monkeypatch.setattr(C, "_attempt", lambda *a: tried.append(a[6]) or {"error": "503 UNAVAILABLE", "llm_calls": 9})
    with pytest.raises(RuntimeError, match="1 model"):
        C.run("AAPL", ["market"])
    assert tried == ["m1"]


def test_quota_guard_refuses_a_run_that_cannot_fit(monkeypatch):
    monkeypatch.setattr(C, "available", lambda: (True, ""))
    monkeypatch.setattr(C, "quota_left", lambda m: 5)
    with pytest.raises(RuntimeError, match="Gemini calls needed"):
        C.run("AAPL", ["market"], quick_model="test-model")

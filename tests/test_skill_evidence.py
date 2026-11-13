"""Gate 3 with replication (audit 9, A-101, D-057): the 2017-18 direction-skill flag must also have replicated on
2020-2022 before the dashboard's risk engine trades a direction."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha.trading import skill_evidence as SE  # noqa: E402
from nexus_alpha.trading.plan import RiskConfig, plan_trade  # noqa: E402

SETUP = {"family": "TREND_PULLBACK_LONG", "direction": "LONG", "rule_ids": ["E023"], "invalidation": 95.0,
         "first_obstacle": 120.0}


def test_table_matches_the_committed_evidence():
    rep = json.loads((REPO / SE.SOURCE).read_text(encoding="utf-8"))
    for model, table in SE.REPLICATION.items():
        assert set(table) == set(rep[model]["horizons"])
        for h, (auc, lo, hi) in table.items():
            r = rep[model]["horizons"][h]
            assert (auc, lo, hi) == (r["test_up_vs_down_auroc"], r["ci95"][0], r["ci95"][1])
            assert SE.replicated(model, h) == r["replicated"]


def test_bundle_ids_are_the_frozen_records():
    for bid, model in SE.BUNDLES.items():
        rec = json.loads((REPO / "registry" / "frozen_models" / f"{model}.json").read_text(encoding="utf-8"))
        assert rec["bundle_id"] == bid


# the frozen 2017-18 flags of v1.1 (bundle_meta direction_skill; notebook section 15): validated at 5D-1M only
V11_FLAGS = {"1D": False, "2D": False, "5D": True, "10D": True, "20D": True, "1M": True}


@pytest.mark.parametrize("h", list(V11_FLAGS))
def test_no_v11_horizon_trades_direction_today(h):
    # 5D-1M were validated but did not replicate; 2D replicated (lower bound 0.5001) but was never validated
    ok, why = SE.direction_gate("v1.1", h, V11_FLAGS[h])
    assert not ok and why


@pytest.mark.parametrize("h", list(SE.REPLICATION["v2"]))
def test_no_v2_horizon_replicated(h):
    for validated in (True, False):                     # v2: no horizon replicated, so the flag cannot open the gate
        ok, why = SE.direction_gate("v2", h, validated)
        assert not ok and why


def test_reasons_and_unknown_bundles():
    ok, why = SE.direction_gate("v1.1", "5D", True)
    assert "did not replicate on 2020-22" in why and "0.508" in why
    ok, why = SE.direction_gate(None, "5D", True)                      # a research bundle: no out-of-sample evidence
    assert not ok and "never checked out of sample" in why
    assert SE.model_of_bundle("20261106T020411_e08_final_model_5c4327") == "v1.1" and SE.model_of_bundle("x") is None
    assert SE.replicated(None, "5D") is None


def test_open_gate_when_both_hold(monkeypatch):
    monkeypatch.setitem(SE.REPLICATION, "demo", {"5D": (0.56, 0.52, 0.60)})
    assert SE.direction_gate("demo", "5D", True) == (True, None)


def test_desk_labels_never_overstate_the_gate():
    # audit 10 (Antigravity pass 2): a replicated-but-not-validated horizon (v1.1 2D) must not read as validated, and
    # a research bundle (replicated None) must read "not checked", not "not replicated"
    from types import SimpleNamespace as NS
    sys.path.insert(0, str(REPO / "app"))
    import desk_multi_tab as D
    from nexus_alpha.trading.explain_plan import _skill_words
    cases = {(False, True): "not validated", (True, False): "not replicated", (True, None): "not checked out of sample",
             (True, True): "validated and replicated"}
    for (val, rep), words in cases.items():
        f = NS(direction_skill_validated=val, direction_skill_replicated=rep)
        assert words in D._skill(f) and words.split()[0] in _skill_words(f).lower()


def test_planner_reports_the_replication_reason_and_keeps_its_default():
    probs = {"up": 0.45, "down": 0.25, "side": 0.30}
    common = dict(horizon="5D", sessions=5, probs=probs, calibrated=True, levels_available=True,
                  weekly={"regime": "BULL_TREND", "permission": "LONG_PREFERRED"}, setups=[SETUP],
                  trigger_by_side={"LONG": {"trigger": "BREAKOUT", "rule_ids": ["E121"]}}, atr=2.0, close=100.0,
                  risk=RiskConfig())
    ok, why = SE.direction_gate("v1.1", "5D", True)
    p = plan_trade(skill_validated=ok, skill_reason=why, **common)
    assert p.decision == "NO_TRADE" and p.reasons[0] == why
    p0 = plan_trade(skill_validated=False, **common)                    # research backtests: wording unchanged
    assert p0.reasons[0].startswith("no validated directional skill at this horizon (validation up-vs-down AUROC")
    assert plan_trade(skill_validated=True, **common).decision == "LONG"

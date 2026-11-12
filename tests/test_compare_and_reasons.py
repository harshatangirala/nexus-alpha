"""Compare tab data layer and the reason card / conditional levels (display only; never a decision)."""
import math

import numpy as np
import pandas as pd

from nexus_alpha import compare as C
from nexus_alpha.trading import explain_plan as EP
from nexus_alpha.trading.plan import RiskConfig, plan_trade


def _bars(n=300, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(drift + 0.015 * rng.standard_normal(n)))
    d = pd.DataFrame({"open": c * (1 + 0.002 * rng.standard_normal(n)), "close": c, "volume": 1e6})
    d["high"] = d[["open", "close"]].max(axis=1) * 1.01
    d["low"] = d[["open", "close"]].min(axis=1) * 0.99
    d.index = pd.bdate_range("2025-01-02", periods=n)
    return d


def test_risk_stats_and_normalisation():
    b = {"AAA": _bars(seed=1), "BBB": _bars(seed=2, drift=-0.001)}
    s = C.risk_stats(b)
    assert set(s.index) == {"AAA", "BBB"}
    assert (s["max drawdown (window)"] <= 0).all() and (s["drawdown now"] <= 0).all()
    assert 0.1 < s.loc["AAA", "volatility 1Y (a year)"] < 0.4          # 1.5% daily ~ 24% a year
    p = C.normalized(C.close_panel(b, 63))
    assert np.allclose(p.iloc[0], 100)
    c = C.correlation(b)
    assert np.allclose(np.diag(c), 1)


def test_lean_matrix_and_verdict():
    fc = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB", "BBB"], "horizon": ["5D", "20D", "5D", "20D"],
                       "p_up": [0.4, 0.3, 0.2, 0.25], "p_down": [0.3, 0.3, 0.4, 0.2], "p_side": [0.3, 0.4, 0.4, 0.55]})
    m = C.lean_matrix(fc, ["BBB", "AAA"], ["5D", "20D"])
    assert list(m.index) == ["BBB", "AAA"] and list(m.columns) == ["5D", "20D"]
    assert math.isclose(m.loc["AAA", "5D"], 0.1) and math.isclose(m.loc["BBB", "5D"], -0.2)
    s = C.risk_stats({"AAA": _bars(seed=1), "BBB": _bars(seed=2)})
    v = C.verdict(s, None, pd.Series({"AAA": 80.0, "BBB": 20.0}, name="v2 20D"))
    assert "Most volatile" in v and "did not hold up" in v and "AAA" in v


def test_real_forward_logs_load_if_present():
    sess = C.latest_session()
    if sess is None:
        return
    fc = C.forecasts(sess, ["AAPL", "MSFT"])
    assert "v1.1" in fc and set(fc["v1.1"].ticker) <= {"AAPL", "MSFT"}
    assert not fc["v1.1"].duplicated(["ticker", "horizon"]).any()
    r = C.rank_percentile(sess, "v2", "20D")
    assert r.empty or (r.between(0, 100).all() and len(r) > 400)


SETUP = {"family": "TREND_PULLBACK_LONG", "direction": "LONG", "rule_ids": ["E023", "E074", "E083"],
         "invalidation": 95.0, "first_obstacle": 110.0}
EVID = {"indicators": {"asof_session": "2026-11-11", "close": 100.0, "atr14": 2.0, "rsi14": 55.0, "macd_hist": 0.1,
                       "sma20": 99.0, "sma50": 97.0, "bb_up": 104.0, "bb_lo": 96.0, "vol_ratio": 1.1},
        "rules": {"weekly_regime": "BEAR_TREND", "weekly_permission": "SHORT_PREFERRED", "daily_setups": [SETUP],
                  "hourly_triggers": {"LONG": {"trigger": "NO_TRIGGER"}, "SHORT": {"trigger": "NO_TRIGGER"}}}}


def test_conditional_levels_reuse_the_engine_geometry():
    risk = RiskConfig()
    cond = EP.conditional_plans(EVID, "5D", 5, risk)
    assert len(cond) == 1 and cond[0]["geometry_ok"] and cond[0]["label"].startswith("CONDITIONAL")
    p = cond[0]["plan"]
    assert math.isclose(p.stop, 95.0 - 0.25 * 2.0) and p.target_1 == 110.0 and p.stop < p.entry_reference < p.target_1
    # the same numbers as the engine's own rule-only path (P0)
    p0 = plan_trade("5D", 5, None, False, False, True, {"permission": "EDGES_ONLY"}, [SETUP],
                    {"LONG": {"trigger": "X", "rule_ids": []}}, 2.0, 100.0, risk, side_override="LONG")
    assert (p.stop, p.target_1, p.quantity, p.net_rr) == (p0.stop, p0.target_1, p0.quantity, p0.net_rr)
    # the gates the real decision fails are listed
    assert any("weekly regime BEAR_TREND" in g for g in cond[0]["missing_gates"])
    assert any("no hourly long trigger" in g for g in cond[0]["missing_gates"])


def test_reason_card_never_hides_the_real_decision():
    real = plan_trade("5D", 5, {"up": 0.4, "down": 0.3, "side": 0.3}, True, False, True,
                      EVID["rules"] | {"permission": "SHORT_PREFERRED"}, [SETUP], EVID["rules"]["hourly_triggers"],
                      2.0, 100.0, RiskConfig())
    assert real.decision == "NO_TRADE"
    card = EP.reason_card(real, EVID, None, EP.conditional_plans(EVID, "5D", 5, RiskConfig()))
    assert card["Decision"][0].startswith("NO_TRADE")
    assert card["Entry"][0].startswith("CONDITIONAL, not approved")
    assert "X001" in card["Stop loss"][0] and "X021" in card["Targets"][0]
    assert "Sharpe -0.87 at 5D" in card["Evidence on rule-only plans"][0]
    assert any("RSI 14 = 55.0" in x for x in card["Indicators"])


def test_p0_evidence_quotes_the_committed_backtest():
    import json
    from pathlib import Path
    b = json.loads((Path(__file__).resolve().parents[1] / "reports" / "backtest" / "e04_backtest_v1.1.json")
                   .read_text(encoding="utf-8"))
    s5 = b["test_final"]["rule_only|5D|theta=0.1|base"]["sharpe"]
    s20 = b["test_final"]["rule_only|20D|theta=0.1|base"]["sharpe"]
    bench = b["benchmark_ew_survivors"]["test"]["sharpe"]
    assert f"Sharpe {s5:.2f} at 5D and {s20:.2f} at 20D, against {bench:.2f}" in EP.P0_EVIDENCE


def test_no_setup_means_no_levels():
    ev = {**EVID, "rules": {**EVID["rules"], "daily_setups": []}}
    assert EP.conditional_plans(ev, "5D", 5, RiskConfig()) == []
    card = EP.reason_card(plan_trade("5D", 5, None, False, False, True, {}, [], {}, 2.0, 100.0, RiskConfig()), ev)
    assert "Entry" not in card and any("no daily setup" in x for x in card["Chart patterns and rules"])

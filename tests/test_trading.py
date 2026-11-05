import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nexus_alpha.inference.schema import HorizonForecast, TradePlan
from nexus_alpha.trading.backtest import Order, simulate
from nexus_alpha.trading.costs import CostModel
from nexus_alpha.trading.plan import RiskConfig, plan_trade

WEEKLY_BULL = {"regime": "BULL_TREND", "permission": "LONG_PREFERRED"}
SETUP_LONG = {"family": "TREND_PULLBACK_LONG", "direction": "LONG", "rule_ids": ["E023"], "invalidation": 95.0,
              "first_obstacle": 112.0}
TRIG = {"LONG": {"trigger": "STRUCTURE_BREAK", "rule_ids": ["E066"]}, "SHORT": {"trigger": "NO_TRIGGER"}}
PROBS = {"up": 0.55, "down": 0.25, "side": 0.20}


def _plan(**kw):
    args = dict(horizon="5D", sessions=5, probs=PROBS, calibrated=True, skill_validated=True, levels_available=True,
                weekly=WEEKLY_BULL, setups=[SETUP_LONG], trigger_by_side=TRIG, atr=2.0, close=100.0, risk=RiskConfig())
    args.update(kw)
    return plan_trade(**args)


def test_valid_long_plan_geometry_and_sizing():
    p = _plan()
    assert p.decision == "LONG"
    assert p.stop == pytest.approx(95.0 - 0.25 * 2.0) and p.stop < p.entry_reference < p.target_1
    cost_ps = 100.0 * 10 / 1e4
    assert p.quantity == int(np.floor(100_000 * 0.005 / (p.risk_per_share + cost_ps)))
    assert p.net_rr == pytest.approx((12.0 - cost_ps) / (5.5 + cost_ps))
    assert p.planned_loss_cash <= 100_000 * 0.005 + 1e-6


def test_high_probability_alone_is_not_a_trade():
    assert _plan(skill_validated=False).decision == "NO_TRADE"
    assert _plan(setups=[]).decision == "NO_TRADE"
    assert _plan(trigger_by_side={"LONG": {"trigger": "NO_TRIGGER"}}).decision == "NO_TRADE"
    assert _plan(weekly={"regime": "BEAR_TREND", "permission": "SHORT_PREFERRED"}).decision == "NO_TRADE"
    assert _plan(probs={"up": 0.36, "down": 0.32, "side": 0.32}).decision == "NO_TRADE"     # edge below minimum


def test_screenshot_only_gives_no_numeric_plan():
    p = _plan(levels_available=False)
    assert p.decision == "NO_SIGNAL" and p.stop is None and p.quantity is None


def test_poor_reward_to_risk_and_infeasible_stop_are_rejected():
    near = dict(SETUP_LONG, first_obstacle=103.0)
    assert _plan(setups=[near]).decision == "NO_TRADE"
    far_stop = dict(SETUP_LONG, invalidation=1.0, first_obstacle=400.0)
    tiny = RiskConfig(account_capital=100.0)
    assert _plan(setups=[far_stop], risk=tiny).decision == "NO_TRADE"


def test_schema_rejects_bad_plans_and_probabilities():
    with pytest.raises(ValidationError):
        TradePlan(decision="LONG", entry_reference=100, stop=101, target_1=110, quantity=5)      # stop above entry
    with pytest.raises(ValidationError):
        TradePlan(decision="SHORT", entry_reference=100, stop=99, target_1=90, quantity=5)       # stop below entry
    with pytest.raises(ValidationError):
        HorizonForecast(up_probability=0.5, down_probability=0.4, sideways_probability=0.2)
    with pytest.raises(ValidationError):
        HorizonForecast(up_probability=0.5, down_probability=None, sideways_probability=0.5)
    HorizonForecast(up_probability=0.5, down_probability=0.3, sideways_probability=0.2)


def _path(rows):
    s = pd.bdate_range("2021-01-04", periods=len(rows))
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"date": s, "open": o, "high": h, "low": l, "close": c, "volume": 1e6})


def test_gap_through_stop_fills_at_open_not_stop():
    d = _path([(100, 101, 99, 100), (100, 101, 99.5, 100.5), (101, 102, 100, 101), (90, 91, 88, 89), (89, 90, 88, 89)])
    r = simulate(Order("s", "T", 0, +1, stop=97.0, target=110.0, horizon=4), d, CostModel())
    assert r["exit_reason"] == "stop_gap" and r["exit"] == 90 and r["net_return"] < (97 / 100 - 1)


def test_same_bar_stop_and_target_is_stop_first():
    d = _path([(100, 101, 99, 100), (100, 111, 96, 105), (105, 106, 104, 105)])
    r = simulate(Order("s", "T", 0, +1, stop=97.0, target=110.0, horizon=2), d, CostModel())
    assert r["exit_reason"] == "stop" and r["ambiguous_bar"] and r["exit"] == 97.0


def test_chase_gap_cancels_and_costs_apply():
    d = _path([(100, 101, 99, 100), (104, 105, 103, 104), (104, 105, 103, 104)])
    assert simulate(Order("s", "T", 0, +1, 97, 110, 2, chase=102.0), d, CostModel())["fill_status"] == "CANCELLED"
    d2 = _path([(100, 101, 99, 100), (100, 101, 99.5, 100), (100, 101, 99.5, 100)])
    r = simulate(Order("s", "T", 0, +1, 97, 110, 2), d2, CostModel())
    assert r["exit_reason"] == "time" and r["gross_return"] == 0 and r["net_return"] == pytest.approx(-0.001)
    rs = simulate(Order("s", "T", 0, -1, 103, 90, 2), d2, CostModel())
    assert rs["net_return"] == pytest.approx(-0.001 - 0.005 * 2 / 252)
    assert sum(x for _, x in r["daily"]) == pytest.approx(r["net_return"], abs=1e-9)


def test_daily_pnl_sums_to_trade_return_on_a_moving_path():
    d = _path([(100, 101, 99, 100), (100, 104, 99.5, 103), (103, 106, 102, 105), (105, 107, 101, 102), (102, 103, 101, 102)])
    for side, stop, tgt in ((+1, 90.0, 130.0), (-1, 110.0, 80.0)):
        r = simulate(Order("s", "T", 0, side, stop, tgt, 4), d, CostModel())
        assert r["exit_reason"] == "time"
        assert sum(x for _, x in r["daily"]) == pytest.approx(r["net_return"], abs=1e-12)


def test_portfolio_kill_switch_and_gross_cap():
    from nexus_alpha.trading.backtest import portfolio
    days = pd.bdate_range("2021-01-04", periods=6)
    def trade(sid, d0, path, risk=0.05):
        return {"sample_id": sid, "fill_status": "FILLED", "entry_date": d0, "risk_fraction": risk,
                "daily": [(days[i], r) for i, r in path]}
    t = pd.DataFrame([trade("a", days[0], [(0, -0.30), (1, 0.0)], risk=0.05),      # weight 0.1 -> -3% day 0
                      trade("b", days[1], [(1, 0.01)]),                               # blocked by kill-switch
                      trade("c", days[2], [(2, 0.01)])])                              # allowed
    ret, kept = portfolio(t, days, risk_pct=0.005, max_position_pct=0.10, max_daily_loss=0.02)
    assert ret[days[0]] == pytest.approx(-0.03) and set(kept.sample_id) == {"a", "c"}
    assert int(kept.blocked_by_daily_loss.iloc[0]) == 1

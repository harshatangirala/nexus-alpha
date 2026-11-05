"""Deterministic trade planner (docs/RISK_ENGINE_SPECIFICATION.md). Forecasting and trading are separate: a high UP
probability is never by itself a LONG.

Gate order (first failure decides; every reason is reported):
  1  data sufficiency: exact levels need structured OHLCV for the as-of session (E001/E003/E006/E007); screenshots alone
     give NO_SIGNAL for numeric plans
  2  calibrated forecast available for the horizon
  3  directional skill gate: the horizon's validation up-vs-down AUROC lower CI bound must exceed 0.5 (model card);
     otherwise NO_TRADE "no validated directional skill"
  4  edge gate: |P(UP) - P(DOWN)| >= min_edge (chosen on validation, frozen in the risk config)
  5  weekly permission compatible with the side (E011-E018, decision-engine permission matrix)
  6  a daily setup on the same side supplies the structural invalidation and the first obstacle (E023/E027/E031/...)
  7  hourly trigger on the as-of session in the setup direction (E121/E122), else WAIT -> NO_TRADE
  8  stop = invalidation -/+ 0.25 ATR buffer (X001, X011); entry reference = last close; stop must be on the losing side
  9  target 1 = first opposing obstacle (X021); net reward-to-risk after round-trip costs >= min_net_rr (E039, E149)
  10 size = floor(account * risk_pct / (risk per share + cost per share)), capped by max position exposure;
     quantity < 1 -> NO_TRADE (stop infeasible, X010)
  11 time exit after the horizon's sessions (X071); max chase = 1 ATR beyond the reference (E111)
The planner never widens a stop to fit a size, never invents a level, and never reads prices from chart pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..inference.schema import TradePlan
from .costs import CostModel

STOP_ATR_BUFFER = 0.25
CHASE_ATR = 1.0


@dataclass
class RiskConfig:
    account_capital: float = 100_000.0
    risk_pct_per_trade: float = 0.005
    max_position_pct: float = 0.10
    max_portfolio_gross_pct: float = 1.0
    max_daily_loss_pct: float = 0.02
    allow_short: bool = True
    min_edge: float = 0.10
    min_net_rr: float = 1.5
    cost_scenario: str = "base"


def plan_trade(horizon: str, sessions: int, probs: dict | None, calibrated: bool, skill_validated: bool,
               levels_available: bool, weekly: dict, setups: list[dict], trigger_by_side: dict, atr: float | None,
               close: float | None, risk: RiskConfig, data_gate_reasons: list[str] | None = None) -> TradePlan:
    cost = CostModel.from_scenario(risk.cost_scenario)
    reasons = list(data_gate_reasons or [])
    if reasons:
        return TradePlan(decision="NO_SIGNAL", reasons=reasons)
    if not levels_available or close is None or atr is None or not math.isfinite(atr) or atr <= 0:
        return TradePlan(decision="NO_SIGNAL", reasons=["exact price levels unavailable: structured OHLCV for the as-of "
                                                         "session is required (screenshot-only input gives no numeric plan)"])
    if probs is None:
        return TradePlan(decision="NO_SIGNAL", reasons=["no forecast for this horizon"])
    if not calibrated:
        reasons.append("forecast is not calibrated")
    if not skill_validated:
        reasons.append("no validated directional skill at this horizon (validation up-vs-down AUROC interval includes 0.5)")
    edge = probs["up"] - probs["down"]
    side = "LONG" if edge > 0 else "SHORT"
    if abs(edge) < risk.min_edge:
        reasons.append(f"edge |P(UP)-P(DOWN)| = {abs(edge):.3f} below the frozen minimum {risk.min_edge:.2f}")
    if side == "SHORT" and not risk.allow_short:
        reasons.append("short selling disabled in the risk configuration")
    perm = weekly.get("permission", "NONE")
    allowed = {"LONG": {"LONG_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"},
               "SHORT": {"SHORT_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"}}[side]
    if perm not in allowed:
        reasons.append(f"weekly regime {weekly.get('regime', 'UNKNOWN')} does not permit {side} ({perm})")
    setup = next((s for s in setups if s["direction"] == side), None)
    if setup is None:
        reasons.append(f"no daily {side.lower()} setup with a structural invalidation level")
    trig = trigger_by_side.get(side, {"trigger": "NO_TRIGGER"})
    if trig.get("trigger", "NO_TRIGGER") == "NO_TRIGGER":
        reasons.append(f"no hourly {side.lower()} trigger on the as-of session (WAIT)")
    if reasons:
        return TradePlan(decision="NO_TRADE", reasons=reasons, setup_family=setup["family"] if setup else None)
    entry = float(close)
    if side == "LONG":
        stop = setup["invalidation"] - STOP_ATR_BUFFER * atr
        target = setup.get("first_obstacle")
        chase = entry + CHASE_ATR * atr
        ok_geom = target is not None and stop < entry < target
    else:
        stop = setup["invalidation"] + STOP_ATR_BUFFER * atr
        target = setup.get("first_obstacle")
        chase = entry - CHASE_ATR * atr
        ok_geom = target is not None and target < entry < stop
    if not ok_geom:
        return TradePlan(decision="NO_TRADE", reasons=["stop/target geometry invalid for the setup (no opposing obstacle "
                                                       "on the right side, or stop on the wrong side)"],
                         setup_family=setup["family"])
    risk_ps = abs(entry - stop)
    reward_ps = abs(target - entry)
    cost_ps = cost.per_share_round_trip(entry) + (cost.borrow(entry, sessions) if side == "SHORT" else 0.0)
    gross_rr = reward_ps / risk_ps
    net_rr = (reward_ps - cost_ps) / (risk_ps + cost_ps)
    if net_rr < risk.min_net_rr:
        return TradePlan(decision="NO_TRADE", reasons=[f"net reward-to-risk {net_rr:.2f} below {risk.min_net_rr:.2f} after costs"],
                         setup_family=setup["family"], gross_rr=gross_rr, net_rr=net_rr)
    qty_risk = math.floor(risk.account_capital * risk.risk_pct_per_trade / (risk_ps + cost_ps))
    qty_cap = math.floor(risk.account_capital * risk.max_position_pct / entry)
    qty = min(qty_risk, qty_cap)
    if qty < 1:
        return TradePlan(decision="NO_TRADE", reasons=["stop distance too wide for the risk budget (X010): quantity < 1"],
                         setup_family=setup["family"])
    t2 = entry + 2 * (target - entry)
    return TradePlan(decision=side, reasons=["all gates passed"], setup_family=setup["family"], entry_reference=entry,
                     entry_rule="market at next session open if within the max chase price, else cancel",
                     max_chase_price=chase, stop=stop, target_1=target, target_2=t2, risk_per_share=risk_ps,
                     reward_per_share=reward_ps, gross_rr=gross_rr, net_rr=net_rr, quantity=int(qty),
                     position_value=qty * entry, planned_loss_cash=qty * (risk_ps + cost_ps), holding_sessions=sessions,
                     invalidation=f"daily close beyond {setup['invalidation']:.2f} (setup {setup['family']})",
                     cost_assumptions=cost.to_dict(), rule_ids=sorted(set(setup["rule_ids"] + trig.get("rule_ids", []) +
                                                                       ["X001", "X011", "X021", "X071", "E111", "E149"])),
                     levels_source="structured OHLCV (sealed TriFrame bars or live Yahoo bars), never chart pixels")

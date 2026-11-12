"""Why these levels? Reason cards and conditional levels for the trade planner (display layer; never a decision).

The risk engine (`trading.plan.plan_trade`) decides LONG, SHORT, NO_TRADE or NO_SIGNAL. When it says NO_TRADE, the
dashboard used to show no levels at all. The owner asked to see the entry, stop and targets together with the reasons.
This module adds two things without touching the engine:

1. **Conditional levels.** For each daily rule setup, the engine's own geometry is computed as if the setup were
   taken (`plan_trade(..., side_override=...)`, the rule-only P0 path used in the backtests). The record is labelled
   CONDITIONAL and lists the gates the real decision failed. P0 lost money in the audited 2020-22 backtest (Sharpe
   -0.87 at 5D and -0.34 at 20D), which every card says.
2. **A reason card.** Plain-language reasons for the entry, stop, targets, exit and size, each tied to its rule id,
   plus the evidence the analysis saw: weekly regime, daily setup, hourly trigger, indicators, the model's
   probabilities, company texts (news, filings), fundamentals and macro context, and book or rulebook citations.

Nothing here changes a probability, a gate or a logged forecast.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .. import paths
from ..inference.schema import TradePlan
from .plan import CHASE_ATR, STOP_ATR_BUFFER, RiskConfig, plan_trade

P0_EVIDENCE = ("Rule-only plans (P0) lost money in the audited 2020-22 backtest (reports/backtest/e04_backtest_v1.1.json, "
               "base costs: Sharpe -0.87 at 5D and -0.34 at 20D, against 0.53 for equal-weight buy-and-hold). "
               "Conditional levels show the geometry, not an approved trade.")


@lru_cache(maxsize=1)
def rule_titles() -> dict[str, str]:
    """Rule id -> title, from the rulebook chunks of the knowledge base (the owner's own rulebooks)."""
    f = paths.ARTIFACTS / "rag" / "chunks.parquet"
    if not f.exists():
        return {}
    import pandas as pd
    c = pd.read_parquet(f, columns=["rule_id", "text"])
    out = {}
    for rid, txt in c.dropna(subset=["rule_id"]).itertuples(index=False):
        first = str(txt).strip().splitlines()[0] if str(txt).strip() else ""
        first = first.lstrip("#").strip()
        if first.startswith(rid):
            first = first[len(rid):].lstrip(" —-:").strip()
        title = first.split(". ")[0].rstrip(".")                  # the rule's name, before its definition sentence
        if len(title) > 70:
            title = title[:70].rsplit(" ", 1)[0] + "..."
        out.setdefault(rid, title)
    return out


def _rid(ids) -> str:
    t = rule_titles()
    return ", ".join(f"{r} ({t[r]})" if r in t and t[r] else r for r in ids)


def conditional_plans(evidence: dict, horizon: str, sessions: int, risk: RiskConfig) -> list[dict]:
    """The engine's geometry for each daily setup, computed as if the setup were taken (rule-only P0 path)."""
    rules = evidence.get("rules") or {}
    ind = evidence.get("indicators") or {}
    atr, close = ind.get("atr14"), ind.get("close")
    if atr is None or close is None or not math.isfinite(atr) or atr <= 0:
        return []
    out = []
    for s in rules.get("daily_setups") or []:
        side = s["direction"]
        trig = (rules.get("hourly_triggers") or {}).get(side, {"trigger": "NO_TRIGGER"})
        # permissive context: only the geometry, cost, reward-to-risk and size gates (8-10) can fail
        p = plan_trade(horizon, sessions, None, calibrated=False, skill_validated=False, levels_available=True,
                       weekly={"permission": "EDGES_ONLY"}, setups=[s],
                       trigger_by_side={side: {"trigger": "CONDITIONAL", "rule_ids": trig.get("rule_ids", [])}},
                       atr=float(atr), close=float(close), risk=risk, side_override=side)
        missing = []
        perm = rules.get("weekly_permission", "NONE")
        allowed = {"LONG": {"LONG_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"},
                   "SHORT": {"SHORT_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"}}[side]
        if perm not in allowed:
            missing.append(f"weekly regime {rules.get('weekly_regime', 'UNKNOWN')} does not permit {side} ({perm})")
        if trig.get("trigger", "NO_TRIGGER") == "NO_TRIGGER":
            missing.append(f"no hourly {side.lower()} trigger on the as-of session (WAIT)")
        out.append({"setup": s, "side": side, "plan": p, "geometry_ok": p.decision in ("LONG", "SHORT"),
                    "missing_gates": missing, "label": "CONDITIONAL (not approved)"})
    return out


def _indicator_lines(ind: dict) -> list[str]:
    out = []
    c = ind.get("close")
    rsi = ind.get("rsi14")
    if rsi is not None:
        zone = "overbought (> 70)" if rsi > 70 else "oversold (< 30)" if rsi < 30 else "neutral (30-70)"
        out.append(f"RSI 14 = {rsi:.1f}: {zone}")
    if ind.get("macd_hist") is not None:
        out.append(f"MACD histogram {ind['macd_hist']:+.3f}: momentum {'rising' if ind['macd_hist'] > 0 else 'falling'}")
    if c is not None and ind.get("sma20") and ind.get("sma50"):
        a = "above" if c > ind["sma20"] else "below"
        b = "above" if c > ind["sma50"] else "below"
        t = "up" if ind["sma20"] > ind["sma50"] else "down"
        out.append(f"close {c:.2f} is {a} SMA20 ({ind['sma20']:.2f}) and {b} SMA50 ({ind['sma50']:.2f}); "
                   f"SMA20 vs SMA50 trend {t}")
    if c is not None and ind.get("bb_up") and ind.get("bb_lo") and ind["bb_up"] > ind["bb_lo"]:
        pb = (c - ind["bb_lo"]) / (ind["bb_up"] - ind["bb_lo"])
        out.append(f"Bollinger %B {pb:.2f} ({'upper band' if pb > 0.8 else 'lower band' if pb < 0.2 else 'mid band'})")
    if ind.get("vol_ratio") is not None:
        out.append(f"volume {ind['vol_ratio']:.2f}x its 20-day average"
                   + (" (participation expansion, E101)" if ind["vol_ratio"] >= 1.5 else ""))
    if c and ind.get("atr14"):
        out.append(f"ATR 14 = {ind['atr14']:.2f} ({ind['atr14'] / c:.1%} of price): sets the stop buffer and the chase limit")
    return out


def reason_card(plan: TradePlan, evidence: dict, forecast=None, conditional: list[dict] | None = None,
                context: dict | None = None) -> dict:
    """Sections of plain-language reasons. `context` may carry 'macro' and 'fundamentals' lines from the caller."""
    ind = evidence.get("indicators") or {}
    rules = evidence.get("rules") or {}
    card: dict[str, list[str]] = {}
    card["Decision"] = [f"{plan.decision}: " + ("; ".join(plan.reasons) if plan.reasons else "")]
    shown = plan if plan.decision in ("LONG", "SHORT") else None
    cond = None
    if shown is None and conditional:
        ok = [c for c in conditional if c["geometry_ok"]]
        cond = ok[0] if ok else None
        shown = cond["plan"] if cond else None
    if shown is not None:
        s = (cond or {}).get("setup") or next((x for x in rules.get("daily_setups") or [] if x["family"] == shown.setup_family), {})
        pre = "CONDITIONAL, not approved: " if cond else ""
        card["Entry"] = [f"{pre}reference = last close {shown.entry_reference:.2f} (as of {ind.get('asof_session', '?')}); "
                         f"{shown.entry_rule}; max chase {shown.max_chase_price:.2f} = close {'+' if shown.stop < shown.entry_reference else '-'} "
                         f"{CHASE_ATR:g} ATR (E111)"]
        inv = s.get("invalidation")
        card["Stop loss"] = [f"{shown.stop:.2f} = setup invalidation {inv:.2f} {'-' if shown.stop < shown.entry_reference else '+'} "
                             f"{STOP_ATR_BUFFER:g} ATR buffer (X001, X011). The invalidation comes from the "
                             f"{s.get('family', '?')} setup: {_rid(s.get('rule_ids', []))}" if inv is not None else
                             f"{shown.stop:.2f}"]
        card["Targets"] = [f"T1 {shown.target_1:.2f}: the first opposing obstacle (X021); T2 {shown.target_2:.2f}: twice the T1 "
                           f"distance. Gross reward-to-risk {shown.gross_rr:.2f}, net of costs {shown.net_rr:.2f} "
                           "(minimum 1.5, E039/E149)"]
        card["Exit"] = [f"stop, target, or a time exit after {shown.holding_sessions} sessions (X071); "
                        f"thesis invalid on a {shown.invalidation}"]
        card["Size"] = [f"{shown.quantity} shares (value {shown.position_value:,.0f}): floor(capital x risk % / (risk per share "
                        f"{shown.risk_per_share:.2f} + costs)), capped by the maximum position; planned loss at the stop "
                        f"{shown.planned_loss_cash:,.0f}"]
        if cond:
            card["Gates the real decision failed"] = (cond["missing_gates"] + [r for r in plan.reasons
                                                                              if r not in cond["missing_gates"]])
            card["Evidence on rule-only plans"] = [P0_EVIDENCE]
    tech = [f"weekly regime {rules.get('weekly_regime', 'UNKNOWN')}, permission {rules.get('weekly_permission', 'NONE')} "
            "(E011-E018)"]
    for s in rules.get("daily_setups") or []:
        tech.append(f"daily setup {s['family']} ({s['direction']}): {_rid(s['rule_ids'])}")
    if not rules.get("daily_setups"):
        tech.append("no daily setup fired (no structural invalidation level, so no plan)")
    for side, t in (rules.get("hourly_triggers") or {}).items():
        if t.get("trigger", "NO_TRIGGER") != "NO_TRIGGER":
            tech.append(f"hourly trigger {t['trigger']} ({side.lower()}): {_rid(t.get('rule_ids', []))}")
    card["Chart patterns and rules"] = tech
    card["Indicators"] = _indicator_lines(ind)
    if forecast is not None:
        card["Model"] = [f"P(up) {forecast.up_probability:.2f}, P(sideways) {forecast.sideways_probability:.2f}, "
                         f"P(down) {forecast.down_probability:.2f}; lean {forecast.up_probability - forecast.down_probability:+.3f}; "
                         f"direction skill {'validated' if forecast.direction_skill_validated else 'NOT validated'} at this horizon"]
    micro = []
    for n in evidence.get("news") or []:
        tone = n.get("lm_net_tone")
        micro.append(f"news: {str(n.get('headline', ''))[:110]}" + (f" (net tone {tone:+.2f})" if tone is not None else "")
                     + ("" if n.get("used", True) else " [not used: after the decision time or duplicate]"))
    for f in (evidence.get("sec_filings") or [])[:4]:
        items = f" items {f['items']}" if f.get("items") not in (None, "", []) else ""
        micro.append(f"SEC {f.get('forms') or f.get('type', 'filing')}{items}, accepted {f.get('acceptance_time', '?')}"
                     + (" (text read by the model)" if f.get("used_for_text") else ""))
    micro += list((context or {}).get("fundamentals", []))
    if micro:
        card["Company (micro) evidence"] = micro
    if (context or {}).get("macro"):
        card["Macro context"] = list(context["macro"])
    cites = [c for c in evidence.get("rag_sources") or [] if c.get("doc_id")]
    if cites:
        card["Book and rulebook references (RAG; explanation only)"] = [
            f"{c['doc_id']}" + (f" p.{c['page']}" if c.get("page") else "") + (f" {c['rule_id']}" if c.get("rule_id") else "")
            + f": {c.get('why', '')}" for c in cites[:5]]
    return card

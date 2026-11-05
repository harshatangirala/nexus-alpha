"""Executable, causal detectors for a tested subset of the rulebook (all other rules are catalogued only).

Conventions (binding for every detector here):
* Inputs are completed bars only, oldest first. A value at bar t uses bars <= t.
* Swing pivots need k bars on each side, so a pivot at bar i becomes KNOWN at bar i + k (its confirmation bar). It is
  stored at the confirmation bar and never back-dated (decision-engine spec 4.2, rulebook global safeguard 1).
* Numeric candle definitions follow the usual Nison-style geometry with explicit thresholds (constants below). They are
  research definitions to be validated, not claims that a pattern predicts anything.
* Every detector output follows the rulebook output contract: rule_id, state, direction, timeframe, observed_at,
  source_bar_end, anchor, reason_code, rule_version.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..indicators.core import indicators

RULE_VERSION = "nexus-detectors-v1"
PIVOT_K_DAILY = 3
PIVOT_K_WEEKLY = 2
HAMMER_WICK_BODY = 2.0          # long wick at least 2x the real body
SMALL_WICK_BODY = 0.5           # opposite wick at most 0.5x body
STAR_BODY_FRAC = 0.3            # middle candle body at most 30% of the first candle's body
LONG_BODY_RANGE = 0.6           # "long" candle: body at least 60% of its range
CONTEXT_BARS = 5                # prior trend context: close[t-1] vs close[t-1-CONTEXT_BARS]
SUPPORT_ATR = 1.0               # "at support": low within 1 ATR of the last confirmed swing low / 20-bar low
MIN_ADV_USD = 1_000_000         # E006 liquidity gate (same threshold as TriFrame eligibility)
MIN_PRICE = 5.0
REL_VOL_EXPANSION = 1.2         # E101
CHASE_ATR = 1.0                 # E111/E128 maximum distance from trigger level in daily ATR
MIN_OBSTACLE_ATR = 0.5          # X021 'material' obstacle: at least 0.5 ATR beyond the reference


@dataclass
class RuleEvent:
    rule_id: str
    state: str                  # FIRED, NOT_FIRED, UNKNOWN, BLOCKED
    direction: str              # LONG, SHORT, NEUTRAL, N/A
    timeframe: str
    observed_at: str
    source_bar_end: str
    anchor: float | None
    reason_code: str
    rule_version: str = RULE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ causal pivots
def confirmed_pivots(high: np.ndarray, low: np.ndarray, k: int) -> pd.DataFrame:
    """Swing highs/lows known at each bar. Bar i is a swing high if high[i] is the strict maximum of
    high[i-k .. i+k] (ties go to the earlier bar); it is recorded at bar i + k. Returns, per bar, the last two
    confirmed swing highs and lows (values and pivot bar indices)."""
    n = len(high)
    sh_val, sl_val = np.full(n, np.nan), np.full(n, np.nan)
    sh_idx, sl_idx = np.full(n, -1), np.full(n, -1)
    for i in range(k, n - k):
        win_h = high[i - k: i + k + 1]
        if high[i] == win_h.max() and np.argmax(win_h) == k:
            sh_val[i + k], sh_idx[i + k] = high[i], i
        win_l = low[i - k: i + k + 1]
        if low[i] == win_l.min() and np.argmin(win_l) == k:
            sl_val[i + k], sl_idx[i + k] = low[i], i
    out = {}
    for name, v, ix in (("sh", sh_val, sh_idx), ("sl", sl_val, sl_idx)):
        last = np.full(n, np.nan); prev = np.full(n, np.nan)
        last_i = np.full(n, -1); prev_i = np.full(n, -1)
        lv, pv, li, pi = np.nan, np.nan, -1, -1
        for t in range(n):
            if not np.isnan(v[t]):
                pv, pi = lv, li
                lv, li = v[t], ix[t]
            last[t], prev[t], last_i[t], prev_i[t] = lv, pv, li, pi
        out[f"{name}_last"], out[f"{name}_prev"] = last, prev
        out[f"{name}_last_idx"], out[f"{name}_prev_idx"] = last_i, prev_i
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ candle geometry
def candle_parts(o, h, l, c):
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return body, rng, upper, lower


def candle_patterns(d: pd.DataFrame, atr: pd.Series | None = None) -> pd.DataFrame:
    """Boolean pattern columns at each completed daily bar (E041-E046, E055) with their context conditions."""
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    body, rng, upper, lower = candle_parts(o, h, l, c)
    atr = indicators(d).atr14.to_numpy() if atr is None else np.asarray(atr, float)
    piv = confirmed_pivots(h, l, PIVOT_K_DAILY)
    low20 = pd.Series(l).rolling(20, min_periods=5).min().shift(1).to_numpy()
    high20 = pd.Series(h).rolling(20, min_periods=5).max().shift(1).to_numpy()
    sup = np.fmax(np.nan_to_num(piv.sl_last.to_numpy(), nan=-np.inf), np.nan_to_num(low20, nan=-np.inf))
    res = np.fmin(np.nan_to_num(piv.sh_last.to_numpy(), nan=np.inf), np.nan_to_num(high20, nan=np.inf))
    at_support = np.abs(l - sup) <= SUPPORT_ATR * atr
    at_resistance = np.abs(h - res) <= SUPPORT_ATR * atr
    c1 = np.r_[np.nan, c[:-1]]
    o1 = np.r_[np.nan, o[:-1]]
    h1 = np.r_[np.nan, h[:-1]]
    l1 = np.r_[np.nan, l[:-1]]
    body1 = np.r_[np.nan, body[:-1]]
    prior_decline = c1 < np.r_[[np.nan] * (CONTEXT_BARS + 1), c[:-(CONTEXT_BARS + 1)]]
    prior_advance = c1 > np.r_[[np.nan] * (CONTEXT_BARS + 1), c[:-(CONTEXT_BARS + 1)]]
    p = pd.DataFrame(index=d.index)
    p["bull_engulfing"] = (c1 < o1) & (c > o) & (o <= c1) & (c >= o1) & (body > body1) & prior_decline
    p["bear_engulfing"] = (c1 > o1) & (c < o) & (o >= c1) & (c <= o1) & (body > body1) & prior_advance
    p["hammer"] = (lower >= HAMMER_WICK_BODY * np.maximum(body, 0.05 * rng)) & (upper <= SMALL_WICK_BODY * np.maximum(body, 0.05 * rng)) \
        & ((c - l) / rng >= 0.6) & prior_decline
    p["shooting_star"] = (upper >= HAMMER_WICK_BODY * np.maximum(body, 0.05 * rng)) & (lower <= SMALL_WICK_BODY * np.maximum(body, 0.05 * rng)) \
        & ((h - c) / rng >= 0.6) & prior_advance
    # three-candle stars: t-2 long body, t-1 small body, t closes beyond the midpoint of t-2's body
    o2, c2 = np.r_[[np.nan] * 2, o[:-2]], np.r_[[np.nan] * 2, c[:-2]]
    body2 = np.abs(c2 - o2)
    rng2 = np.r_[[np.nan] * 2, rng[:-2]]
    mid2 = (o2 + c2) / 2
    decline_before_star = np.r_[[np.nan] * 2, prior_decline[:-2]].astype(float) == 1
    advance_before_star = np.r_[[np.nan] * 2, prior_advance[:-2]].astype(float) == 1
    p["morning_star"] = (c2 < o2) & (body2 >= LONG_BODY_RANGE * rng2) & (body1 <= STAR_BODY_FRAC * body2) & (c > o) \
        & (c > mid2) & decline_before_star
    p["evening_star"] = (c2 > o2) & (body2 >= LONG_BODY_RANGE * rng2) & (body1 <= STAR_BODY_FRAC * body2) & (c < o) \
        & (c < mid2) & advance_before_star
    p["inside_bar"] = (h <= h1) & (l >= l1)
    p["at_support"] = at_support
    p["at_resistance"] = at_resistance
    return p.fillna(False).astype(bool)


# ------------------------------------------------------------------------------------------------ weekly regime
WEEKLY_PERMISSION = {"BULL_TREND": "LONG_PREFERRED", "BEAR_TREND": "SHORT_PREFERRED", "RANGE": "EDGES_ONLY",
                     "TRANSITION": "REDUCED_CONFIRMATION_REQUIRED", "HIGH_VOL_DISORDERLY": "BLOCKED", "UNKNOWN": "NONE"}


def weekly_regime(wk: pd.DataFrame, i: int | None = None, x: pd.DataFrame | None = None,
                  piv_all: pd.DataFrame | None = None) -> dict:
    """Regime at completed week i (default: the last row) of `wk` (completed weekly bars only): E011-E015, E018.
    Precomputed causal indicators `x` and pivots `piv_all` for the whole series may be passed; values at row i use
    rows <= i only, so this equals recomputing on wk.iloc[:i+1]."""
    i = len(wk) - 1 if i is None else i
    n = i + 1
    asof = str(pd.Timestamp(wk.iloc[i].get("last_day", wk.index[i])).date()) if n else ""
    if n < 30:
        return {"regime": "UNKNOWN", "permission": "NONE", "reason": "fewer than 30 completed weeks", "evidence": {}}
    h, l, c = (wk[k].to_numpy(float)[:n] for k in ("high", "low", "close"))
    x = indicators(wk.iloc[:n]) if x is None else x.iloc[:n]
    piv = (confirmed_pivots(h, l, PIVOT_K_WEEKLY) if piv_all is None else piv_all.iloc[:n]).iloc[-1]
    sma20 = x.sma20.to_numpy()
    slope = sma20[-1] / sma20[-5] - 1 if np.isfinite(sma20[-5]) else np.nan
    atr_pct = (x.atr14.to_numpy() / c)
    hist = atr_pct[-104:]
    vol_rank = float((hist[np.isfinite(hist)] <= atr_pct[-1]).mean()) if np.isfinite(atr_pct[-1]) else np.nan
    ev = {"swing_high_last": piv.sh_last, "swing_high_prev": piv.sh_prev, "swing_low_last": piv.sl_last,
          "swing_low_prev": piv.sl_prev, "close": c[-1], "sma20w": sma20[-1], "sma20w_slope_4w": slope,
          "atr_pct_rank_104w": vol_rank}
    two_each = not np.isnan([piv.sh_last, piv.sh_prev, piv.sl_last, piv.sl_prev]).any()
    one_each = not np.isnan([piv.sh_last, piv.sl_last]).any()
    if not one_each or not np.isfinite(slope):
        regime, why = "UNKNOWN", "no confirmed weekly swing high/low pair yet, or too little history for the SMA20w slope"
    elif vol_rank >= 0.95:
        regime, why = "HIGH_VOL_DISORDERLY", "weekly ATR% in the top 5% of its 104-week history (E018)"
    elif not two_each:
        regime, why = "TRANSITION", "only one confirmed swing of a kind: structure not yet established (E014)"
    elif vol_rank >= 0.95:
        regime, why = "HIGH_VOL_DISORDERLY", "weekly ATR% in the top 5% of its 104-week history (E018)"
    elif piv.sh_last > piv.sh_prev and piv.sl_last > piv.sl_prev and c[-1] > sma20[-1] and c[-1] > piv.sl_last:
        regime, why = "BULL_TREND", "higher high and higher low, close above rising-structure support and SMA20w (E011, E015)"
    elif piv.sh_last < piv.sh_prev and piv.sl_last < piv.sl_prev and c[-1] < sma20[-1] and c[-1] < piv.sh_last:
        regime, why = "BEAR_TREND", "lower high and lower low, close below SMA20w (E012, E015)"
    elif abs(slope) < 0.02 and min(piv.sl_last, piv.sl_prev) <= c[-1] <= max(piv.sh_last, piv.sh_prev):
        regime, why = "RANGE", "flat SMA20w (|4-week slope| < 2%) and close inside the confirmed swing envelope (E013)"
    else:
        regime, why = "TRANSITION", "structure broken or mixed without confirmation (E014)"
    return {"regime": regime, "permission": WEEKLY_PERMISSION[regime], "reason": why, "asof": asof,
            "evidence": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v))
                         for k, v in ev.items()}}


# ------------------------------------------------------------------------------------------------ daily setups
def daily_setups(d: pd.DataFrame, k: int, weekly: dict, x_all: pd.DataFrame | None = None,
                 piv_all: pd.DataFrame | None = None) -> list[dict]:
    """Daily setup candidates at bar k (E023/E024 pullbacks, E027/E066/E067 breakouts, E026/E031/E032 range edges).
    Each carries a structural invalidation price, the first opposing obstacle and its rule ids. Levels come from OHLCV,
    never from chart pixels. Precomputed causal indicators/pivots for the whole series may be passed (rows <= k used)."""
    if k < 252:
        return []
    o, h, l, c, v = (d[x].to_numpy(float)[: k + 1] for x in ("open", "high", "low", "close", "volume"))
    x = indicators(d.iloc[: k + 1]) if x_all is None else x_all.iloc[: k + 1]
    atr, sma20, sma50 = x.atr14.iloc[-1], x.sma20.iloc[-1], x.sma50.iloc[-1]
    vol_ratio = x.vol_ratio.iloc[-1]
    piv = confirmed_pivots(h, l, PIVOT_K_DAILY) if piv_all is None else piv_all.iloc[: k + 1].reset_index(drop=True)
    sh_last, sl_last = piv.sh_last.iloc[-1], piv.sl_last.iloc[-1]
    hh = pd.Series(h)
    ll = pd.Series(l)
    prior20_high = hh.iloc[-21:-1].max()
    prior20_low = ll.iloc[-21:-1].min()
    hi252, lo252 = hh.iloc[-253:-1].max(), ll.iloc[-253:-1].min()      # prior bars only (audit fix A-01)
    close, rng_ = c[-1], h[-1] - l[-1]
    cl_loc = (close - l[-1]) / rng_ if rng_ > 0 else 0.5
    perm = weekly.get("permission", "NONE")
    atr_now = x.atr14.iloc[-1]
    measured = float(prior20_high - prior20_low)                        # X024 measured-move height

    def obstacle_above(level):
        """Nearest MATERIAL opposing level above (X021): a confirmed swing high or the prior 252-bar high at least
        0.5 ATR above `level`; with none (e.g. a breakout to new highs) a measured move of the prior 20-bar range."""
        cands = [p for p in (piv.sh_last.iloc[-1], piv.sh_prev.iloc[-1], hi252)
                 if np.isfinite(p) and p >= level + MIN_OBSTACLE_ATR * atr_now]
        return float(min(cands)) if cands else float(level + max(measured, MIN_OBSTACLE_ATR * atr_now))

    def obstacle_below(level):
        cands = [p for p in (piv.sl_last.iloc[-1], piv.sl_prev.iloc[-1], lo252)
                 if np.isfinite(p) and p <= level - MIN_OBSTACLE_ATR * atr_now]
        return float(max(cands)) if cands else float(max(level - max(measured, MIN_OBSTACLE_ATR * atr_now), 0.01))

    out = []
    touched20_long = (np.abs(l[-5:] - x.sma20.to_numpy()[-5:]) <= 0.5 * atr).any()
    touched20_short = (np.abs(h[-5:] - x.sma20.to_numpy()[-5:]) <= 0.5 * atr).any()
    if perm in ("LONG_PREFERRED", "REDUCED_CONFIRMATION_REQUIRED") and close > sma50 and sma20 > sma50 \
            and touched20_long and np.isfinite(sl_last) and close > sl_last:
        out.append({"family": "TREND_PULLBACK_LONG", "direction": "LONG", "rule_ids": ["E023", "E074", "E083"],
                    "invalidation": float(min(sl_last, l[-5:].min())), "first_obstacle": obstacle_above(close),
                    "zone": [float(sma20 - 0.5 * atr), float(sma20 + 0.5 * atr)]})
    if perm in ("SHORT_PREFERRED", "REDUCED_CONFIRMATION_REQUIRED") and close < sma50 and sma20 < sma50 \
            and touched20_short and np.isfinite(sh_last) and close < sh_last:
        out.append({"family": "TREND_PULLBACK_SHORT", "direction": "SHORT", "rule_ids": ["E024", "E074", "E084"],
                    "invalidation": float(max(sh_last, h[-5:].max())), "first_obstacle": obstacle_below(close),
                    "zone": [float(sma20 - 0.5 * atr), float(sma20 + 0.5 * atr)]})
    if perm != "BLOCKED" and close > prior20_high and cl_loc >= 0.6:
        out.append({"family": "BREAKOUT_LONG", "direction": "LONG", "rule_ids": ["E027", "E066", "E143"] + (["E101"] if vol_ratio >= REL_VOL_EXPANSION else []),
                    "invalidation": float(prior20_high - atr), "first_obstacle": obstacle_above(close),
                    "zone": [float(prior20_high), float(close)], "breakout_level": float(prior20_high)})
    if perm != "BLOCKED" and close < prior20_low and cl_loc <= 0.4:
        out.append({"family": "BREAKDOWN_SHORT", "direction": "SHORT", "rule_ids": ["E027", "E067", "E143"] + (["E101"] if vol_ratio >= REL_VOL_EXPANSION else []),
                    "invalidation": float(prior20_low + atr), "first_obstacle": obstacle_below(close),
                    "zone": [float(close), float(prior20_low)], "breakout_level": float(prior20_low)})
    lo60, hi60 = ll.iloc[-60:].min(), hh.iloc[-60:].max()
    pos = (close - lo60) / (hi60 - lo60) if hi60 > lo60 else 0.5
    if perm in ("EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED", "LONG_PREFERRED") and pos <= 0.2 and cl_loc >= 0.5 and close > lo60:
        out.append({"family": "RANGE_SUPPORT_LONG", "direction": "LONG", "rule_ids": ["E026", "E031"],
                    "invalidation": float(lo60 - 0.5 * atr), "first_obstacle": float((lo60 + hi60) / 2),
                    "zone": [float(lo60), float(lo60 + 0.2 * (hi60 - lo60))]})
    if perm in ("EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED", "SHORT_PREFERRED") and pos >= 0.8 and cl_loc <= 0.5 and close < hi60:
        out.append({"family": "RANGE_RESISTANCE_SHORT", "direction": "SHORT", "rule_ids": ["E026", "E032"],
                    "invalidation": float(hi60 + 0.5 * atr), "first_obstacle": float((lo60 + hi60) / 2),
                    "zone": [float(hi60 - 0.2 * (hi60 - lo60)), float(hi60)]})
    for s in out:
        s.update({"atr": float(atr), "close": float(close), "source_bar_end": str(pd.Timestamp(d.date.iloc[k]).date()),
                  "expiry_daily_bars": 3})
    return out


# ------------------------------------------------------------------------------------------------ hourly triggers
def hourly_trigger(hb: pd.DataFrame, direction: str) -> dict:
    """Trigger on the LAST completed hourly bar (the as-of session's final bar): structure break of the previous
    7 bars, engulfing, or wick rejection, in the setup direction (decision-engine spec 6.1)."""
    if len(hb) < 9:
        return {"trigger": "NO_TRIGGER", "reason": "fewer than 9 hourly bars"}
    o, h, l, c = (hb[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    body, rng, upper, lower = candle_parts(o, h, l, c)
    prev_hi, prev_lo = h[-8:-1].max(), l[-8:-1].min()
    if direction == "LONG":
        if c[-1] > prev_hi:
            return {"trigger": "STRUCTURE_BREAK", "level": float(prev_hi), "rule_ids": ["E066", "E121"]}
        if c[-2] < o[-2] and c[-1] > o[-1] and o[-1] <= c[-2] and c[-1] >= o[-2]:
            return {"trigger": "ENGULFING_CONFIRMATION", "level": float(h[-1]), "rule_ids": ["E041"]}
        if lower[-1] >= HAMMER_WICK_BODY * max(body[-1], 0.05 * rng[-1]) and (c[-1] - l[-1]) / rng[-1] >= 0.6:
            return {"trigger": "REJECTION_CANDLE", "level": float(h[-1]), "rule_ids": ["E043"]}
    else:
        if c[-1] < prev_lo:
            return {"trigger": "STRUCTURE_BREAK", "level": float(prev_lo), "rule_ids": ["E067", "E122"]}
        if c[-2] > o[-2] and c[-1] < o[-1] and o[-1] >= c[-2] and c[-1] <= o[-2]:
            return {"trigger": "ENGULFING_CONFIRMATION", "level": float(l[-1]), "rule_ids": ["E042"]}
        if upper[-1] >= HAMMER_WICK_BODY * max(body[-1], 0.05 * rng[-1]) and (h[-1] - c[-1]) / rng[-1] >= 0.6:
            return {"trigger": "REJECTION_CANDLE", "level": float(l[-1]), "rule_ids": ["E044"]}
    return {"trigger": "NO_TRIGGER", "reason": "no structure break, engulfing or rejection on the last hourly bar"}


# ------------------------------------------------------------------------------------------------ data gates
def data_gates(d: pd.DataFrame, k: int, hourly_bars_in_session: int | None, expected_hourly: int = 7) -> list[RuleEvent]:
    asof = str(pd.Timestamp(d.date.iloc[k]).date())
    ev = []
    row = d.iloc[: k + 1]
    bad = ((row[["open", "high", "low", "close"]] <= 0).any(axis=1) | (row.high < row[["open", "close"]].max(axis=1))
           | (row.low > row[["open", "close"]].min(axis=1))).iloc[-200:].any()
    ev.append(RuleEvent("E003", "BLOCKED" if bad else "FIRED", "N/A", "1D", asof, asof, None,
                        "OHLC_INVALID" if bad else "OHLC_OK"))
    ev.append(RuleEvent("E007", "FIRED" if k >= 199 else "BLOCKED", "N/A", "1D", asof, asof, None,
                        "WARM" if k >= 199 else "INSUFFICIENT_HISTORY"))
    adv = float((row.close * row.volume).iloc[-20:].mean())
    liquid = adv >= MIN_ADV_USD and float(row.close.iloc[-1]) >= MIN_PRICE
    ev.append(RuleEvent("E006", "FIRED" if liquid else "BLOCKED", "N/A", "1D", asof, asof, adv,
                        "LIQUID" if liquid else "ADV_OR_PRICE_BELOW_MINIMUM"))
    if hourly_bars_in_session is None:
        ev.append(RuleEvent("E001", "UNKNOWN", "N/A", "1H", asof, asof, None, "NO_HOURLY_BARS"))
    else:
        complete = hourly_bars_in_session >= expected_hourly
        ev.append(RuleEvent("E001", "FIRED" if complete else "BLOCKED", "N/A", "1H", asof, asof, None,
                            "SESSION_COMPLETE" if complete else "INCOMPLETE_HOURLY_SESSION"))
    return ev


IMPLEMENTED = {rid: {"detector": det, "tests": ["tests/test_rules.py"]} for rid, det in {
    "E001": "data_gates", "E003": "data_gates", "E006": "data_gates", "E007": "data_gates",
    "E011": "weekly_regime", "E012": "weekly_regime", "E013": "weekly_regime", "E014": "weekly_regime",
    "E015": "weekly_regime", "E018": "weekly_regime",
    "E023": "daily_setups", "E024": "daily_setups", "E026": "daily_setups", "E027": "daily_setups",
    "E031": "daily_setups", "E032": "daily_setups", "E066": "daily_setups/hourly_trigger",
    "E067": "daily_setups/hourly_trigger", "E074": "daily_setups", "E083": "daily_setups", "E084": "daily_setups",
    "E101": "daily_setups", "E143": "daily_setups",
    "E041": "candle_patterns/hourly_trigger", "E042": "candle_patterns/hourly_trigger",
    "E043": "candle_patterns/hourly_trigger", "E044": "candle_patterns/hourly_trigger",
    "E045": "candle_patterns", "E046": "candle_patterns", "E055": "candle_patterns",
    "E111": "trading.plan (chase distance)", "E121": "hourly_trigger", "E122": "hourly_trigger",
    "E139": "trading.plan (net reward-to-risk)", "E149": "trading.plan (net reward-to-risk)",
    "E150": "trading.engine (final gate)",
    "X001": "trading.plan (structure stop)", "X010": "trading.plan (stop infeasibility)",
    "X011": "trading.plan (ATR buffer)", "X021": "trading.plan (first obstacle target)",
    "X071": "trading.plan (time stop)", "X083": "backtest (gap-through stop fills at open)",
    "X091": "trading.risk (open-risk cap)", "X093": "trading.risk (daily loss kill-switch)",
    "X114": "trading.engine (hard stop beats model)"}.items()}

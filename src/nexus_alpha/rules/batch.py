"""Evaluate the executable detectors for many samples of one ticker efficiently (indicators/pivots computed once on the
sealed series; every value at an as-of bar uses bars <= that bar only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..cv.render import rth_hourly
from ..data import bars as B
from ..data.calendar import calendar, offset_session
from ..indicators.core import indicators
from . import detectors as D

REGIMES = ["BULL_TREND", "BEAR_TREND", "RANGE", "TRANSITION", "HIGH_VOL_DISORDERLY", "UNKNOWN"]
FAMILIES = ["TREND_PULLBACK_LONG", "TREND_PULLBACK_SHORT", "BREAKOUT_LONG", "BREAKDOWN_SHORT", "RANGE_SUPPORT_LONG",
            "RANGE_RESISTANCE_SHORT"]
PATTERNS = ["bull_engulfing", "bear_engulfing", "hammer", "shooting_star", "morning_star", "evening_star", "inside_bar"]


def completed_weeks(d: pd.DataFrame) -> pd.DataFrame:
    ws = B.week_start(d.date)
    nxt = offset_session(d.date, 1)
    done = pd.Series(B.week_start(pd.Series(nxt)).values != ws.values, index=d.index) | pd.isna(nxt)
    wk = (d.assign(ws=ws.values).groupby("ws").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                    close=("close", "last"), volume=("volume", "sum"),
                                                    last_day=("date", "max")).reset_index())
    return wk[wk.last_day.isin(set(d.date[done.values]))].reset_index(drop=True)


def evaluate_ticker(ticker: str, asof_dates, daily: pd.DataFrame | None = None, hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    d = B.read_daily(ticker) if daily is None else daily
    h = rth_hourly(B.read_hourly(ticker), early) if hourly is None else hourly
    x = indicators(d)
    pivd = D.confirmed_pivots(d.high.to_numpy(float), d.low.to_numpy(float), D.PIVOT_K_DAILY)
    pats = D.candle_patterns(d, x.atr14)
    wk = completed_weeks(d)
    xw = indicators(wk)
    pivw = D.confirmed_pivots(wk.high.to_numpy(float), wk.low.to_numpy(float), D.PIVOT_K_WEEKLY)
    hsess = pd.DatetimeIndex(h.timestamp).tz_convert("America/New_York").tz_localize(None).normalize() if len(h) else pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(d.date)
    rows = []
    for a in pd.DatetimeIndex(asof_dates):
        k = int(dates.searchsorted(a))
        if k >= len(dates) or dates[k] != a:
            continue
        wi = int(np.searchsorted(wk.last_day.values.astype("datetime64[ns]"), np.datetime64(a, "ns"), side="right")) - 1
        reg = D.weekly_regime(wk, wi, xw, pivw) if wi >= 0 else {"regime": "UNKNOWN", "permission": "NONE"}
        setups = D.daily_setups(d, k, reg, x, pivd)
        e = int(np.searchsorted(hsess.values, np.datetime64(a, "ns"), side="right"))
        hb = h.iloc[max(0, e - 14): e]
        cur = len(hb) and hsess[e - 1] == a
        trig = {s: (D.hourly_trigger(hb, s) if cur else {"trigger": "NO_TRIGGER", "reason": "no hourly bars for the session"})
                for s in ("LONG", "SHORT")}
        r = {"ticker": ticker, "asof_date": a, "regime": reg["regime"], "permission": reg["permission"],
             "setups_json": json.dumps(setups), "trigger_long": trig["LONG"]["trigger"], "trigger_short": trig["SHORT"]["trigger"],
             "trigger_long_json": json.dumps(trig["LONG"]), "trigger_short_json": json.dumps(trig["SHORT"]),
             "atr14": float(x.atr14.iloc[k]), "close": float(d.close.iloc[k]), "k": k,
             "hourly_bars_session": int((hsess == a).sum()) if len(hsess) else 0}
        for g in REGIMES:
            r[f"rule_regime_{g.lower()}"] = float(reg["regime"] == g)
        fams = {s["family"] for s in setups}
        for f in FAMILIES:
            r[f"rule_setup_{f.lower()}"] = float(f in fams)
        for p in PATTERNS:
            r[f"rule_candle_{p}"] = float(pats[p].iloc[k])
        r["rule_trigger_long"] = float(trig["LONG"]["trigger"] != "NO_TRIGGER")
        r["rule_trigger_short"] = float(trig["SHORT"]["trigger"] != "NO_TRIGGER")
        rows.append(r)
    return pd.DataFrame(rows)

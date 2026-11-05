"""Market-context features for v2 (all known at the close of the as-of session t, used at 09:00 New York on t+1).

Two kinds:
* date-level `mkt_*` (same for every stock on a date): S&P 500 (SPY) trend, volatility and drawdown, the last session's
  hourly shape, VIX level and term structure, Treasury curve, dollar, oil, gold, copper/gold, credit (HYG, LQD vs IEF),
  size/growth/emerging-market relative strength, sector breadth and dispersion;
* stock-level `rel_*`: beta and correlation to SPY, idiosyncratic volatility, relative strength vs SPY and vs the stock's
  own sector ETF. The sector is inferred point-in-time as the SPDR sector ETF with the highest trailing 252-session
  return correlation (re-chosen every session; no look-ahead, no external classification needed).
Inputs are pandas objects so the same functions serve training (sealed stores) and live inference (Yahoo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]


def _lr(s: pd.Series, n: int) -> pd.Series:
    return np.log(s / s.shift(n))


def date_features(etf_close: pd.DataFrame, spy_daily: pd.DataFrame, spy_hourly: pd.DataFrame | None,
                  mkt: pd.DataFrame) -> pd.DataFrame:
    """etf_close: date x ETF closes; spy_daily: SPY OHLC by date; spy_hourly: SPY RTH hourly (UTC timestamp);
    mkt: date x Yahoo series (vix, vix3m, y3m, y5y, y10y, y30y, dxy, wti, gold, copper, spx)."""
    sd = spy_daily.set_index("date").sort_index()
    c = sd.close
    r1 = np.log(c).diff()
    f = pd.DataFrame(index=sd.index)
    for n in (1, 5, 21, 63, 126, 252):
        f[f"mkt_spy_ret{n}"] = _lr(c, n)
    f["mkt_spy_rv20"] = r1.rolling(20).std() * np.sqrt(252)
    f["mkt_spy_rv63"] = r1.rolling(63).std() * np.sqrt(252)
    f["mkt_spy_dd252"] = c / c.rolling(252, min_periods=126).max() - 1
    f["mkt_spy_sma50"] = c / c.rolling(50).mean() - 1
    f["mkt_spy_sma200"] = c / c.rolling(200).mean() - 1
    f["mkt_spy_gap"] = np.log(sd.open / c.shift(1))
    f["mkt_spy_close_loc"] = (c - sd.low) / (sd.high - sd.low).replace(0, np.nan)
    if spy_hourly is not None and len(spy_hourly):
        h = spy_hourly.copy()
        et = pd.DatetimeIndex(h.timestamp).tz_convert("America/New_York")
        h["day"] = et.tz_localize(None).normalize()
        g = h.groupby("day")
        last = g.tail(1).set_index("day")
        first = g.head(1).set_index("day")
        f["mkt_spy_last_hour_ret"] = np.log(last.close / last.open).reindex(f.index)
        f["mkt_spy_first_hour_ret"] = np.log(first.close / first.open).reindex(f.index)
    m = mkt.reindex(f.index).ffill(limit=5)
    f["mkt_vix_log"] = np.log(m.vix)
    f["mkt_vix_chg5"] = _lr(m.vix, 5)
    f["mkt_vix_vs_rv20"] = np.log(m.vix / (f.mkt_spy_rv20 * 100))
    f["mkt_vix_term"] = np.log(m.vix3m / m.vix) if "vix3m" in m else np.nan
    f["mkt_y10y"] = m.y10y
    f["mkt_term_10y_3m"] = m.y10y - m.y3m
    f["mkt_term_30y_5y"] = m.y30y - m.y5y
    f["mkt_y10y_chg21"] = m.y10y - m.y10y.shift(21)
    f["mkt_y3m_chg63"] = m.y3m - m.y3m.shift(63)
    f["mkt_dxy_ret21"] = _lr(m.dxy, 21)
    f["mkt_dxy_ret63"] = _lr(m.dxy, 63)
    wti = m.wti.where(m.wti > 0)                         # WTI settled below zero on 2020-04-20: undefined log return
    f["mkt_wti_ret21"] = _lr(wti, 21)
    f["mkt_wti_ret63"] = _lr(wti, 63)
    f["mkt_gold_ret63"] = _lr(m.gold, 63)
    f["mkt_copper_gold_chg63"] = _lr(m.copper / m.gold, 63)
    e = etf_close.reindex(f.index)
    rel = lambda a, b, n: _lr(e[a], n) - _lr(e[b], n) if a in e and b in e else np.nan
    f["mkt_hyg_ief_rel21"] = rel("HYG", "IEF", 21)
    f["mkt_lqd_ief_rel21"] = rel("LQD", "IEF", 21)
    f["mkt_tlt_ret21"] = _lr(e["TLT"], 21) if "TLT" in e else np.nan
    f["mkt_iwm_spy_rel21"] = rel("IWM", "SPY", 21)
    f["mkt_qqq_spy_rel21"] = rel("QQQ", "SPY", 21)
    f["mkt_eem_spy_rel63"] = rel("EEM", "SPY", 63)
    sec = e[[s for s in SECTORS if s in e]]
    f["mkt_sector_breadth50"] = (sec > sec.rolling(50).mean()).where(sec.notna()).mean(axis=1)
    f["mkt_sector_disp21"] = np.log(sec / sec.shift(21)).std(axis=1)
    f["mkt_defensive_rel63"] = (_lr(e[["XLP", "XLU", "XLV"]], 63).mean(axis=1)
                                - _lr(e[["XLY", "XLK", "XLI"]], 63).mean(axis=1))
    return f.replace([np.inf, -np.inf], np.nan)


def stock_relative(daily: pd.DataFrame, etf_close: pd.DataFrame) -> pd.DataFrame:
    """Per-session stock-vs-market features for one stock (daily: date, close). Index = the stock's dates."""
    s = daily.set_index("date").close.sort_index()
    e = etf_close.reindex(s.index)
    rs = np.log(s).diff()
    rm = np.log(e["SPY"]).diff()
    f = pd.DataFrame(index=s.index)
    cov = rs.rolling(252, min_periods=126).cov(rm)
    var = rm.rolling(252, min_periods=126).var()
    beta = cov / var
    f["rel_beta252"] = beta
    f["rel_corr63"] = rs.rolling(63, min_periods=42).corr(rm)
    resid = rs - beta * rm
    f["rel_idio_vol63"] = resid.rolling(63, min_periods=42).std() * np.sqrt(252)
    for n in (5, 21, 63, 126):
        f[f"rel_spy_rs{n}"] = _lr(s, n) - _lr(e["SPY"], n)
    secs = [x for x in SECTORS if x in e]
    rsec = np.log(e[secs]).diff()
    corr = pd.concat({x: rs.rolling(252, min_periods=126).corr(rsec[x]) for x in secs}, axis=1)
    best = corr.dropna(how="all").idxmax(axis=1).reindex(corr.index)
    f["rel_sector_corr252"] = corr.max(axis=1)
    # relative strength vs the inferred sector, using the sector chosen at t for the whole look-back window
    for n in (21, 63):
        sec_ret = pd.Series(np.nan, index=s.index)
        for x in secs:
            m = best == x
            sec_ret[m] = _lr(e[x], n)[m]
        f[f"rel_sector_rs{n}"] = _lr(s, n) - sec_ret
    f["rel_sector_id"] = best.map({x: i for i, x in enumerate(SECTORS)}).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)

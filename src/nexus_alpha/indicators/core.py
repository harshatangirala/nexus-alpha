"""Causal indicators with TriFrame's frozen conventions (us_stocks_ohlcv/scripts/tf_data.py::indicators), so values can
be verified against the frozen file final/meta/indicators/ on the development period:

* SMA20 / SMA50: simple rolling means of close.
* MACD(12, 26, 9): EMA with adjust=False seeded at the first close (pandas ewm span convention); signal = EMA9 of MACD.
* RSI(14): Wilder smoothing as ewm(alpha=1/14, adjust=False) of gains and losses; 100 if no losses, 50 if flat.
* Bollinger(20, 2): population standard deviation (ddof=0); width = (upper - lower) / middle.
* ATR(14): ewm(alpha=1/14, adjust=False) of true range (first bar: high - low).
* Volume: 20-bar mean including the current bar; ratio = volume / mean.
* warm: True from bar WARM-1 on (EMA seeding effects are negligible after ~200 bars; earlier rows are flagged).

Every value at bar t uses bars <= t only. Indicators are computed per timeframe on that timeframe's own completed bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WARM = 200
FEATURE_VERSION = "nexus-structured-v1"


def indicators(d: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = (d[k].astype(float) for k in ("close", "high", "low", "volume"))
    o = pd.DataFrame(index=d.index)
    o["sma20"] = c.rolling(20).mean()
    o["sma50"] = c.rolling(50).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    o["macd"], o["macd_sig"], o["macd_hist"] = macd, sig, macd - sig
    dl = c.diff()
    ag = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-dl).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    o["rsi14"] = rsi.where(al > 0, np.where(ag > 0, 100.0, 50.0))
    sd = c.rolling(20).std(ddof=0)
    o["bb_mid"] = o.sma20
    o["bb_up"] = o.sma20 + 2 * sd
    o["bb_lo"] = o.sma20 - 2 * sd
    o["bb_width"] = (o.bb_up - o.bb_lo) / o.bb_mid.abs().replace(0, np.nan)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    o["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    o["vol_avg20"] = v.rolling(20).mean()
    o["vol_ratio"] = v / o.vol_avg20.replace(0, np.nan)
    o["warm"] = np.arange(len(d)) >= WARM - 1
    return o


def ewma_sigma(close: pd.Series) -> pd.Series:
    """Daily return volatility, sqrt of EWMA(r^2) with alpha 0.06 (lambda 0.94), TriFrame's label convention."""
    r1 = close.astype(float).pct_change()
    return np.sqrt((r1 ** 2).ewm(alpha=0.06, adjust=False).mean())


def _scale_free(prefix: str, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"].astype(float)
    atr = ind.atr14.replace(0, np.nan)
    f = pd.DataFrame(index=bars.index)
    f[f"{prefix}_atr_pct"] = ind.atr14 / c
    f[f"{prefix}_rsi14"] = ind.rsi14
    f[f"{prefix}_macd_atr"] = ind.macd / atr
    f[f"{prefix}_macd_hist_atr"] = ind.macd_hist / atr
    f[f"{prefix}_close_sma20"] = c / ind.sma20 - 1
    f[f"{prefix}_close_sma50"] = c / ind.sma50 - 1
    f[f"{prefix}_sma20_sma50"] = ind.sma20 / ind.sma50 - 1
    band = (ind.bb_up - ind.bb_lo).replace(0, np.nan)
    f[f"{prefix}_bb_pctb"] = (c - ind.bb_lo) / band
    f[f"{prefix}_bb_width"] = ind.bb_width
    f[f"{prefix}_log_vol_ratio"] = np.log(ind.vol_ratio.clip(lower=1e-3))
    return f


def daily_features(d: pd.DataFrame) -> pd.DataFrame:
    """Scale-free daily features at each bar (as of that bar's close)."""
    ind = indicators(d)
    f = _scale_free("d", d, ind)
    c, o, h, l = (d[k].astype(float) for k in ("close", "open", "high", "low"))
    lr = np.log(c).diff()
    for n in (1, 5, 21, 63):
        f[f"d_ret{n}"] = c / c.shift(n) - 1
    f["d_rv5"] = lr.rolling(5).std(ddof=0) * np.sqrt(252)
    f["d_rv20"] = lr.rolling(20).std(ddof=0) * np.sqrt(252)
    f["d_gap"] = o / c.shift() - 1
    f["d_range_atr"] = (h - l) / ind.atr14.replace(0, np.nan)
    f["d_close_loc"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    f["d_dist_high252"] = c / h.rolling(252, min_periods=126).max() - 1
    f["d_sigma_ewma"] = ewma_sigma(c)
    f["d_warm"] = ind.warm.astype(float)
    return f


STRUCTURED_DAILY = ["d_atr_pct", "d_rsi14", "d_macd_atr", "d_macd_hist_atr", "d_close_sma20", "d_close_sma50",
                    "d_sma20_sma50", "d_bb_pctb", "d_bb_width", "d_log_vol_ratio", "d_ret1", "d_ret5", "d_ret21",
                    "d_ret63", "d_rv5", "d_rv20", "d_gap", "d_range_atr", "d_close_loc", "d_dist_high252",
                    "d_sigma_ewma"]
STRUCTURED_WEEKLY = ["w_atr_pct", "w_rsi14", "w_macd_atr", "w_macd_hist_atr", "w_close_sma20", "w_close_sma50",
                     "w_sma20_sma50", "w_bb_pctb", "w_bb_width", "w_log_vol_ratio", "w_ret1", "w_ret4", "w_wtd_ret"]
STRUCTURED_HOURLY = ["h_atr_pct", "h_rsi14", "h_macd_atr", "h_macd_hist_atr", "h_close_sma20", "h_close_sma50",
                     "h_sma20_sma50", "h_bb_pctb", "h_bb_width", "h_log_vol_ratio", "h_ret7", "h_ret35",
                     "h_last_session_close_loc"]
STRUCTURED_ALL = STRUCTURED_DAILY + STRUCTURED_WEEKLY + STRUCTURED_HOURLY


def structured_features(daily: pd.DataFrame, hourly: pd.DataFrame, asof_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily, weekly and hourly features as of each requested session close (one row per as-of date).

    Weekly features use COMPLETED weeks only (the week in progress is summarised separately by w_wtd_ret, its
    week-to-date return); hourly features use the regular-session hourly bars up to and including the as-of session.
    """
    d = daily.sort_values("date").reset_index(drop=True)
    fd = daily_features(d)
    fd.insert(0, "date", d.date.values)
    asof = pd.DatetimeIndex(pd.to_datetime(asof_dates)).normalize()
    out = fd.set_index("date").reindex(asof)

    # weekly: completed weeks only
    from ..data.bars import week_start
    from ..data.calendar import offset_session
    ws = week_start(d.date)
    nxt = offset_session(d.date, 1)
    week_done = pd.Series(week_start(pd.Series(nxt)).values != ws.values, index=d.index) | pd.isna(nxt)
    wk = (d.assign(ws=ws.values).groupby("ws")
          .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
               volume=("volume", "sum"), last_day=("date", "max")).reset_index())
    last_day_done = set(d.date[week_done.values])
    wk = wk[wk.last_day.isin(last_day_done)].reset_index(drop=True)
    wi = indicators(wk)
    fw = _scale_free("w", wk, wi)
    fw["w_ret1"] = wk.close / wk.close.shift(1) - 1
    fw["w_ret4"] = wk.close / wk.close.shift(4) - 1
    fw["avail"] = wk.last_day.values
    # as-of mapping: last completed week whose last_day <= as-of date
    pos = np.searchsorted(fw.avail.values.astype("datetime64[ns]"), asof.values.astype("datetime64[ns]"), side="right") - 1
    wcols = [c for c in STRUCTURED_WEEKLY if c != "w_wtd_ret"]
    W = pd.DataFrame(np.nan, index=asof, columns=wcols)
    ok = pos >= 0
    W.loc[asof[ok], wcols] = fw[wcols].values[pos[ok]]
    # week-to-date return: as-of close vs the last completed week's close
    last_wclose = np.where(ok, wk.close.values[np.maximum(pos, 0)], np.nan)
    W["w_wtd_ret"] = d.set_index("date").close.reindex(asof).values / last_wclose - 1

    # hourly: bars of sessions up to the as-of session
    if len(hourly):
        hh = hourly.sort_values("timestamp").reset_index(drop=True)
        hi = indicators(hh)
        fh = _scale_free("h", hh, hi)
        c = hh.close.astype(float)
        fh["h_ret7"] = c / c.shift(7) - 1
        fh["h_ret35"] = c / c.shift(35) - 1
        sess = pd.DatetimeIndex(hh.timestamp).tz_convert("America/New_York").tz_localize(None).normalize()
        g = pd.DataFrame({"s": sess, "h": hh.high.values, "l": hh.low.values, "c": hh.close.values})
        hs = g.groupby("s").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
        loc = ((hs.c - hs.l) / (hs.h - hs.l).replace(0, np.nan)).fillna(0.5)
        fh["sess"] = sess
        last = fh.groupby("sess").tail(1).set_index("sess")
        last_sess = last.index.values.astype("datetime64[ns]")
        hpos = np.searchsorted(last_sess, asof.values.astype("datetime64[ns]"), side="right") - 1
        hcols = [c for c in STRUCTURED_HOURLY if c != "h_last_session_close_loc"]
        Hf = pd.DataFrame(np.nan, index=asof, columns=hcols + ["h_last_session_close_loc"])
        # hourly evidence is "current" only if its last session IS the as-of session (no stale carry-forward)
        cur = (hpos >= 0) & (last_sess[np.maximum(hpos, 0)] == asof.values.astype("datetime64[ns]"))
        Hf.loc[asof[cur], hcols] = last[hcols].values[hpos[cur]]
        Hf.loc[asof[cur], "h_last_session_close_loc"] = loc.reindex(last.index).values[hpos[cur]]
    else:
        Hf = pd.DataFrame(np.nan, index=asof, columns=STRUCTURED_HOURLY)
    res = pd.concat([out[STRUCTURED_DAILY + ["d_warm"]], W[STRUCTURED_WEEKLY], Hf[STRUCTURED_HOURLY]], axis=1)
    res.index.name = "asof_date"
    return res.astype(float)

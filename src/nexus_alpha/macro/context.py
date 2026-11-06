"""Market/macro context snapshots for the Trade Desk and for building v2 training features.

* `dev_date_features()` - the date-level `mkt_*` table for 2001-2022 from the sealed ETF bars and the Yahoo dev store
  (cached in artifacts/features/mkt_dev.parquet).
* `live_date_features()` - the same features from live Yahoo data (inference only).
"""
from __future__ import annotations

import pandas as pd

from .. import paths
from ..cv.render import rth_hourly
from . import features as MF
from . import market as MK

ETF_LIST = ["SPY", "HYG", "LQD", "IEF", "TLT", "IWM", "QQQ", "EEM"] + MF.SECTORS
CACHE = "features/mkt_dev.parquet"


def dev_date_features(rebuild: bool = False) -> pd.DataFrame:
    p = paths.ARTIFACTS / CACHE
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    from ..data import etf_bars as EB
    from ..data.calendar import calendar
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    etf = EB.closes([t for t in ETF_LIST if t in EB.available()])
    spy_d = EB.read_daily("SPY")
    spy_h = rth_hourly(EB.read_hourly("SPY"), early)
    f = MF.date_features(etf, spy_d, spy_h, MK.load_dev())
    f.index.name = "date"
    f.to_parquet(paths.artifact(CACHE))
    return f


def period_date_features(beyond_seal: bool) -> pd.DataFrame:
    """dev_date_features' computation from the sources the current gate allows: development (cached) or, inside the
    opened sealed evaluation, the sealed-inclusive ETF bars and Yahoo series (never cached)."""
    if not beyond_seal:
        return dev_date_features()
    from .. import seal
    from ..data import etf_bars as EB
    from ..data.calendar import calendar
    seal.require_open("market context on/after the seal")
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    etf = EB.closes([t for t in ETF_LIST if t in EB.available()])
    spy_d = EB.read_daily("SPY")
    spy_h = rth_hourly(EB.read_hourly("SPY"), early)
    f = MF.date_features(etf, spy_d, spy_h, MK.load_full())
    f.index.name = "date"
    return f


def _yahoo_daily(sym: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.Ticker(sym).history(start=start, auto_adjust=False, actions=False)
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    return pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                         "low": d["Low"].values, "close": d["Close"].values})


def live_date_features(days: int = 800) -> pd.DataFrame:
    import yfinance as yf
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    spy = _yahoo_daily("SPY", start)
    now = pd.Timestamp.now(tz="America/New_York")
    if len(spy) and spy.date.iloc[-1] == now.tz_localize(None).normalize() and (now.hour, now.minute) < (16, 5):
        spy = spy.iloc[:-1]                                    # today's session is still forming
    etf = pd.concat({t: _yahoo_daily(t, start).set_index("date").close for t in ETF_LIST}, axis=1).reindex(spy.date)
    h = yf.Ticker("SPY").history(period="60d", interval="60m", auto_adjust=False, actions=False, prepost=False)
    hi = pd.DatetimeIndex(h.index)
    hi = hi.tz_localize("America/New_York") if hi.tz is None else hi
    spy_h = pd.DataFrame({"timestamp": hi.tz_convert("UTC"), "open": h["Open"].values, "high": h["High"].values,
                          "low": h["Low"].values, "close": h["Close"].values})
    spy_h = spy_h[pd.DatetimeIndex(spy_h.timestamp).tz_convert("America/New_York").tz_localize(None).normalize()
                  .isin(spy.date)]
    return MF.date_features(etf, spy, spy_h, MK.fetch_live(days))

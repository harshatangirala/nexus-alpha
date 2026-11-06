"""Daily market-macro series from Yahoo Finance (index levels, volatility, Treasury yields, dollar, commodities).

These are market prices, not revised statistics: the close of session t is final when it prints, so it is available for
the decision at 09:00 New York on t+1. Futures settle before 16:00 New York, also before the decision.
Development data stop before the seal; `fetch_live` serves inference only.
"""
from __future__ import annotations

import pandas as pd

from .. import paths
from ..seal import SEAL_DATE

YAHOO_SERIES = {
    "^GSPC": "spx",        # S&P 500 index (price level; benchmark and cross-check of SPY)
    "^VIX": "vix",         # CBOE 30-day implied volatility of the S&P 500
    "^VIX3M": "vix3m",     # 3-month implied volatility (from 2007-12; missing before)
    "^IRX": "y3m",         # 13-week T-bill yield, percent
    "^FVX": "y5y",         # 5-year Treasury yield, percent
    "^TNX": "y10y",        # 10-year Treasury yield, percent
    "^TYX": "y30y",        # 30-year Treasury yield, percent
    "DX-Y.NYB": "dxy",     # US dollar index
    "CL=F": "wti",         # WTI crude front-month future
    "GC=F": "gold",        # gold front-month future
    "HG=F": "copper",      # copper front-month future
}
STORE = "macro/yahoo_market_daily.parquet"


def _download(sym: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf
    d = yf.Ticker(sym).history(start=start, end=end, auto_adjust=False, actions=False)
    if d is None or not len(d):
        return pd.DataFrame(columns=["date", "close"])
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    return pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                         "low": d["Low"].values, "close": d["Close"].values}).dropna(subset=["close"])


def build_dev_store(start: str = "2000-01-01") -> pd.DataFrame:
    """Download every series up to the day before the seal and store it (long format) under artifacts/."""
    frames = []
    for sym, name in YAHOO_SERIES.items():
        d = _download(sym, start, SEAL_DATE.strftime("%Y-%m-%d"))
        d = d[d.date < SEAL_DATE]
        d.insert(0, "series", name)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(paths.artifact(STORE), index=False)
    return out


STORE_FULL = "macro/yahoo_market_daily_with_sealed.parquet"


def build_full_store(start: str = "2000-01-01") -> pd.DataFrame:
    """Every series to today, including the sealed period: only inside the opened sealed evaluation."""
    from .. import seal
    seal.require_open("Yahoo market series on/after the seal")
    frames = []
    for sym, name in YAHOO_SERIES.items():
        d = _download(sym, start, None)
        d.insert(0, "series", name)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(paths.artifact(STORE_FULL), index=False)
    return out


def load_full() -> pd.DataFrame:
    from .. import seal
    seal.require_open("Yahoo market series on/after the seal")
    d = pd.read_parquet(paths.ARTIFACTS / STORE_FULL)
    return d.pivot_table(index="date", columns="series", values="close").sort_index()


def load_dev() -> pd.DataFrame:
    d = pd.read_parquet(paths.ARTIFACTS / STORE)
    d = d[d.date < SEAL_DATE]
    return d.pivot_table(index="date", columns="series", values="close").sort_index()


def fetch_live(days: int = 800) -> pd.DataFrame:
    """Recent closes for live inference (wide: date x series)."""
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    frames = []
    for sym, name in YAHOO_SERIES.items():
        d = _download(sym, start, None)
        d.insert(0, "series", name)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out.pivot_table(index="date", columns="series", values="close").sort_index()

"""Split history for moving vendor prices and SEC share counts to the as-traded basis (audit 3).

Source order per ticker: the price vendor's own Yahoo split files (us_stocks_ohlcv/work/yf, then work/verify_fresh),
then Yahoo through yfinance for the 38 tickers without a vendor file (TSLA, META, V, ...), plus any S03 corporate-action
split the chosen source lacks. S03
alone covered only part of the universe (GOOG's 2022 20:1 split was absent, which made GOOG's as-traded price 20x too
low before July 2022). Ratios are new shares per old share (20.0 = 20-for-1; < 1 = reverse split).

Read-only use of the source projects; the combined table is cached at artifacts/fundamentals/splits.parquet.
"""
from __future__ import annotations

import glob

import pandas as pd

from .. import paths

CACHE = ("fundamentals", "splits.parquet")


def _vendor_file(ticker: str) -> pd.Series | None:
    for sub in ("yf", "verify_fresh"):
        p = paths.OHLCV_ROOT / "work" / sub / f"{ticker}.splits.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if d.empty:
                return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            d = d.rename(columns={"Date": "date"})
            idx = pd.to_datetime(d["date"]).dt.tz_localize(None) if getattr(d["date"].dt, "tz", None) else pd.to_datetime(d["date"])
            return pd.Series(d["ratio"].astype(float).to_numpy(), index=pd.DatetimeIndex(idx).normalize())
    return None


def _s03() -> pd.DataFrame:
    p = glob.glob(str(paths.SEC_FULL500_RUN / "S03" / "*" / "corporate_actions.parquet"))
    if not p:
        return pd.DataFrame(columns=["ticker", "date", "value"])
    ca = pd.read_parquet(p[0])
    return ca[ca.type == "split"][["ticker", "date", "value"]]


def _yahoo(ticker: str) -> pd.Series | None:
    """Yahoo split history through yfinance (the vendor's source) for tickers without a vendor split file."""
    try:
        import yfinance as yf
        s = yf.Ticker(ticker.replace(".", "-")).splits
    except Exception:
        return None
    if s is None:
        return None
    di = pd.DatetimeIndex(s.index)
    di = di.tz_localize(None) if di.tz is not None else di
    return pd.Series(s.to_numpy(float), index=di.normalize())


def build(tickers: list[str]) -> pd.DataFrame:
    """Vendor file, else Yahoo (yfinance), else S03; S03 rows missing from the chosen source are added (none were
    found over 2009..2026-02 for tickers with both, audit 3)."""
    s03 = _s03()
    rows = []
    for t in tickers:
        v, src = _vendor_file(t), "vendor_yahoo"
        if v is None:
            v, src = _yahoo(t), "yahoo_yfinance"
        if v is None:
            v, src = pd.Series(dtype=float, index=pd.DatetimeIndex([])), "none"
        rows += [(t, d, float(r), src) for d, r in v.items() if r > 0 and r != 1.0]
        x = s03[s03.ticker == t]
        for d, r in zip(pd.to_datetime(x.date), x.value):
            if r > 0 and r != 1.0 and not ((abs((v.index - d).days) <= 5) & (abs(v.to_numpy(float) - r) <= 0.01 * r)).any():
                rows.append((t, d, float(r), "sec_s03"))
    out = pd.DataFrame(rows, columns=["ticker", "date", "value", "source"])
    out["date"] = pd.to_datetime(out.date).astype("datetime64[ns]")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def table() -> pd.DataFrame:
    p = paths.ARTIFACTS.joinpath(*CACHE)
    if p.exists():
        return pd.read_parquet(p)
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    t = build(sorted(cm.ticker.unique()))
    p.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(p, index=False)
    return t


def for_ticker(ticker: str) -> pd.Series:
    t = table()
    x = t[t.ticker == ticker]
    return pd.Series(x.value.to_numpy(float), index=pd.DatetimeIndex(x.date))

"""Fundamentals for one ticker, for the Trade Desk: statements, ratios, multiples, universe percentiles and a DCF.

historical(ticker, asof): the development store (facts filed before the seal) and the stored panel's percentiles.
live(ticker): current SEC facts (`sec_facts.load_live`) priced at the latest Yahoo close; percentiles are not shown
live (there is no live cross-section yet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from . import ratios as RT, sec_facts as SF, statements as ST


def cik_for(ticker: str) -> int | None:
    m = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    r = m[m.ticker == ticker]
    return int(r.cik.iloc[0]) if len(r) else None


def _splits(ticker: str) -> pd.Series:
    import glob
    p = glob.glob(str(paths.SEC_FULL500_RUN / "S03" / "*" / "corporate_actions.parquet"))
    if not p:
        return pd.Series(dtype=float)
    ca = pd.read_parquet(p[0])
    return ca[(ca.ticker == ticker) & (ca.type == "split")].set_index("date").value


def historical(ticker: str, asof) -> dict | None:
    cik = cik_for(ticker)
    snap_p = paths.ARTIFACTS / "fundamentals" / "snapshots" / f"CIK{int(cik or 0):010d}.parquet"
    panel_p = paths.ARTIFACTS / "fundamentals" / "panel.parquet"
    if cik is None or not snap_p.exists():
        return None
    s = pd.read_parquet(snap_p)
    s = s[s.snap_date <= pd.Timestamp(asof)]
    if s.empty:
        return None
    row = None
    if panel_p.exists():
        import pyarrow.parquet as pq
        t = pq.read_table(panel_p, filters=[("ticker", "=", ticker), ("date", "<=", pd.Timestamp(asof).to_pydatetime())])
        d = t.to_pandas()
        row = d.sort_values("date").iloc[-1] if len(d) else None
    return {"snapshot": s.iloc[-1], "ratios": row, "source": "SEC XBRL, filed before the seal (development store)"}


def live(ticker: str) -> dict | None:
    from ..data import bars as B  # noqa: F401  (keeps the import graph identical to historical use)
    from ..macro.context import _yahoo_daily
    cik = cik_for(ticker)
    if cik is None:
        return None
    f = SF.load_live(cik)
    if f.empty:
        return None
    today = pd.Timestamp.now().normalize()
    s = ST.snapshots(f, at=[today])
    from ..inference.pipeline import yahoo_symbol
    bars = _yahoo_daily(yahoo_symbol(ticker), (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    if bars.empty:
        return None
    s["snap_date"] = bars.date.iloc[-1]
    d = RT.daily(s, bars.tail(1), pd.Series(dtype=float))
    return {"snapshot": s.iloc[-1], "ratios": d.iloc[-1] if len(d) else None,
            "source": f"SEC XBRL live (latest filing {f.filed.max().date()}), Yahoo close {bars.date.iloc[-1].date()}"}


def dcf_per_share(fcf: float, growth: float, r: float, g_term: float, years: int, debt: float, cash: float,
                  shares: float) -> float:
    if not all(np.isfinite([fcf, growth, r, g_term, shares])) or shares <= 0 or r <= g_term:
        return float("nan")
    pv = sum(fcf * (1 + growth) ** k / (1 + r) ** k for k in range(1, years + 1))
    tv = fcf * (1 + growth) ** years * (1 + g_term) / (r - g_term) / (1 + r) ** years
    return float((pv + tv - np.nan_to_num(debt) + np.nan_to_num(cash)) / shares)

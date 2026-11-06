"""Sealed readers for the ETF bars built by scripts/data/build_etf_bars.py (SPY, sector SPDRs, rates/credit/commodity ETFs).

Same contract as `bars.py`: every read pushes the seal into the parquet scan, so rows on/after 2023-01-03 never return.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pyarrow.parquet as pq

from .. import paths
from .. import seal as _seal
from ..seal import SEAL_DATE, SEAL_UTC  # noqa: F401

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]   # the nine SPDR sectors trading since 1998


def _p(tf: str, t: str):
    if _seal.is_open() and (paths.ARTIFACTS / "etf_sealed" / tf / f"{t}.parquet").exists():
        return paths.ARTIFACTS / "etf_sealed" / tf / f"{t}.parquet"      # built inside the opened evaluation only
    return paths.ARTIFACTS / "etf" / tf / f"{t}.parquet"


def available() -> list[str]:
    return sorted(p.stem for p in (paths.ARTIFACTS / "etf" / "1day").glob("*.parquet"))


def read_daily(t: str) -> pd.DataFrame:
    return _read_daily(t, _seal.read_limit_date())      # the cache is keyed on the read limit (never stale across the gate)


@lru_cache(maxsize=64)
def _read_daily(t: str, limit: pd.Timestamp) -> pd.DataFrame:
    d = pq.read_table(_p("1day", t), filters=[("date", "<", limit.to_pydatetime())]).to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.sort_values("date").reset_index(drop=True)


def read_hourly(t: str) -> pd.DataFrame:
    h = pq.read_table(_p("1hour", t), filters=[("timestamp", "<", _seal.read_limit_utc().to_pydatetime())]).to_pandas()
    return h.sort_values("timestamp").reset_index(drop=True)


def closes(tickers, field: str = "close") -> pd.DataFrame:
    """Wide date x ticker table of daily closes (sealed)."""
    return pd.concat({t: read_daily(t).set_index("date")[field] for t in tickers}, axis=1).sort_index()

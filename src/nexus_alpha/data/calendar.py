"""XNYS session calendar.

Sessions 2001-01-02 .. 2026-02-27 are the TriFrame data store's market_calendar.csv (vendored as
resources/xnys_sessions.csv, byte-identical, sha256 54f50bd6...; verified there against the NYSE calendar: 0 missing,
0 extra sessions in 979 stocks). Later sessions to 2027-12-31 come from pandas_market_calendars, which reproduces the
vendored file exactly on the overlap (scripts/data/extend_calendar.py).

The calendar is public schedule information, so using session dates after the seal to locate a label window's end is not
a look at sealed market data.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
CALENDAR_CSV = Path(__file__).resolve().parents[1] / "resources" / "xnys_sessions_extended.csv"


@lru_cache(maxsize=1)
def calendar() -> pd.DataFrame:
    c = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    c = c.sort_values("date").reset_index(drop=True)
    if c.date.duplicated().any():
        raise ValueError("duplicate sessions in market calendar")
    c["open_et"] = c.date + pd.Timedelta(hours=9, minutes=30)
    c["close_et"] = c.date + pd.to_timedelta(np.where(c.early_close, 13 * 60, 16 * 60), unit="m")
    return c


@lru_cache(maxsize=1)
def sessions() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(calendar().date.values)


def session_index(dates) -> np.ndarray:
    """Position of each date in the session list; -1 if the date is not a session."""
    s = sessions()
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    i = s.searchsorted(d)
    ok = (i < len(s)) & (s[np.minimum(i, len(s) - 1)] == d)
    return np.where(ok, i, -1)


def offset_session(dates, h: int) -> pd.DatetimeIndex:
    """The h-th session after each date (h >= 0). Returns NaT beyond the calendar."""
    s = sessions()
    i = session_index(dates)
    if (i < 0).any():
        raise ValueError("offset_session: some dates are not sessions")
    j = i + h
    out = np.full(len(j), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = j < len(s)
    out[ok] = s.values[j[ok]]
    return pd.DatetimeIndex(out)


def decision_time_utc(asof_sessions) -> pd.DatetimeIndex:
    """Decision timestamp for a sample whose last completed session is `asof`: 09:00 New York time on the next session
    (pre-open). Bars are complete through the close of `asof`; news and SEC filings count if available by then; the
    simulated entry is the official open (09:30) of that next session. See docs/POINT_IN_TIME_POLICY.md (D-003)."""
    nxt = offset_session(asof_sessions, 1)
    loc = (nxt + pd.Timedelta(hours=9)).tz_localize(TZ)
    return loc.tz_convert("UTC")

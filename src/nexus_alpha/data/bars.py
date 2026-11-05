"""Sealed readers for the frozen TriFrame bars and causal multi-timeframe views.

Bars are split-adjusted, not dividend-adjusted (TriFrame data contract 2.1). Daily `date` is the session date; hourly
`timestamp` is the UTC bar start of regular-session bars stamped 09:30 ... 15:30 New York time; weekly bars are built
here from daily bars so that the bar of a week in progress contains only the days up to the as-of session.

Every reader pushes the seal into the parquet scan (`filters=`), so rows on or after the seal are never returned.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import paths
from ..seal import SEAL_DATE, SEAL_UTC
from .calendar import offset_session

OHLCV = ["open", "high", "low", "close", "volume"]


def _path(tf: str, ticker: str) -> Path:
    return paths.OHLCV_FINAL / tf / f"{ticker}.parquet"


def available_tickers() -> list[str]:
    return sorted(p.stem for p in (paths.OHLCV_FINAL / "1day").glob("*.parquet"))


def read_daily(ticker: str) -> pd.DataFrame:
    t = pq.read_table(_path("1day", ticker), filters=[("date", "<", SEAL_DATE.to_pydatetime())])
    d = t.to_pandas().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d


def read_hourly(ticker: str) -> pd.DataFrame:
    t = pq.read_table(_path("1hour", ticker), filters=[("timestamp", "<", SEAL_UTC.to_pydatetime())])
    h = t.to_pandas().sort_values("timestamp").reset_index(drop=True)
    return h


def read_weekly_frozen(ticker: str) -> pd.DataFrame:
    """The frozen weekly file, sealed on the week's last trading day (used only to verify `weekly_from_daily`)."""
    t = pq.read_table(_path("1week", ticker), filters=[("last_day", "<", SEAL_DATE.to_pydatetime())])
    return t.to_pandas().sort_values("week_start").reset_index(drop=True)


def total_rows(tf: str, ticker: str) -> int:
    """Row count from the parquet footer (metadata only; no values are read)."""
    return pq.ParquetFile(_path(tf, ticker)).metadata.num_rows


def week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.normalize()


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Completed-or-current calendar weeks (Monday start) aggregated from daily bars."""
    d = daily.copy()
    d["week_start"] = week_start(d.date)
    g = d.groupby("week_start", sort=True)
    w = pd.DataFrame({"first_day": g.date.min(), "last_day": g.date.max(), "open": g.open.first(), "high": g.high.max(),
                      "low": g.low.min(), "close": g.close.last(), "volume": g.volume.sum(),
                      "trading_days": g.date.count()}).reset_index()
    return w


def ohlc_violations(df: pd.DataFrame) -> pd.Series:
    """Rows breaking the OHLC invariants (non-positive prices, high below max(open, close), low above min(open, close))."""
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    tol = 1e-9 * np.maximum(1.0, h.abs())
    return (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h + tol < np.maximum(o, c)) | (l - tol > np.minimum(o, c)) | (df["volume"] < 0)


class StockBars:
    """All sealed bars of one stock with causal as-of views. Everything returned for as-of index k uses daily bars
    dated <= date[k] and hourly bars that started on or before that session."""

    def __init__(self, ticker: str, daily: pd.DataFrame | None = None, hourly: pd.DataFrame | None = None):
        self.ticker = ticker
        self.d = read_daily(ticker) if daily is None else daily.sort_values("date").reset_index(drop=True)
        self.h = read_hourly(ticker) if hourly is None else hourly.sort_values("timestamp").reset_index(drop=True)
        self.dates = pd.DatetimeIndex(self.d.date.values)
        et = pd.DatetimeIndex(self.h.timestamp).tz_convert("America/New_York")
        self.h_session = et.tz_localize(None).normalize()
        ws = week_start(self.d.date).values
        self.wid = np.cumsum(np.r_[0, (ws[1:] != ws[:-1]).astype(int)])
        self.first = np.flatnonzero(np.r_[True, self.wid[1:] != self.wid[:-1]])
        o, h, l, c, v = (self.d[k].values.astype(float) for k in OHLCV)
        self._o, self._h, self._l, self._c, self._v = o, h, l, c, v
        self.Wo = o[self.first]
        self.Wc = c[np.r_[self.first[1:] - 1, len(o) - 1]]
        self.Wh = np.maximum.reduceat(h, self.first) if len(o) else h
        self.Wl = np.minimum.reduceat(l, self.first) if len(o) else l
        self.Wv = np.add.reduceat(v, self.first) if len(o) else v

    def k_of(self, date) -> int:
        i = self.dates.searchsorted(pd.Timestamp(date))
        if i >= len(self.dates) or self.dates[i] != pd.Timestamp(date):
            raise KeyError(f"{self.ticker}: {date} is not a bar date")
        return int(i)

    def daily_window(self, k: int, n: int) -> pd.DataFrame:
        return self.d.iloc[max(0, k - n + 1): k + 1]

    def weekly_window(self, k: int, n: int) -> pd.DataFrame:
        """Last n weekly bars as of daily index k; the final bar is the week in progress (days up to date[k] only)."""
        w = self.wid[k]
        f = self.first[w]
        s = max(0, w - (n - 1))
        out = pd.DataFrame({
            "week_start": week_start(self.d.date.iloc[self.first[s:w + 1]]).values,
            "open": np.r_[self.Wo[s:w], self._o[f]],
            "high": np.r_[self.Wh[s:w], self._h[f:k + 1].max()],
            "low": np.r_[self.Wl[s:w], self._l[f:k + 1].min()],
            "close": np.r_[self.Wc[s:w], self._c[k]],
            "volume": np.r_[self.Wv[s:w], self._v[f:k + 1].sum()],
        })
        out["partial"] = False
        # The current week is complete only if the next exchange session falls in a later week (calendar, not data).
        nxt = offset_session([self.dates[k]], 1)[0]
        out.loc[out.index[-1], "partial"] = bool(pd.notna(nxt) and week_start(pd.Series([nxt]))[0] == out.week_start.iloc[-1])
        return out

    def hourly_window(self, k: int, n: int) -> pd.DataFrame:
        e = int(np.searchsorted(self.h_session.values, self.dates.values[k], side="right"))
        return self.h.iloc[max(0, e - n): e]

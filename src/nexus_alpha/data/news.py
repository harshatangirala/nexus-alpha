"""News: ticker mapping, point-in-time availability and de-duplication for the financial-news-multisource corpus.

Availability policy (docs/POINT_IN_TIME_POLICY.md, D-004):
* minute-precision rows: available_at = published (UTC) + 15 minutes processing latency;
* a "minute" row stamped exactly 00:00:00 UTC is treated as day precision (a default time, not a publication time);
* day-precision rows (date known, time unknown): available_at = end of that calendar day in US Pacific time, i.e. the
  next calendar day 08:00 UTC. The corpus's own `date_trading` anchor (next NYSE open) is kept for reference only.
* An article counts for a sample only if available_at <= the sample's decision time (09:00 New York, next session).

Ticker mapping is conservative: only explicit ticker tags in `extra_fields.stocks` that match the current-ticker universe
after normalisation. No mapping from company names (master spec section 8). Renamed tickers (for example FB -> META)
are therefore missed; this is recorded as a limitation, never patched by guesswork.

The corpus licence is non-commercial research only with no redistribution of article text: text stays in artifacts/.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

MINUTE_LATENCY = pd.Timedelta(minutes=15)
DAY_AVAILABLE_OFFSET = pd.Timedelta(days=1, hours=8)          # end of the calendar day in US Pacific time, in UTC
TICKER_TAGGED_SUBSETS = ("fnspid_news", "benzinga_6000stocks", "sentarl_combined", "yahoo_finance_felixdrinkall",
                         "yahoo_finance_articles", "reddit_finance_sp500")

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z])?$")


def normalize_ticker(raw: str) -> str | None:
    if raw is None:
        return None
    t = str(raw).strip().upper().lstrip("$")
    t = t.replace("-", ".").replace("/", ".")
    if t.endswith(".US"):
        t = t[:-3]
    return t if _TICKER_RE.match(t) else None


def parse_extra(s: str | None) -> dict:
    if not s:
        return {}
    try:
        e = json.loads(s)
    except (ValueError, TypeError):
        return {"_unparseable": True}
    return e if isinstance(e, dict) else {}


def tickers_of(extra: dict) -> list[str]:
    st = extra.get("stocks")
    if st is None:
        return []
    if isinstance(st, str):
        st = re.split(r"[,\s;]+", st)
    out = []
    for x in st:
        n = normalize_ticker(x)
        if n and n not in out:
            out.append(n)
    return out


def availability(date_utc: pd.Series, precision: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (published_utc, available_utc, effective_precision) for vectors of corpus `date` strings/timestamps."""
    d = pd.to_datetime(date_utc, utc=True, errors="coerce")
    prec = precision.fillna("unknown").astype(str).values.copy()
    midnight = (d.dt.hour == 0) & (d.dt.minute == 0) & (d.dt.second == 0)
    eff = np.where((prec == "minute") & midnight.values, "day_from_midnight", prec)
    is_minute = eff == "minute"
    avail = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    avail[is_minute] = d[is_minute] + MINUTE_LATENCY
    day = d.dt.floor("D")
    avail[~is_minute] = day[~is_minute] + DAY_AVAILABLE_OFFSET
    return d, avail, pd.Series(eff, index=d.index)


def available_for(index: pd.DataFrame, ticker: str, decision_utc: pd.Timestamp, lookback: pd.Timedelta,
                  time_col: str = "available_utc") -> pd.DataFrame:
    """Articles for one ticker that were available at the decision time and inside the lookback window."""
    m = (index["ticker"] == ticker) & (index[time_col] <= decision_utc) & (index[time_col] > decision_utc - lookback)
    return index.loc[m]


# ------------------------------------------------------------------------------------------------ de-duplication
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", _NONWORD.sub(" ", (s or "").lower())).strip()


def text_hash(s: str) -> str:
    return hashlib.sha1(norm_text(s).encode()).hexdigest()


def headline(s: str) -> str:
    return (s or "").split("\n\n", 1)[0].strip()


def shingles(s: str, k: int = 3) -> set:
    w = norm_text(s).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))} if w else set()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def cluster_near_duplicates(headlines: Iterable[str], times: Iterable[pd.Timestamp],
                            threshold: float = 0.8, window: pd.Timedelta = pd.Timedelta(days=2)) -> np.ndarray:
    """Greedy clustering of near-duplicate headlines for ONE ticker, in time order. Returns a cluster id per article;
    the earliest article of a cluster is its representative (so a re-published story never looks newer than it is)."""
    h = list(headlines)
    t = pd.to_datetime(pd.Series(list(times)), utc=True)
    order = np.argsort(t.dt.tz_convert(None).to_numpy(), kind="stable")
    sh = [shingles(x) for x in h]
    cid = np.full(len(h), -1)
    reps: list[tuple[int, pd.Timestamp]] = []
    for i in order:
        best = -1
        for rid, rt in reversed(reps):
            if t.iloc[i] - rt > window:   # representatives are in time order: older ones are further away
                break
            if jaccard(sh[i], sh[rid]) >= threshold:
                best = cid[rid]
                break
        if best < 0:
            cid[i] = i
            reps.append((i, t.iloc[i]))
        else:
            cid[i] = best
    return cid

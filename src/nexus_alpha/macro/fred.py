"""US macro releases as they were known at the time: FRED/ALFRED real-time vintages.

Needs FRED_API_KEY in the environment (free key from the St. Louis Fed; set by the owner, never stored in the repo).
For every series all vintages are downloaded (realtime_start/realtime_end per observation). The value of observation
date o known at as-of session t is the vintage with realtime_start <= t < realtime_end.

Availability rule (pre-registered): a vintage released on day R is used only for as-of sessions t >= R, i.e. for
decisions at 09:00 New York on t+1 >= R+1. This is at least one day after any release time (most BLS/BEA releases are
08:30 on R). Dates before a series' first archived vintage get NaN (missing), never a later vintage.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .. import paths
from ..seal import SEAL_DATE

API = "https://api.stlouisfed.org/fred/series/observations"
# id -> (short name, transform). Transforms are applied within one vintage (so no revised data leak into a change).
SERIES = {
    "CPIAUCSL": ("cpi", "yoy12"), "CPILFESL": ("core_cpi", "yoy12"), "PCEPILFE": ("core_pce", "yoy12"),
    "PAYEMS": ("payrolls", "diff1_3mavg"), "UNRATE": ("unrate", "level_and_sahm"), "ICSA": ("claims", "log_4w_vs_52w"),
    "INDPRO": ("indpro", "yoy12"), "RSAFS": ("retail", "yoy12"), "HOUST": ("housing_starts", "yoy12"),
    "UMCSENT": ("umich_sentiment", "level_and_diff12"), "GDPC1": ("real_gdp", "qoq_saar"), "M2SL": ("m2", "yoy12"),
    "NFCI": ("nfci", "level_and_diff4"), "FEDFUNDS": ("fedfunds", "level_and_diff12"),
}
STORE = "macro/alfred_vintages.parquet"


def have_key() -> bool:
    return bool(os.environ.get("FRED_API_KEY"))


def _get(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, "api_key": os.environ["FRED_API_KEY"], "file_type": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "nexus-alpha-research"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"ALFRED request failed for {params.get('series_id')}")


def fetch_vintages(series_id: str, obs_start: str = "1990-01-01") -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        j = _get({"series_id": series_id, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                  "observation_start": obs_start, "limit": 100000, "offset": offset})
        obs = j.get("observations", [])
        rows += obs
        if len(obs) < 100000:
            break
        offset += 100000
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["value"] = pd.to_numeric(d["value"].replace(".", np.nan), errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    d["realtime_start"] = pd.to_datetime(d["realtime_start"])
    d["realtime_end"] = pd.to_datetime(d["realtime_end"].replace("9999-12-31", "2262-04-11"))
    d.insert(0, "series_id", series_id)
    return d[["series_id", "date", "realtime_start", "realtime_end", "value"]]


def build_store() -> pd.DataFrame:
    """Download all vintages; vintages released on/after the seal are dropped (they are post-seal information)."""
    if not have_key():
        raise RuntimeError("FRED_API_KEY is not set: ask the owner to set it (never paste it into a file or chat).")
    frames = [fetch_vintages(s) for s in SERIES]
    d = pd.concat(frames, ignore_index=True)
    d = d[d.realtime_start < SEAL_DATE]
    d.loc[d.realtime_end > SEAL_DATE, "realtime_end"] = pd.Timestamp("2262-04-11")
    d.to_parquet(paths.artifact(STORE), index=False)
    return d


def _transform(s: pd.Series, how: str) -> dict[str, float]:
    """s: one vintage, indexed by observation date (ascending). Returns features from its latest observations."""
    s = s.dropna()
    if len(s) < 2:
        return {}
    v = s.to_numpy()
    if how == "yoy12":
        return {"yoy": float(np.log(v[-1] / v[-13]) * 100) if len(v) > 12 else np.nan,
                "yoy_change3": float(np.log(v[-1] / v[-13]) * 100 - np.log(v[-4] / v[-16]) * 100) if len(v) > 15 else np.nan}
    if how == "diff1_3mavg":
        return {"chg3m_avg": float(np.mean(np.diff(v[-4:]))) if len(v) > 3 else np.nan}
    if how == "level_and_sahm":
        lvl = float(v[-1])
        sahm = float(pd.Series(v).rolling(3).mean().iloc[-1] - pd.Series(v).rolling(3).mean().iloc[-13:].min()) if len(v) > 14 else np.nan
        return {"level": lvl, "sahm": sahm}
    if how == "log_4w_vs_52w":
        return {"log4w_vs_52w": float(np.log(np.mean(v[-4:]) / np.mean(v[-52:]))) if len(v) > 52 else np.nan}
    if how == "level_and_diff12":
        return {"level": float(v[-1]), "diff12": float(v[-1] - v[-13]) if len(v) > 12 else np.nan}
    if how == "level_and_diff4":
        return {"level": float(v[-1]), "diff4": float(v[-1] - v[-5]) if len(v) > 4 else np.nan}
    if how == "qoq_saar":
        return {"qoq_saar": float(((v[-1] / v[-2]) ** 4 - 1) * 100)}
    raise ValueError(how)


def asof_features(dates: pd.DatetimeIndex, store: pd.DataFrame | None = None) -> pd.DataFrame:
    """Real-time macro features for each as-of session (vintage known at t), plus staleness in days."""
    st = pd.read_parquet(paths.ARTIFACTS / STORE) if store is None else store
    dates = pd.DatetimeIndex(dates).sort_values()
    out = pd.DataFrame(index=dates)
    for sid, g in st.groupby("series_id"):
        name, how = SERIES[sid]
        vint = np.sort(g.realtime_start.unique())
        # the vintage in force at t is the last realtime_start <= t; recompute only when it changes
        pos = np.searchsorted(vint, dates.values, side="right") - 1
        cache = {}
        recs = []
        for t, p in zip(dates, pos):
            if p < 0:
                recs.append({})
                continue
            vs = vint[p]
            if vs not in cache:
                gg = g[(g.realtime_start <= vs) & (g.realtime_end > vs)].sort_values("date")
                f = _transform(gg.set_index("date").value, how)
                cache[vs] = (f, gg.date.max() if len(gg) else pd.NaT)
            f, last_obs = cache[vs]
            r = dict(f)
            r["obs_age_days"] = float((t - last_obs).days) if pd.notna(last_obs) else np.nan
            recs.append(r)
        df = pd.DataFrame(recs, index=dates)
        df.columns = [f"fred_{name}_{c}" for c in df.columns]
        out = out.join(df)
    return out

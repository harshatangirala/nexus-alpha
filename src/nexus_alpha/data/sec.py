"""SEC filings as a point-in-time modality, reusing the SEC study's 500-firm event table (read-only).

Availability (docs/POINT_IN_TIME_POLICY.md, D-007): the SEC study's decision_time = acceptance + 5 min (assumed public)
+ 30 min (processing). NEXUS uses it as available_at, except for timestamp quality C (derived or uncertain), where
available_at = max(decision_time, filing date 18:05 New York): EDGAR gives a filing accepted after 17:30 the next
business day's date, so a filing dated D was public by 17:30 on D; the bound can only make availability later.

Coverage is explicit: only the SEC study's 500 firms have events, from 2000 to 2026-07-31. A sample outside that coverage
gets sec_covered = 0 and NaN SEC features ("not observed"), never zeros ("no filings").
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..seal import SEAL_UTC

ITEMS = {"2.02": "results", "5.02": "officers", "1.01": "material_agreement", "8.01": "other_events",
         "7.01": "reg_fd", "2.01": "acquisition_disposition", "2.06": "impairment", "4.02": "non_reliance",
         "5.07": "shareholder_vote", "1.02": "agreement_termination"}
SEC_FEATURES = (["sec_covered", "sec_n_8k_7d", "sec_n_8k_30d", "sec_days_since_8k", "sec_days_since_10q",
                 "sec_days_since_10k", "sec_earnings_3d", "sec_last_post_close"]
                + [f"sec_item_{v}_7d" for v in ITEMS.values()])
COVERAGE_END = pd.Timestamp("2026-07-31", tz="America/New_York")


def _stage(stage: str, name: str):
    hits = list((paths.SEC_FULL500_RUN / stage).glob(f"*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{stage}/{name}")
    return hits[0]


def load_events(dev_only: bool = True) -> pd.DataFrame:
    cols = ["event_id", "ticker", "cik", "event_type", "forms", "items", "acceptance_time", "decision_time",
            "ts_quality", "filing_date", "session_bucket", "year", "accessions"]
    ev = pd.read_parquet(_stage("S07", "events.parquet"), columns=cols)
    fd = pd.to_datetime(ev.filing_date, errors="coerce").dt.tz_localize("America/New_York")
    bound = fd + pd.Timedelta(hours=18, minutes=5)
    avail = ev.decision_time.copy()
    c = ev.ts_quality.eq("C")
    avail[c] = pd.concat([ev.decision_time[c], bound[c]], axis=1).max(axis=1)
    avail = avail.fillna(bound)
    ev["available_utc"] = avail.dt.tz_convert("UTC")
    ev["item_list"] = ev["items"].fillna("[]").map(lambda s: json.loads(s) if s.startswith("[") else s.split("|"))
    if dev_only:
        ev = ev[ev.available_utc < SEAL_UTC]
    return ev.sort_values(["ticker", "available_utc"]).reset_index(drop=True)


def sec_features(samples: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Structured filing features per sample from events available at its decision time."""
    covered = set(ev.ticker.unique())
    first_cov = ev.groupby("ticker").available_utc.min()
    rows = []
    for t, g in samples.groupby("ticker"):
        dts = pd.to_datetime(g.decision_utc, utc=True).to_numpy(dtype="datetime64[ns]")
        e = ev[ev.ticker == t]
        if t not in covered or e.empty:
            for sid in g.sample_id:
                rows.append({"sample_id": sid, "sec_covered": 0.0})
            continue
        at = e.available_utc.to_numpy(dtype="datetime64[ns]")
        et = e.event_type.to_numpy()
        items = e.item_list.to_numpy()
        post = (e.session_bucket == "POST").to_numpy()
        for sid, d in zip(g.sample_id.to_numpy(), dts):
            r = {"sample_id": sid}
            if d < first_cov[t].to_datetime64() or d > COVERAGE_END.tz_convert("UTC").to_datetime64():
                r["sec_covered"] = 0.0
                rows.append(r)
                continue
            r["sec_covered"] = 1.0
            hi = np.searchsorted(at, d, side="right")
            lo7 = np.searchsorted(at, d - np.timedelta64(7, "D"), side="right")
            lo30 = np.searchsorted(at, d - np.timedelta64(30, "D"), side="right")
            lo3 = np.searchsorted(at, d - np.timedelta64(3, "D"), side="right")
            k7 = slice(lo7, hi)
            r["sec_n_8k_7d"] = float((et[k7] == "8K").sum())
            r["sec_n_8k_30d"] = float((et[lo30:hi] == "8K").sum())
            for typ, name in (("8K", "8k"), ("periodic_10Q", "10q"), ("periodic_10K", "10k")):
                ix = np.flatnonzero(et[:hi] == typ)
                r[f"sec_days_since_{name}"] = float((d - at[ix[-1]]) / np.timedelta64(1, "D")) if len(ix) else np.nan
            it7 = [x for lst in items[k7] for x in (lst or [])]
            for code, name in ITEMS.items():
                r[f"sec_item_{name}_7d"] = float(code in it7)
            r["sec_earnings_3d"] = float(any("2.02" in (lst or []) for lst in items[lo3:hi]))
            r["sec_last_post_close"] = float(post[hi - 1]) if hi > lo7 else np.nan
            rows.append(r)
    out = pd.DataFrame(rows).set_index("sample_id")
    return out.reindex(columns=SEC_FEATURES)


def latest_event_ids(samples: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Most recent 8-K (30 days) and periodic filing (120 days) available at each sample's decision time."""
    rows = []
    for t, g in samples.groupby("ticker"):
        e = ev[ev.ticker == t]
        if e.empty:
            continue
        at = e.available_utc.to_numpy(dtype="datetime64[ns]")
        et = e.event_type.to_numpy()
        ids = e.event_id.to_numpy()
        for sid, d in zip(g.sample_id.to_numpy(), pd.to_datetime(g.decision_utc, utc=True).to_numpy(dtype="datetime64[ns]")):
            hi = np.searchsorted(at, d, side="right")
            rec = {"sample_id": sid}
            for kind, days, mask in (("8k", 30, et[:hi] == "8K"), ("periodic", 120, np.isin(et[:hi], ["periodic_10Q", "periodic_10K"]))):
                ix = np.flatnonzero(mask)
                if len(ix) and (d - at[ix[-1]]) <= np.timedelta64(days, "D"):
                    rec[f"event_{kind}"] = ids[ix[-1]]
            rows.append(rec)
    return pd.DataFrame(rows)

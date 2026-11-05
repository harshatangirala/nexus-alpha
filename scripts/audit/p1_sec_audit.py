"""Phase 1: audit of the SEC study's 500-firm event table for reuse (point-in-time contract, coverage, universe overlap).

Reads the CPU-part outputs of the SEC study (sec_inputs_staging/full, read-only). Event text is not printed or copied.
Output: data_manifests/sec_audit.json
Run: .venv/Scripts/python -I scripts/audit/p1_sec_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nexus_alpha import paths
from nexus_alpha.cv import store as ST
from nexus_alpha.data import bars as B

REPO = Path(__file__).resolve().parents[2]


def stage_file(stage: str, name: str) -> Path:
    hits = list((paths.SEC_FULL500_RUN / stage).glob(f"*/{name}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{stage}/{name}: {hits}")
    return hits[0]


def main() -> int:
    ev = pd.read_parquet(stage_file("S07", "events.parquet"))
    out = {"audited_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": str(paths.SEC_FULL500_RUN)}
    out["n_events"] = len(ev)
    out["n_firms_cik"] = int(ev.cik.nunique())
    out["n_tickers"] = int(ev.ticker.nunique())
    out["event_type_counts"] = ev.event_type.value_counts().to_dict()
    out["forms_top"] = ev.forms.value_counts().head(10).to_dict()
    out["ts_quality"] = ev.ts_quality.value_counts().to_dict()
    out["public_time_source"] = ev.public_time_source.value_counts().to_dict()
    out["timing_policy"] = ev.timing_policy.iloc[0]
    out["events_by_year"] = {int(k): int(v) for k, v in ev.year.value_counts().sort_index().items()}
    lag = (ev.decision_time - ev.acceptance_time).dt.total_seconds() / 60
    out["decision_minus_acceptance_minutes"] = lag.describe().round(2).to_dict()
    out["session_bucket"] = ev.session_bucket.value_counts().to_dict()
    cats = ev.categories.dropna().str.split("|").explode()
    out["categories_top"] = cats.value_counts().head(25).to_dict()
    items = ev["items"].dropna().str.split("|").explode()
    out["items_8k_top"] = items.value_counts().head(25).to_dict()
    universe = set(B.available_tickers())
    s500 = set(ST.sample_index().ticker.unique())
    sec = set(ev.ticker.dropna().unique())
    out["sec_tickers_in_ohlcv_universe"] = len(sec & universe)
    out["sec_tickers_not_in_universe"] = sorted(sec - universe)[:30]
    out["overlap_sec_and_triframe_s500"] = len(sec & s500)
    out["triframe_s500_tickers"] = len(s500)
    mini = pd.read_parquet(next((paths.SEC_MINI_RUN / "S07").glob("events.parquet")) if (paths.SEC_MINI_RUN / "S07" / "events.parquet").exists()
                           else next((paths.SEC_MINI_RUN / "S07").glob("*/events.parquet")), columns=["ticker"])
    out["sec_mini150_tickers"] = int(mini.ticker.nunique())
    out["overlap_mini150_and_s500"] = len(set(mini.ticker.unique()) & s500)
    txt = pq.ParquetFile(stage_file("S09", "event_text.parquet"))
    out["event_text_rows"] = txt.metadata.num_rows
    lens = pq.read_table(stage_file("S09", "event_text.parquet"), columns=["text"]).column("text").to_pandas().str.len()
    out["event_text_chars"] = lens.describe(percentiles=[.1, .5, .9, .99]).round(0).to_dict()
    (REPO / "data_manifests" / "sec_audit.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("events_by_year", "categories_top", "items_8k_top")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

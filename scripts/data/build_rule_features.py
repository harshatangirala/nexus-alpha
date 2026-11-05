"""Evaluate the executable rule detectors for every s500 sample (causal, sealed bars).

Outputs: artifacts/features/rule_features.parquet (one-hot rule features, indexed by sample_id) and
artifacts/features/rule_events.parquet (regime, setups and triggers as JSON, ATR, close, daily index k) for the
decision engine and backtests.
Run: .venv/Scripts/python -I scripts/data/build_rule_features.py
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time

import pandas as pd

from nexus_alpha import paths


def one(args):
    from nexus_alpha.rules.batch import evaluate_ticker
    t, dates = args
    return evaluate_ticker(t, dates)


def main() -> int:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "asof_date"])
    jobs = list(fr.groupby("ticker").asof_date.apply(list).items())
    parts = []
    with cf.ProcessPoolExecutor(max_workers=5) as ex:
        for i, p in enumerate(ex.map(one, jobs, chunksize=4)):
            parts.append(p)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} tickers, {time.time() - t0:.0f} s", flush=True)
    ev = pd.concat(parts, ignore_index=True)
    ev["sample_id"] = ev.ticker + ":" + ev.asof_date.dt.strftime("%Y-%m-%d")
    out = paths.ARTIFACTS / "features"
    out.mkdir(parents=True, exist_ok=True)
    rule_cols = [c for c in ev.columns if c.startswith("rule_")]
    ev.set_index("sample_id")[rule_cols].to_parquet(out / "rule_features.parquet")
    ev.drop(columns=rule_cols).to_parquet(out / "rule_events.parquet", index=False)
    print(f"{len(ev):,} samples evaluated in {time.time() - t0:.0f} s")
    print(ev.regime.value_counts(normalize=True).round(3).to_string())
    print(ev[[c for c in rule_cols if c.startswith("rule_setup") or c.startswith("rule_trigger")]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

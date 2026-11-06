"""Build the bar-derived evaluation frame and chart embeddings for a period (rehearsal or, after the owner's unlock,
the sealed period). Output: artifacts/sealed/<tag>/{frame.parquet, emb_W.npy, emb_D.npy, emb_H.npy, samples.parquet}.

    python scripts/sealed/build_period.py --rehearsal 2022            # the development samples of 2022 (parity test)
    python scripts/sealed/build_period.py --schedule-check 2022       # does the continuation rule reproduce 2022 dates?
The sealed build is called from scripts/sealed/run_sealed_eval.py after seal.open_for_evaluation().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.sealed import frame as SF  # noqa: E402


def last_dev_asof(before: str) -> dict:
    from nexus_alpha.cv import store as ST
    idx = ST.sample_index()
    idx = idx[pd.to_datetime(idx.asof_date) < pd.Timestamp(before)]
    return idx.groupby("ticker").asof_date.max().to_dict()


def save(tag: str, samples: pd.DataFrame, fr: pd.DataFrame, E: dict, meta: dict, events: pd.DataFrame | None = None) -> Path:
    out = paths.ARTIFACTS / "sealed" / tag
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / "samples.parquet", index=False)
    fr.to_parquet(out / "frame.parquet", index=False)
    if events is not None:
        events.to_parquet(out / "events.parquet", index=False)
    for v, e in E.items():
        np.save(out / f"emb_{v}.npy", e)
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", type=int)
    ap.add_argument("--schedule-check", type=int)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    if a.schedule_check:
        y = a.schedule_check
        from nexus_alpha.cv import store as ST
        idx = ST.sample_index()
        dev = idx[pd.to_datetime(idx.asof_date).dt.year == y]
        s = SF.continue_schedule(last_dev_asof(f"{y}-01-01"), f"{y}-01-01", str(pd.to_datetime(dev.asof_date).max().date()))
        hit = s.sample_id.isin(set(dev.sample_id))
        print(f"{y}: development samples {len(dev):,}; continuation produced {len(s):,}; reproduced {int(hit.sum()):,} "
              f"({hit.sum() / len(dev):.1%} of development samples)")
        return
    if a.rehearsal:
        y = a.rehearsal
        dev = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "asof_date"])
        samples = dev[dev.asof_date.dt.year == y].reset_index(drop=True)
        fr, E, ev = SF.build(samples, workers=a.workers, log=lambda m: print(m, flush=True))
        out = save(f"rehearsal_{y}", samples, fr, E, {"mode": "rehearsal", "year": y, "rows": len(fr),
                                                         "elapsed_s": round(time.time() - t0, 1)}, ev)
        print(f"saved {len(fr):,} rows to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

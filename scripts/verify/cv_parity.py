"""Verify the reused CV contract end to end (master spec 6.1: verify checkpoint loading and preprocessing).

1. Renderer parity: NEXUS's vendored r1 renderer on sealed bars vs TriFrame's cached training images (ImageStore, sealed).
2. Embedding parity: NEXUS's embedder on those images vs the stored float16 s500 embeddings, per backbone.
3. Read-only check: TriFrame's scripts folder is unchanged (no __pycache__ written by the bridge).

Output: reports/verification/cv_parity.json
Run: .venv/Scripts/python -I scripts/verify/cv_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nexus_alpha import paths, sources_bridge
from nexus_alpha.cv import backbones as BB
from nexus_alpha.cv import render as R
from nexus_alpha.cv import store as ST

REPO = Path(__file__).resolve().parents[2]
BACKBONES = ["convnext_tiny", "resnet18", "vit_small_patch14_dinov2", "ft_convnext_tiny", "resnet18_random_s0"]


def listing(p: Path) -> list[str]:
    return sorted(str(x.relative_to(p)) for x in p.rglob("*"))


def main() -> int:
    t0 = time.time()
    src_scripts = paths.OHLCV_ROOT / "scripts"
    before = listing(src_scripts)
    idx = ST.sample_index()
    rng = np.random.default_rng(20261105)
    picks = []
    for y in (2003, 2007, 2009, 2012, 2015, 2017, 2019, 2020, 2021, 2022):
        sub = idx[idx.asof_date.dt.year == y]
        picks += list(sub.iloc[rng.choice(len(sub), 2, replace=False)].index)
    sel = idx.loc[picks].reset_index(drop=True)
    store = sources_bridge.triframe_image_store(who="nexus_cv_parity")
    rec = {"samples": [], "renderer": {}, "embeddings": {}}
    mine = []
    n_exact = 0
    for _, r in sel.iterrows():
        sb = R.stock_bars_for_render(r.ticker)
        k = sb.k_of(r.asof_date)
        t = R.triplet_at(sb, k)
        ref = store.triplet(r.sample_id)
        eq = {v: bool(np.array_equal(t[v], ref[v])) for v in ("W", "D", "H")}
        diffpix = {v: int((t[v] != ref[v]).sum()) for v in ("W", "D", "H")}
        n_exact += all(eq.values())
        rec["samples"].append({"sample_id": r.sample_id, "exact": eq, "differing_pixels": diffpix,
                               "n_valid": t["n_valid"]})
        mine.append(t)
    rec["renderer"] = {"n": len(sel), "all_three_views_exact": n_exact}
    print(f"renderer: {n_exact}/{len(sel)} triplets byte-exact", flush=True)
    for b in BACKBONES:
        stats = []
        for v in ("W", "D", "H"):
            ref = np.asarray(ST.view_matrix(b, v)[sel.row.values], dtype=np.float32)
            got = BB.embed(b, [m[v] for m in mine])
            cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
            rel = np.abs(got - ref).max(1) / np.abs(ref).max(1)
            stats.append({"view": v, "cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
                          "max_rel_err": float(rel.max()), "max_abs_err": float(np.abs(got - ref).max())})
        rec["embeddings"][b] = stats
        print(b, [(s["view"], round(s["cos_min"], 6), round(s["max_rel_err"], 4)) for s in stats], flush=True)
    after = listing(src_scripts)
    rec["source_scripts_unchanged"] = before == after
    rec["elapsed_s"] = round(time.time() - t0, 1)
    out = REPO / "reports" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_parity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("source scripts folder unchanged:", rec["source_scripts_unchanged"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

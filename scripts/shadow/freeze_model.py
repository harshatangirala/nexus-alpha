"""Freeze a deployed bundle for the forward shadow test and the joint sealed evaluation.

    python scripts/shadow/freeze_model.py --version v1.1 --bundle 20261106T020411_e08_final_model_5c4327

Writes registry/frozen_models/<version>.json: SHA-256 of every bundle file, the code commit, label rule, horizons and the
class prior of the bundle's training blocks (the baseline the forward test scores against). Refuses to overwrite an
existing record: a frozen model is never re-frozen under the same version name.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.targets import HORIZONS, LABEL_VERSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--train-blocks", nargs="+", default=["train", "inner", "valid"])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = SH.FROZEN_DIR / f"{a.version}.json"
    if out.exists():
        sys.exit(f"{out} exists: frozen records are never overwritten")
    bdir = paths.ARTIFACTS / "bundles" / a.bundle
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    cols = ["block"] + [f"y_{h}" for h in HORIZONS]
    fr = pq.read_table(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=cols).to_pandas()
    fr = fr[fr.block.isin(a.train_blocks)]
    prior = {}
    for h in HORIZONS:
        y = fr[f"y_{h}"].dropna().astype(int).to_numpy()
        y = y[y >= 0]
        prior[h] = [float((y == c).mean()) for c in range(3)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rec = SH.freeze_record(a.version, bdir, {
        "frozen_utc": SH.now_utc().isoformat(timespec="seconds"),
        "code_commit_at_freeze": commit,
        "recipe": meta.get("recipe"), "trained_through_block": meta.get("trained_through_block"),
        "label_version": LABEL_VERSION, "horizons": HORIZONS,
        "decision_rule": "09:00 New York on session t+1; entry at the t+1 open; exit at the t+h close",
        "train_class_prior_blocks": a.train_blocks, "train_class_prior": prior,
        "note": a.note})
    SH.FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"froze {a.version}: {len(rec['files'])} files; record sha256 {SH.sha256_file(out)}")
    print("prior:", {h: [round(x, 3) for x in p] for h, p in prior.items()})


if __name__ == "__main__":
    main()

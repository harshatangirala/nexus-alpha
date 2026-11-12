"""E28 input: the SEC price-impact expert's per-sample evidence for the s500 frame (pre-registered, D-053).

Model and rules: `nexus_alpha.nlp.sec_impact`. Outputs:
* artifacts/features/sec_impact_features.parquet (index sample_id): the `sec_impact` fusion block;
* artifacts/features/sec_impact/event_predictions.parquet: per-event probabilities (out of fold before 2016);
* artifacts/features/sec_impact/event_labels.parquet: the h1/h5 labels read (exits before 2019 only), for the dashboard;
* artifacts/features/sec_impact/meta.json: tree counts, event counts and the expert's AUCs (OOF 2003-15, valid 2017-18).

Run: .venv/Scripts/python -I scripts/data/build_sec_impact_features.py [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.nlp import sec_impact as SI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    t0 = time.time()
    tab, cols = SI.load_table()
    print(f"{len(tab):,} development events, {len(cols)} features; labelled h1 {int(tab.y_h1.notna().sum()):,}, "
          f"h5 {int(tab.y_h5.notna().sum()):,} (labels read only for exits before {SI.LABEL_READ_BEFORE.date()})", flush=True)
    pred, info = SI.fit_predict(tab, cols, threads=a.threads)
    diag = SI.diagnostics(tab, pred)
    print(json.dumps(diag, indent=1), flush=True)
    out_dir = paths.ARTIFACTS / "features" / "sec_impact"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.join(tab[["ticker", "event_type", "available_utc"]]).to_parquet(out_dir / "event_predictions.parquet")
    tab[[f"y_h{h}" for h in SI.HORIZONS]].to_parquet(out_dir / "event_labels.parquet")   # exits before 2019 only
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "decision_utc"])
    feats = SI.sample_features(fr, tab, pred)
    feats.to_parquet(paths.ARTIFACTS / "features" / "sec_impact_features.parquet")
    meta = {"features": cols, "n_features": len(cols), "params": SI.PARAMS, "folds": SI.FOLDS, "fit": info,
            "diagnostics": diag, "present_share": float(feats.si_present.mean()), "events": len(tab),
            "label_read_before": str(SI.LABEL_READ_BEFORE.date()), "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"sec_impact block present for {feats.si_present.mean():.3f} of samples ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

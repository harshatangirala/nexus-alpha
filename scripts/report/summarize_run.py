"""Compact per-horizon table for one experiment run's metrics.json.

Run: .venv/Scripts/python -I scripts/report/summarize_run.py <run_id or path> [block] [models,comma,separated]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    run = sys.argv[1]
    block = sys.argv[2] if len(sys.argv) > 2 else "test"
    models = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    p = Path(run) if Path(run).exists() else REPO / "experiments" / run / "metrics.json"
    if p.is_dir():
        p = p / "metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for h, blocks in m.items():
        if block not in blocks:
            continue
        for name, r in blocks[block].items():
            if models and name not in models:
                continue
            d = r.get("dlogloss_vs_prior") or r.get("dlogloss_vs_ref") or {}
            rows.append({"h": h, "model": name, "n": r["n"], "logloss": r["log_loss"],
                         "dLL": d.get("estimate"), "dLL_lo": (d.get("ci95") or [None, None])[0],
                         "dLL_hi": (d.get("ci95") or [None, None])[1], "auc_macro": r.get("auroc_macro"),
                         "auc_up_dn": r.get("auroc_up_vs_down"), "auc_side": r.get("auroc_SIDEWAYS"),
                         "bal_acc": r["balanced_accuracy"], "macro_f1": r["macro_f1"], "ece": r["ece_top"]})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

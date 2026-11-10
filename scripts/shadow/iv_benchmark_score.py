"""E26 forward benchmark scorer: does NEXUS's P(move) add to option-implied volatility? (docs/V3_RESEARCH.md, E26)

    python scripts/shadow/iv_benchmark_score.py

Joins each model's scored forward rows (shadow/<version>/scores.csv: the realized return, the label rule's tau and
class) with the private IV snapshot of the same session. Only on-time snapshots are used, and only if the private
file's SHA-256 matches the public record. Writes shadow/iv_snapshot/benchmark.json with counts and log-loss statistics
only, no option data. The decision needs at least 120 scored sessions at 20D; before that it reports progress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.evaluation import iv_benchmark as IB  # noqa: E402

MODELS = ["v1.1", "v2"]
HORIZONS = {"5D": 5, "20D": 20}


def snapshots() -> tuple[pd.DataFrame, list[str]]:
    rows, notes = [], []
    for pub in sorted((SH.SHADOW_ROOT / "iv_snapshot").glob("????-??-??.json")):
        rec = json.loads(pub.read_text(encoding="utf-8"))
        if rec.get("late"):
            notes.append(f"{rec['asof_session']}: late snapshot, not used")
            continue
        f = paths.ARTIFACTS / "iv_snapshots" / f"{rec['asof_session']}.parquet"
        if not f.exists() or SH.sha256_file(f) != rec["summary_sha256"]:
            notes.append(f"{rec['asof_session']}: private summary missing or changed, not used")
            continue
        rows.append(pd.read_parquet(f, columns=["asof_session", "ticker", "iv30"]))
    return (pd.concat(rows) if rows else pd.DataFrame(columns=["asof_session", "ticker", "iv30"])), notes


def main() -> None:
    probs = SH.verify_chain("iv_snapshot")
    iv, notes = snapshots()
    out = {"chain_intact": not probs, "snapshots_used": int(iv.asof_session.nunique()) if len(iv) else 0,
           "notes": notes, "results": {}}
    for v in MODELS:
        sc_f = SH.SHADOW_ROOT / v / "scores.csv"
        if not sc_f.exists() or not len(iv):
            continue
        sc = pd.read_csv(sc_f)
        sc = sc[~sc.late.astype(bool)]
        for h, n in HORIZONS.items():
            r = sc[sc.horizon == h].merge(iv, on=["asof_session", "ticker"], how="inner")
            out["results"][f"{v}|{h}"] = IB.evaluate(r, n)
    # D-051: the four endpoint-2 tests are decided together with Holm's correction, once each has 120 sessions
    ready = {k: r["endpoint2_ll_ivonly_minus_both"]["p_one_sided"] for k, r in out["results"].items()
             if "endpoint2_ll_ivonly_minus_both" in r and r["sessions"] >= IB.MIN_SESSIONS}
    if ready and len(ready) == len(out["results"]):
        rej = IB.holm(ready)
        for k in ready:
            out["results"][k]["decision"] = ("NEXUS adds to implied volatility (Holm-corrected, D-051)" if rej[k]
                                             else "NEXUS's volatility skill is already priced (Holm-corrected, D-051)")
    dst = SH.SHADOW_ROOT / "iv_snapshot" / "benchmark.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    if not out["results"]:
        print(f"iv benchmark: nothing to score yet ({out['snapshots_used']} on-time snapshots)")
    for k, v in out["results"].items():
        print(f"iv benchmark {k}: {v['rows']} rows, {v['sessions']} sessions; {v.get('decision')}")


if __name__ == "__main__":
    main()

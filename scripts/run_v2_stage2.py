"""Stage 2 of the v2 chain (runs beside scripts/run_v2_chain.py, stopping at the first failure).

1. wait until the v2 frame has been rebuilt by the chain (newer than --since), then E13 (sequence models; 4 threads);
2. wait until an E11 run has COMPLETED (E12 needs E11's carried feature set), then E12 (v2 fusion).
Output: artifacts/logs/v2_stage2.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = REPO / "artifacts" / "logs" / "v2_stage2.log"


def step(name, args):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {name}: {' '.join(args)}\n")
        f.flush()
        r = subprocess.run([PY, "-B", *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"==== {name} exit {r.returncode}\n")
    if r.returncode:
        sys.exit(f"{name} failed")


def e11_done(after: float) -> bool:
    for r in (REPO / "experiments").glob("*_e11_macro_ladder_*"):
        m = r / "manifest.json"
        if m.exists() and m.stat().st_mtime > after and json.loads(m.read_text(encoding="utf-8")).get("status") == "COMPLETED":
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, required=True, help="epoch seconds; outputs older than this are ignored")
    a = ap.parse_args()
    frame = REPO / "artifacts" / "samples" / "v2_frame.parquet"
    while not (frame.exists() and frame.stat().st_mtime > a.since and time.time() - frame.stat().st_mtime > 60):
        time.sleep(60)
    step("e13", ["scripts/experiments/e13_sequence.py", "--threads", "4"])
    while not e11_done(a.since):
        time.sleep(60)
    step("e12", ["scripts/experiments/e12_fusion_v2.py"])
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("stage2 complete\n")


if __name__ == "__main__":
    main()

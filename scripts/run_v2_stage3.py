"""Stage 3 of the v2 chain: when the chain (E11, E16) and stage 2 (E13, E12) report completion, run E14 (blend, refit,
calibration, v2 bundle). Output: artifacts/logs/v2_stage3.log."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "artifacts" / "logs"


def has(f, s):
    p = LOGS / f
    return p.exists() and s in p.read_text(encoding="utf-8", errors="ignore")


while not (has("v2_chain_driver.log", "v2 chain complete") and has("v2_stage2.log", "stage2 complete")):
    if has("v2_stage2.log", "exit 1") or (LOGS / "v2_chain_driver.err").exists() and "failed" in (LOGS / "v2_chain_driver.err").read_text(errors="ignore"):
        sys.exit("an earlier stage failed; E14 not started")
    time.sleep(60)
with open(LOGS / "v2_stage3.log", "a", encoding="utf-8") as f:
    f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} e14\n")
    f.flush()
    r = subprocess.run([sys.executable, "-B", "scripts/experiments/e14_blend_freeze_v2.py"], cwd=REPO, stdout=f,
                       stderr=subprocess.STDOUT)
    f.write(f"==== e14 exit {r.returncode}\n{'stage3 complete' if r.returncode == 0 else 'stage3 failed'}\n")

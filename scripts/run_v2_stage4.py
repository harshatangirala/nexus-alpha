"""Stage 4: after E14 (stage 3), save the FinBERT projection (resumes from checkpoints, no re-encoding) and run E15.
Freezing v2 is a separate, reviewed step (scripts/shadow/freeze_model.py). Output: artifacts/logs/v2_stage4.log."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "artifacts" / "logs"


def text(f):
    p = LOGS / f
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


while "stage3 complete" not in text("v2_stage3.log"):
    if "stage3 failed" in text("v2_stage3.log"):
        sys.exit("E14 failed; E15 not started")
    time.sleep(60)
with open(LOGS / "v2_stage4.log", "a", encoding="utf-8") as f:
    for name, args in (("finbert_pca", ["scripts/data/build_finbert_features.py"]),
                       ("e15", ["scripts/experiments/e15_backtest_v2.py"])):
        if name == "finbert_pca" and (REPO / "artifacts" / "features" / "finbert_pca.npz").exists():
            continue
        f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {name}\n")
        f.flush()
        r = subprocess.run([sys.executable, "-B", *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"==== {name} exit {r.returncode}\n")
        f.flush()
    f.write("stage4 complete\n")

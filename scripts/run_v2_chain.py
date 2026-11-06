"""Run the v2 data and selection chain end to end, stopping at the first failure.

    python scripts/run_v2_chain.py [--wait-for-fetch artifacts/logs/fetch_etf_minute.log]

1. (optional) wait until a running ETF minute download logs "done"; otherwise run the resumable download itself
2. build ETF bars (verification table -> reports/data/etf_bar_verification.csv)
3. FRED/ALFRED vintages, if FRED_API_KEY is set and the store does not exist yet
4. v2 frame (rebuilds the market-feature cache from the full ETF bars)
5. E11 macro ladder
6. wait for FinBERT features (built separately by scripts/data/build_finbert_features.py), then E16
Each step's output goes to artifacts/logs/v2_chain.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = REPO / "artifacts" / "logs" / "v2_chain.log"


def step(name: str, args: list[str]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {name}: {' '.join(args)}\n")
        f.flush()
        r = subprocess.run([PY, "-B", *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"==== {name} exit {r.returncode}\n")
    if r.returncode != 0:
        sys.exit(f"{name} failed (exit {r.returncode}); see {LOG}")


def wait_for(cond, what: str, timeout_h: float) -> None:
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout_h * 3600:
            sys.exit(f"timed out waiting for {what}")
        time.sleep(60)


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from nexus_alpha import secrets
    secrets.load()                      # keys from NEXUS_ENV_FILES (owner-held files) if not already in the environment
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-fetch")
    ap.add_argument("--from", dest="start", default="fetch", choices=["fetch", "etf_bars", "fred", "v2_frame", "e11", "e16"])
    a = ap.parse_args()
    order = ["fetch", "etf_bars", "fred", "v2_frame", "e11", "e16"]
    todo = set(order[order.index(a.start):])
    if "fetch" not in todo:
        pass
    elif a.wait_for_fetch:
        p = REPO / a.wait_for_fetch
        wait_for(lambda: p.exists() and p.read_text(encoding="utf-8", errors="ignore").rstrip().endswith("done"),
                 "the ETF download", 12)
    else:
        step("fetch", ["scripts/data/fetch_etf_minute.py"])
    if "etf_bars" in todo:
        step("etf_bars", ["scripts/data/build_etf_bars.py"])
    if "fred" in todo and os.environ.get("FRED_API_KEY") and not (REPO / "artifacts" / "macro" / "alfred_vintages.parquet").exists():
        step("fred", ["-c", "import sys; sys.path.insert(0, 'src'); from nexus_alpha.macro import fred; "
                            "d = fred.build_store(); print(d.groupby('series_id').realtime_start.agg(['min', 'max', 'count']))"])
    if "v2_frame" in todo:
        step("v2_frame", ["scripts/data/build_v2_frame.py", "--rebuild-mkt"])
    if "e11" in todo:
        step("e11", ["scripts/experiments/e11_macro_ladder.py"])
    fb = REPO / "artifacts" / "features" / "finbert_features.parquet"
    wait_for(fb.exists, "FinBERT features", 8)
    step("e16", ["scripts/experiments/e16_text_encoders.py"])
    step("e18", ["scripts/experiments/e18_tabular_final.py"])
    print("v2 chain complete")


if __name__ == "__main__":
    main()

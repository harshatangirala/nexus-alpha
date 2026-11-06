"""Preflight for the nightly forward test (audit 3).

The 2026-11-06 07:00 IST run found Yahoo unreachable: every ticker failed after yfinance's own retries (about 4 minutes
each), the deadline was never set because no forecast succeeded, and the task was killed at its 3-hour limit after 45
tickers. This probe checks Yahoo before any forecast is attempted, and a done marker lets later triggers of the same
day run only when the earlier run did not complete.

    python scripts/shadow/preflight.py            # exit 0 go, 10 already completed today, 2 Yahoo unreachable
    python scripts/shadow/preflight.py --mark     # record that today's run completed
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "artifacts" / "logs"


def marker() -> Path:
    # every daily trigger (07:00 IST, and optional later ones the owner adds) falls on the same local date and as-of session
    return LOGS / f"shadow_ok_{pd.Timestamp.now().date().isoformat()}.txt"


def probe(timeout_s: float = 60.0) -> tuple[bool, str]:
    box: dict = {}

    def run():
        try:
            import yfinance as yf
            box["n"] = len(yf.Ticker("SPY").history(period="5d", auto_adjust=False, actions=False))
        except Exception as e:  # network down, DNS, rate limit
            box["e"] = repr(e)[:200]

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return False, "timeout"
    return box.get("n", 0) > 0, box.get("e", "empty answer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--attempts", type=int, default=60)      # 60 x 90 s: up to 90 minutes
    ap.add_argument("--pause", type=float, default=90.0)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.mark:
        marker().write_text(pd.Timestamp.now().isoformat(timespec="seconds"), encoding="utf-8")
        print(f"preflight: run marked complete ({marker().name})", flush=True)
        return
    if marker().exists():
        print(f"preflight: today's run already completed ({marker().name}); nothing to do", flush=True)
        sys.exit(10)
    for i in range(a.attempts):
        ok, err = probe()
        if ok:
            print(f"preflight: Yahoo reachable (attempt {i + 1})", flush=True)
            return
        print(f"preflight: attempt {i + 1}/{a.attempts} failed: {err}", flush=True)
        time.sleep(a.pause)
    print("PREFLIGHT FAILED: Yahoo unreachable; no forecast attempted (a later trigger today retries)", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()

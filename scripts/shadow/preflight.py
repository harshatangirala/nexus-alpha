"""Preflight for the nightly forward test (audit 3; done marker keyed by session in audit 6, A-64).

The 2026-11-06 07:00 IST run found Yahoo unreachable: every ticker failed after yfinance's own retries (about 4 minutes
each), the deadline was never set because no forecast succeeded, and the task was killed at its 3-hour limit after 45
tickers. This probe checks Yahoo before any forecast is attempted, and a done marker lets a later trigger run only
when the earlier run did not complete.

The marker is keyed by the as-of session the run must log (`shadow.expected_asof()`), not by the local date. Keyed by
date, a trigger after midnight IST and before the US close would have marked the new day complete with the previous
session, and the 07:00 run would then have skipped the new session.

    python scripts/shadow/preflight.py            # exit 0 go, 10 session already completed, 2 Yahoo unreachable
    python scripts/shadow/preflight.py --mark     # record that the expected session's run completed
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


def expected_session() -> str:
    sys.path.insert(0, str(REPO / "src"))
    from nexus_alpha import shadow as SH
    return str(SH.expected_asof().date())


def marker(session: str | None = None) -> Path:
    """Done marker of one as-of session. Its prefix differs from the old date-keyed `shadow_ok_<date>.txt` files, so
    no old marker can ever be read as a session's marker."""
    return LOGS / f"shadow_done_session_{session or expected_session()}.txt"


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
        sys.path.insert(0, str(REPO / "src"))
        from nexus_alpha import shadow as SH
        want = str(SH.expected_asof().date())
        missing = []
        for v in ("v1.1", "v2", "market_e19"):               # the run is complete only if each logged the expected session
            ch = SH.SHADOW_ROOT / v / "chain.csv"
            if not ch.exists() or want not in set(pd.read_csv(ch).asof_session.astype(str)):
                missing.append(v)
        if missing:
            print(f"preflight: NOT marked complete: no {want} forecasts logged for {missing}; a later trigger retries",
                  flush=True)
            return
        marker(want).write_text(pd.Timestamp.now().isoformat(timespec="seconds"), encoding="utf-8")
        print(f"preflight: run marked complete ({marker(want).name}; session {want} logged)", flush=True)
        return
    m = marker()
    if m.exists():
        print(f"preflight: this session's run already completed ({m.name}); nothing to do", flush=True)
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

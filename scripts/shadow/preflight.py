"""Preflight for the nightly forward test (audit 3; done marker keyed by session in audit 6, A-64).

The 2026-11-06 07:00 IST run found Yahoo unreachable: every ticker failed after yfinance's own retries (about 4 minutes
each), the deadline was never set because no forecast succeeded, and the task was killed at its 3-hour limit after 45
tickers. This probe checks Yahoo before any forecast is attempted, and a done marker lets a later trigger run only
when the earlier run did not complete.

The marker is keyed by the as-of session the run must log (`shadow.expected_asof()`), not by the local date. Keyed by
date, a trigger after midnight IST and before the US close would have marked the new day complete with the previous
session, and the 07:00 run would then have skipped the new session.

Audit 7 (A-80, 2026-11-13): reachable is not enough. At 07:00-07:16 IST on 2026-11-13 Yahoo answered, and SPY already
had the 2026-11-12 bar, but stock daily downloads still ended on 2026-11-11. v1.1 and v2 forecast their first stock as
of 2026-11-11, took that session's (passed) decision time as the deadline and stopped after one ticker; four of five
models missed the session until a manual re-run. The probe now also requires that most of a basket of stocks,
downloaded the way the forecaster downloads them, reach the expected session, and waits (same attempts) until they do.

    python scripts/shadow/preflight.py            # exit 0 go, 10 session already completed, 2 Yahoo unreachable or stale
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


BASKET = ("AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "HD", "CAT")   # liquid stocks of the universe, several sectors
MIN_FRESH = 6


def _last_bar(ticker: str):
    """Last daily bar date of one stock, downloaded as the forecaster downloads it (yf.download, daily, unadjusted)."""
    import yfinance as yf
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, actions=False, progress=False,
                    multi_level_index=False, threads=False)
    if d is None or len(d) == 0:
        return None
    d = d.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if not len(d):
        return None
    i = pd.DatetimeIndex(d.index)
    return (i.tz_localize(None) if i.tz is not None else i).normalize().max()


def fresh(want: str, basket=BASKET, last_bar=_last_bar) -> tuple[bool, str]:
    """True when at least MIN_FRESH stocks of the basket have a daily bar for the session `want` (A-80)."""
    got = {}
    for t in basket:
        try:
            got[t] = last_bar(t)
        except Exception:  # one failed stock does not decide; the count does
            got[t] = None
    n = sum(1 for v in got.values() if v is not None and pd.Timestamp(v) >= pd.Timestamp(want))
    stale = sorted(t for t, v in got.items() if v is None or pd.Timestamp(v) < pd.Timestamp(want))
    ok = n >= min(MIN_FRESH, len(basket))
    return ok, f"{n}/{len(basket)} stocks have session {want}" + ("" if ok else f"; behind: {stale}")


def probe(timeout_s: float = 60.0, want: str | None = None) -> tuple[bool, str]:
    box: dict = {}

    def run():
        try:
            import yfinance as yf
            box["n"] = len(yf.Ticker("SPY").history(period="5d", auto_adjust=False, actions=False))
            if box["n"] and want is not None:
                box["fresh"] = fresh(want)
        except Exception as e:  # network down, DNS, rate limit
            box["e"] = repr(e)[:200]

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s if want is None else timeout_s + 120)
    if th.is_alive():
        return False, "timeout"
    if not box.get("n"):
        return False, box.get("e", "empty answer")
    if want is not None:
        ok, why = box.get("fresh", (False, box.get("e", "no freshness answer")))
        return ok, f"reachable, but {why}" if not ok else why
    return True, ""


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
    want = expected_session()
    for i in range(a.attempts):
        ok, err = probe(want=want)
        if ok:
            print(f"preflight: Yahoo reachable and current (attempt {i + 1}; {err})", flush=True)
            return
        print(f"preflight: attempt {i + 1}/{a.attempts} failed: {err}", flush=True)
        time.sleep(a.pause)
    print(f"PREFLIGHT FAILED: Yahoo unreachable or without session {want}; no forecast attempted "
          "(a later trigger today retries)", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()

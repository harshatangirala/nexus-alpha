"""Extend the vendored XNYS session calendar beyond 2026-02-27 (needed for live inference), after verifying that
pandas_market_calendars reproduces the vendored TriFrame calendar exactly on the overlap.

Output: src/nexus_alpha/resources/xnys_sessions_extended.csv (date, session_minutes, early_close, halt_minutes, source)
Run: .venv/Scripts/python -I scripts/data/extend_calendar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas_market_calendars as pmc

RES = Path(__file__).resolve().parents[2] / "src" / "nexus_alpha" / "resources"
END = "2030-12-31"   # extended in audit 2026-11-06 (A-32)


def main() -> int:
    base = pd.read_csv(RES / "xnys_sessions.csv", parse_dates=["date"])
    nyse = pmc.get_calendar("XNYS")
    sch = nyse.schedule(start_date=base.date.min(), end_date=END)
    sch_dates = pd.DatetimeIndex(sch.index).normalize()
    close_et = sch.market_close.dt.tz_convert("America/New_York")
    open_et = sch.market_open.dt.tz_convert("America/New_York")
    minutes = ((close_et - open_et).dt.total_seconds() / 60).round().astype(int).values
    lib = pd.DataFrame({"date": sch_dates, "session_minutes": minutes, "early_close": close_et.dt.hour.values < 16})
    ov = lib[lib.date <= base.date.max()]
    only_base = sorted(set(base.date) - set(ov.date))
    only_lib = sorted(set(ov.date) - set(base.date))
    m = base.merge(ov, on="date", suffixes=("_b", "_l"))
    early_diff = m[m.early_close_b != m.early_close_l]
    print(f"overlap sessions: base {len(base)}, library {len(ov)}; only in base {len(only_base)}, only in library "
          f"{len(only_lib)}; early-close differences {len(early_diff)}")
    if only_base or only_lib:
        print("  only_base:", [str(x.date()) for x in only_base[:20]])
        print("  only_lib:", [str(x.date()) for x in only_lib[:20]])
    if len(early_diff):
        print(early_diff.head(20).to_string())
    ok = not only_base and not only_lib and early_diff.empty
    ext = lib[lib.date > base.date.max()].copy()
    ext["halt_minutes"] = 0
    ext["source"] = "pandas_market_calendars " + pmc.__version__
    base["source"] = "triframe market_calendar.csv"
    out = pd.concat([base, ext[base.columns]], ignore_index=True)
    out.to_csv(RES / "xnys_sessions_extended.csv", index=False, date_format="%Y-%m-%d")
    print(f"extended calendar: {len(out)} sessions to {out.date.max().date()} "
          f"({len(ext)} added); overlap verification {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

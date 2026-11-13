"""Live cross-section of fundamentals for the whole universe. LIVE INFERENCE ONLY (like the live Yahoo bars): it uses
facts filed after the seal and is never used to build a training, selection or evaluation dataset.

    python scripts/shadow/live_fundamentals.py [--workers 6] [--limit N] [--no-refresh]

1. refresh each company's SEC companyfacts file older than 20 hours, serially in this process (one rate limiter:
   <= 5 requests per second to data.sec.gov with the owner's declared User-Agent);
2. per ticker (parallel, no network): the as-known snapshot today, priced at the latest session close from the
   evening's live bar cache (NEXUS_LIVE_CACHE, shared with the forecaster) or Yahoo;
3. within-session universe percentiles (fr_*) and industry-neutral ranks (fi_*), exactly as in development;
4. artifacts/fundamentals/live_panel.parquet (git-ignored), read by the Trade Desk's live mode for percentiles.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402

OUT = ("fundamentals", "live_panel.parquet")


def tonight_cache() -> str:
    d = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    return str(paths.ARTIFACTS / "live_cache" / d)


def one(args) -> dict | None:
    ticker, cik, ambiguous, cache = args
    os.environ["NEXUS_LIVE_CACHE"] = cache
    warnings.filterwarnings("ignore")
    from nexus_alpha.fundamentals import ratios as RT, sec_facts as SF, service as FS, statements as ST
    from nexus_alpha.inference.pipeline import fetch_live
    try:
        f = SF.load_live(cik, max_age_hours=1e9)          # refreshed serially in the parent; no network here
        if f.empty:
            return None
        daily, _ = fetch_live(ticker)
        bar = daily[["date", "close"]].tail(1).copy()
        s = ST.snapshots(f, at=[pd.Timestamp.now().normalize()])
        s["snap_date"] = bar.date.iloc[0]
        d = RT.daily(s, bar, FS._splits(ticker), price_basis=None, share_class_ambiguous=ambiguous)
        if d.empty:
            return None
        r = d.iloc[-1].to_dict()
        r.update({"ticker": ticker, "latest_filing": f.filed.max(), "bar_date": bar.date.iloc[0]})
        try:            # E30 / D-060: extended metrics, display only; prefixed x_ so no frozen input column can change
            from nexus_alpha.fundamentals import extended as XF
            m = s.iloc[[-1]].reset_index(drop=True).assign(
                **{k: [r.get(k, float("nan"))] for k in ("market_cap", "enterprise_value", "pe", "eps_growth")})
            r.update({f"x_{k}": v for k, v in XF.metrics(m).iloc[0].items()})
        except Exception as e:  # noqa: BLE001  (the base row must survive)
            r["x_error"] = f"{type(e).__name__}: {e}"[:120]
        return r
    except Exception as e:  # one stock must not stop the cross-section
        return {"ticker": ticker, "error": f"{type(e).__name__}: {e}"[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    import build_fundamentals as BF
    from nexus_alpha.fundamentals import sec_facts as SF
    from nexus_alpha.fundamentals.ratios import RATIOS
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    if a.limit:
        cm = cm.head(a.limit)
    n_cls = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet").groupby("cik").ticker.size()
    if not a.no_refresh:
        stale = 0
        for c in sorted(cm.cik.unique()):
            p = SF.raw_path(c)
            if not p.exists() or (time.time() - p.stat().st_mtime) / 3600 > 20:
                try:
                    SF.fetch(c, refresh=True)
                    stale += 1
                except Exception as e:
                    print(f"SEC refresh failed for CIK {c}: {type(e).__name__}", flush=True)
        print(f"SEC facts refreshed for {stale} companies ({time.time() - t0:.0f}s)", flush=True)
    cache = tonight_cache()
    jobs = [(t, int(c), bool(n_cls.get(c, 1) > 1), cache) for t, c in cm.itertuples(index=False)]
    rows, errors = [], []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, jobs, chunksize=4):
            if r is None:
                continue
            (errors if "error" in r else rows).append(r)
    P = pd.DataFrame(rows)
    if P.empty:
        print("no live fundamentals rows", errors[:3])
        sys.exit(1)
    P["date"] = P.bar_date.max()                         # one cross-section: the latest session
    rank_cols = RATIOS + ["days_since_filing"]
    for c in rank_cols:
        P[f"fr_{c}"] = P[c].rank(pct=True)
    P = P.merge(BF.industry_table(), on="ticker", how="left")
    P = pd.concat([P, BF.industry_ranks(P, rank_cols)], axis=1)
    P["fund_present"] = P[RATIOS].notna().any(axis=1).astype(float)
    try:                # E30 composites across tonight's universe (x_ prefix; read by the dashboard only)
        from nexus_alpha.fundamentals import extended as XF
        base = P.set_index("ticker")
        xcols = [c for c in base.columns if c.startswith("x_") and c != "x_error"]
        feed = base.drop(columns=xcols + (["x_error"] if "x_error" in base else [])).join(
            base[xcols].rename(columns=lambda c: c[2:]))
        cs = XF.cross_section(feed, sic2=base["sic2"] if "sic2" in base else None)
        P = P.merge(cs.add_prefix("x_").reset_index(), on="ticker", how="left")
    except Exception as e:  # noqa: BLE001
        print(f"extended composites skipped: {type(e).__name__}: {e}", flush=True)
    P["built_utc"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    P.to_parquet(paths.artifact(*OUT), index=False)
    stale_bars = int((P.bar_date < P.bar_date.max()).sum())
    print(f"live fundamentals: {len(P)} tickers, {len(errors)} errors, {stale_bars} with an older last bar, "
          f"session {pd.Timestamp(P.date.iloc[0]).date()} ({time.time() - t0:.0f}s)", flush=True)
    if errors:
        print("errors:", [(e["ticker"], e["error"][:80]) for e in errors[:5]], flush=True)


if __name__ == "__main__":
    main()

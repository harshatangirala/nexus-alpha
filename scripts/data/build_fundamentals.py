"""Point-in-time fundamentals panel for the source universe (E22; docs/V3_RESEARCH.md).

    python scripts/data/build_fundamentals.py [--workers 10]

1. per CIK: as-known statements at every filing date (facts filed before the seal only) -> artifacts/fundamentals/
   snapshots/CIK<cik>.parquet (cached);
2. per ticker: daily ratios for 2009-01-02 .. 2022-12-30 at the as-traded close (split history: fundamentals/splits.py),
   SUE from the SEC study's S06 at its public time (acceptance per S07, or the time its history became public);
3. within-session cross-sectional percentile ranks of every ratio among universe stocks that have it (fr_*), and the
   E23 industry-neutral ranks within the SIC major group / division (fi_*);
4. artifacts/fundamentals/panel.parquet and reports/data/fundamentals_coverage.csv.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, seal  # noqa: E402

START, END = "2009-01-02", "2022-12-30"
SNAP = "fundamentals/snapshots"


def _stage(stage: str, name: str) -> str:
    return glob.glob(str(paths.SEC_FULL500_RUN / stage / "*" / name))[0]


def splits_table() -> pd.DataFrame:
    """The combined split history (vendor Yahoo files, yfinance, S03; nexus_alpha.fundamentals.splits)."""
    from nexus_alpha.fundamentals import splits as SP
    return SP.table()[["ticker", "date", "value"]]


SIC_DIVISIONS = ([-1, 999, 1499, 1999, 3999, 4999, 5199, 5999, 6799, 8999, 9999], list("ABCDEFGHIJ"))
MIN_GROUP = 8      # E23: rank within the SIC major group only when at least 8 stocks have the ratio that day


def industry_table() -> pd.DataFrame:
    """ticker -> SIC major group (first two digits) and SIC division (A-J), from the SEC study's S02 universe (E23)."""
    u = pd.read_parquet(_stage("S02", "universe.parquet"))[["ticker", "sic"]]
    sic = pd.to_numeric(u.sic, errors="coerce")
    extra = paths.ARTIFACTS / "fundamentals" / "sic_sec.parquet"          # scripts/data/fetch_sic.py: S02 gaps
    if extra.exists():
        e = pd.read_parquet(extra).set_index("ticker").sic
        sic = sic.fillna(pd.to_numeric(u.ticker.map(e), errors="coerce"))
    edges, labels = SIC_DIVISIONS
    return pd.DataFrame({"ticker": u.ticker, "sic2": (sic // 100).astype("Int64").astype(str).replace("<NA>", None),
                         "sic_div": pd.cut(sic, bins=edges, labels=labels).astype(str).replace("nan", None)})


def industry_ranks(P: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """E23 (pre-registered): within-session percentile within the SIC major group when >= MIN_GROUP stocks have the
    ratio, else within the SIC division when >= MIN_GROUP, else the universe rank."""
    g2, gd = P.groupby(["date", "sic2"]), P.groupby(["date", "sic_div"])
    out = {}
    for c in cols:
        n2, nd = g2[c].transform("count"), gd[c].transform("count")
        r2, rd = g2[c].rank(pct=True), gd[c].rank(pct=True)
        out[f"fi_{c}"] = np.where(n2 >= MIN_GROUP, r2, np.where(nd >= MIN_GROUP, rd, P[f"fr_{c}"])).astype(np.float32)
    return pd.DataFrame(out, index=P.index)


def sue_table() -> pd.DataFrame:
    sue = pd.read_parquet(_stage("S06", "sue_by_accession.parquet"))[["cik", "accession", "sue", "hist_max_public_time"]]
    fm = pd.read_parquet(_stage("S07", "filing_manifest_timed.parquet"))[["accession", "acceptance_json_et"]]
    s = sue.merge(fm.drop_duplicates("accession"), on="accession", how="left").dropna(subset=["sue"])
    acc = pd.to_datetime(s.acceptance_json_et, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    hist = pd.to_datetime(s.hist_max_public_time, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    avail = pd.concat([acc, hist], axis=1).max(axis=1)
    s["avail_date"] = avail.dt.tz_localize(None).dt.normalize()
    s = s.dropna(subset=["avail_date"])
    return s[s.avail_date < seal.SEAL_DATE][["cik", "avail_date", "sue"]]


def snap_one(cik: int) -> tuple[int, int]:
    from nexus_alpha.fundamentals import sec_facts as SF, statements as ST
    out = paths.ARTIFACTS / SNAP / f"CIK{int(cik):010d}.parquet"
    if out.exists():
        return cik, -1
    s = ST.snapshots(SF.load(cik))
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(out, index=False)
    return cik, len(s)


def ratios_one(args) -> pd.DataFrame:
    ticker, cik, spl, sue, ambiguous = args
    from nexus_alpha.data import bars as B
    from nexus_alpha.fundamentals import ratios as RT
    p = paths.ARTIFACTS / SNAP / f"CIK{int(cik):010d}.parquet"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_parquet(p)
    if s.empty:
        return pd.DataFrame()
    try:
        bars = B.read_daily(ticker)
    except Exception:
        return pd.DataFrame()
    d = RT.daily(s, bars, spl, sue, start=START, end=END, share_class_ambiguous=ambiguous)
    if d.empty:
        return d
    d.insert(0, "ticker", ticker)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    from nexus_alpha.fundamentals.ratios import RATIOS
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    ciks = sorted(cm.cik.unique())
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (c, n) in enumerate(ex.map(snap_one, ciks, chunksize=2)):
            if (i + 1) % 100 == 0:
                print(f"snapshots {i + 1}/{len(ciks)} ({time.time() - t0:.0f}s)", flush=True)
    spl = splits_table()
    sue = sue_table()
    jobs = []
    n_cls = cm.groupby("cik").ticker.transform("size")
    for (t, c), k in zip(cm[["ticker", "cik"]].itertuples(index=False), n_cls):
        sp = spl[spl.ticker == t].set_index("date").value
        su = sue[sue.cik == c]
        jobs.append((t, c, sp, su, bool(k > 1)))
    parts = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, d in enumerate(ex.map(ratios_one, jobs, chunksize=4)):
            if len(d):
                parts.append(d)
            if (i + 1) % 200 == 0:
                print(f"ratios {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    P = pd.concat(parts, ignore_index=True)
    rank_cols = RATIOS + ["days_since_filing"]
    g = P.groupby("date")
    for c in rank_cols:
        P[f"fr_{c}"] = g[c].rank(pct=True).astype(np.float32)
    P["fund_present"] = P[RATIOS].notna().any(axis=1).astype(np.float32)
    P = P.merge(industry_table(), on="ticker", how="left")
    P = pd.concat([P, industry_ranks(P, rank_cols)], axis=1)
    print(f"industry ranks: SIC group used for {np.mean(P.groupby(['date', 'sic2']).earnings_yield.transform('count') >= MIN_GROUP):.1%} "
          f"of rows (earnings yield)", flush=True)
    for c in P.columns:
        if P[c].dtype == np.float64:
            P[c] = P[c].astype(np.float32)
    assert P.date.max() < seal.SEAL_DATE
    P.to_parquet(paths.artifact("fundamentals", "panel.parquet"), index=False)
    cov = P.assign(year=P.date.dt.year).groupby("year")[RATIOS].apply(lambda x: x.notna().mean()).round(3)
    cov.insert(0, "tickers", P.assign(year=P.date.dt.year).groupby("year").ticker.nunique())
    (REPO / "reports" / "data").mkdir(parents=True, exist_ok=True)
    cov.to_csv(REPO / "reports" / "data" / "fundamentals_coverage.csv")
    print(f"panel {P.shape}, tickers {P.ticker.nunique()}, {P.date.min().date()}..{P.date.max().date()} ({time.time() - t0:.0f}s)")
    print(cov.iloc[[0, 3, 7, -1], :12].to_string())


if __name__ == "__main__":
    main()

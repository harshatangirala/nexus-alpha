"""Chief-audit checks of the fundamentals panel (E22 data).

    python scripts/verify/audit_fundamentals.py
1. point in time: 40 random (ticker, date) cells recomputed from facts filed on or before that date only;
2. split continuity: market cap across EVERY split of 2009-2022 (no jump by the split factor);
3. one-day market-cap jumps > 50%: each classified as a price move (the close moved as much) or a share-basis jump;
4. ranking sanity: largest market caps at 2021-12-31, and ten megacaps against public year-end values (+-15%);
5. P/E consistency: market cap / TTM net income vs as-traded price / TTM diluted EPS;
6. currency: reporting currency per company (non-USD reporters carry no money facts) and foreign-filer rows;
7. coverage by year and the share of extreme values per ratio; 8. seal.
Writes reports/verification/fundamentals_audit.json; exits non-zero when a hard check fails.
"""
from __future__ import annotations

import gzip
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.fundamentals import ratios as RT, sec_facts as SF, statements as ST  # noqa: E402


def main() -> None:
    import build_fundamentals as BF
    rep = {}
    P = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "panel.parquet")
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet").set_index("ticker").cik
    spl = BF.splits_table()
    # 1. point in time
    cells = P.dropna(subset=["earnings_yield"]).sample(40, random_state=3)
    bad = []
    for r in cells.itertuples():
        f = SF.load(cm[r.ticker])
        f = f[f.filed <= r.date]
        s = ST.snapshots(f, at=[f.filed.max()])
        sp = spl[spl.ticker == r.ticker].set_index("date").value
        d = RT.daily(s, B.read_daily(r.ticker), sp, start=str(r.date.date()), end=str(r.date.date()),
                     share_class_ambiguous=bool((cm == cm[r.ticker]).sum() > 1))
        v = float(d.earnings_yield.iloc[0]) if len(d) else np.nan
        if not np.isclose(v, r.earnings_yield, rtol=1e-4, equal_nan=True):
            bad.append((r.ticker, str(r.date.date()), float(r.earnings_yield), v))
    rep["pit_recompute"] = {"cells": 40, "mismatches": bad}
    # 2. split continuity across every split of 2009-2022
    P = P.sort_values(["ticker", "date"]).reset_index(drop=True)
    P["dl_mcap"] = np.log(P.market_cap).groupby(P.ticker).diff()
    P["dl_px"] = np.log(P.price_as_traded).groupby(P.ticker).diff()
    win = spl[(spl.date >= "2009-01-10") & (spl.date <= "2022-12-20") & ((spl.value >= 1.25) | (spl.value <= 0.8))]
    bad_split = []
    for t, d, v in win.itertuples(index=False):
        x = P[(P.ticker == t) & (P.date >= d - pd.Timedelta(days=8)) & (P.date <= d + pd.Timedelta(days=8))]
        if x.dl_mcap.notna().any():
            j = float(x.dl_mcap.abs().max())
            if j > 0.5 * abs(np.log(v)) and j > np.log(1.25):
                bad_split.append((t, str(d.date()), v, round(j, 3)))
    rep["split_continuity"] = {"splits_checked": int(len(win)), "jumps_matching_split": bad_split}
    # 3. one-day market-cap jumps > 50%
    big = P[P.dl_mcap.abs() > np.log(1.5)]
    basis = big[(big.dl_mcap - big.dl_px).abs() > np.log(1.2)]
    rep["mcap_jumps_gt_50pct"] = {"rows": int(len(big)), "price_moves": int(len(big) - len(basis)),
                                  "share_basis_jumps": [(t, str(d.date()), round(float(a), 3), round(float(b), 3))
                                                        for t, d, a, b in basis[["ticker", "date", "dl_mcap", "dl_px"]]
                                                        .head(40).itertuples(index=False)],
                                  "share_basis_jump_rows": int(len(basis))}
    # 4. ranking sanity and megacaps vs public year-end 2021 values ($bn, approximate)
    day = P[P.date == P.date[P.date <= "2021-12-31"].max()]
    rep["top25_market_cap_2021_12_31"] = [(t, round(m / 1e9)) for t, m in
                                          day.nlargest(25, "market_cap")[["ticker", "market_cap"]].itertuples(index=False)]
    ref = {"AAPL": 2913, "MSFT": 2522, "GOOG": 1917, "AMZN": 1691, "TSLA": 1061, "META": 936, "NVDA": 735, "UNH": 473,
           "JPM": 466, "JNJ": 450}
    got = day.set_index("ticker").market_cap / 1e9
    rep["megacaps_vs_public"] = {t: {"panel": None if pd.isna(got.get(t)) else round(float(got.get(t))), "public": v,
                                     "ok": bool(pd.notna(got.get(t)) and abs(got.get(t) / v - 1) <= 0.15)}
                                 for t, v in ref.items()}
    # 5. P/E consistency
    q = P[["ticker", "date", "pe", "price_as_traded", "market_cap"]]
    snaps = []
    for t in q.ticker.unique()[:: max(1, q.ticker.nunique() // 150)]:              # about 150 tickers
        sp_ = paths.ARTIFACTS / "fundamentals" / "snapshots" / f"CIK{int(cm[t]):010d}.parquet"
        if sp_.exists():
            sn = pd.read_parquet(sp_, columns=["snap_date", "ttm_eps_diluted"]).sort_values("snap_date")
            sn["snap_date"] = pd.to_datetime(sn.snap_date).astype("datetime64[ns]")
            x = q[q.ticker == t].copy()
            x["date"] = x.date.astype("datetime64[ns]")
            snaps.append(pd.merge_asof(x.sort_values("date"), sn, left_on="date", right_on="snap_date"))
    z = pd.concat(snaps)
    z = z[z.pe.notna() & (z.ttm_eps_diluted.abs() >= 0.05) & (z.pe > 0)]
    ratio = (z.pe / (z.price_as_traded / z.ttm_eps_diluted)).abs()
    rep["pe_consistency"] = {"rows": int(len(z)), "share_within_1.5x": float(((ratio > 1 / 1.5) & (ratio < 1.5)).mean()),
                             "median_ratio": float(ratio.median())}
    # 6. currency and foreign filers
    cur = Counter()
    non_usd = []
    for t, c in cm.items():
        p_ = SF.raw_path(c)
        if not p_.exists():
            continue
        u = SF.reporting_currency(json.loads(gzip.decompress(p_.read_bytes())).get("facts") or {})
        cur[str(u)] += 1
        if u not in (None, "USD"):
            non_usd.append((t, u))
    rep["reporting_currency"] = dict(cur)
    rep["non_usd_reporters_without_money_facts"] = non_usd
    leak = P[P.ticker.isin([t for t, _ in non_usd]) & P.earnings_yield.notna()]
    rep["non_usd_rows_with_ratios"] = int(len(leak))
    rep["foreign_filer_rows"] = int(P.foreign_filer.sum())
    # 5. coverage and extremes
    from nexus_alpha.fundamentals.ratios import RATIOS
    rep["coverage_by_year"] = P.assign(y=P.date.dt.year).groupby("y")[["earnings_yield", "fcf_yield", "roe", "piotroski_f", "sue"]].apply(
        lambda x: x.notna().mean()).round(3).to_dict()
    rep["extreme_share"] = {c: float(((P[c].abs() > 50) & P[c].notna()).mean()) for c in RATIOS}
    # 8. seal
    from nexus_alpha import seal
    rep["seal"] = {"panel_max_date": str(P.date.max().date()), "before_seal": bool(P.date.max() < seal.SEAL_DATE),
                   "evaluation_open": bool(seal.is_open())}
    hard = {"pit": not bad, "splits": not bad_split,
            "share_basis_jumps": rep["mcap_jumps_gt_50pct"]["share_basis_jump_rows"] <= 25,
            "megacaps": all(v["ok"] for v in rep["megacaps_vs_public"].values()),
            "pe_consistency": rep["pe_consistency"]["share_within_1.5x"] >= 0.98,
            "non_usd_clean": rep["non_usd_rows_with_ratios"] == 0,
            "seal": rep["seal"]["before_seal"] and not rep["seal"]["evaluation_open"]}
    rep["hard_checks"] = hard
    out = REPO / "reports" / "verification" / "fundamentals_audit.json"
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("coverage_by_year", "extreme_share")}, indent=1, default=str)[:6000])
    print("extreme (|x|>50) share > 1%:", {k: round(v, 3) for k, v in rep["extreme_share"].items() if v > 0.01})
    print("HARD CHECKS:", hard)
    if not all(hard.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

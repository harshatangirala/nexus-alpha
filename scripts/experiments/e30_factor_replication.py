"""E30 (D-060): do published fundamental composites rank this universe, point in time? One shot, pre-registered in
docs/E30_PREREGISTRATION.md; nothing fitted.

    python scripts/experiments/e30_factor_replication.py

Writes reports/e30/e30_results.json and reports/e30/e30_monthly_ic.csv. Reads only the sealed development stores
(bars before 2023-01-03, the fundamentals snapshots and panel); never the sealed period.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.data.calendar import sessions  # noqa: E402
from nexus_alpha.evaluation.forward_criteria import block_p_value  # noqa: E402
from nexus_alpha.fundamentals import extended as X  # noqa: E402

PRIMARY = ["value_composite", "quality_composite", "gross_profitability_rank", "magic_formula", "piotroski_rank",
           "beneish_safety_rank"]
DESCRIPTIVE = ["quality_profitability", "quality_growth", "quality_safety", "montier_safety"]
PUBLISHED = {"value_composite": 1992, "quality_composite": 2013, "gross_profitability_rank": 2010, "magic_formula": 2005,
             "piotroski_rank": 2000, "beneish_safety_rank": 1999}
H, BLOCK, N_BOOT, SEED, MIN_STOCKS = 20, 6, 10_000, 30, 100
SEAL_LAST = pd.Timestamp("2022-12-30")
SUBPERIODS = [("2005-2011", 2005, 2011), ("2012-2016", 2012, 2016), ("2017-2022", 2017, 2022)]
OUT = REPO / "reports" / "e30"


def signal_dates() -> list[pd.Timestamp]:
    s = sessions()
    s = s[(s >= "2005-01-01") & (s <= SEAL_LAST)]
    month_end = pd.Series(s, index=s).groupby([s.year, s.month]).max().tolist()
    pos = {d: i for i, d in enumerate(s)}
    return [d for d in month_end if pos[d] + H < len(s)]          # exit session on or before 2022-12-30


def returns(ticker: str, dates: list[pd.Timestamp], cal: pd.DatetimeIndex) -> pd.Series:
    """R = close[t+20] / open[t+1] - 1 on exact calendar sessions; NaN when a bar is missing."""
    d = B.read_daily(ticker).set_index("date")
    if d.empty:
        return pd.Series(np.nan, index=dates)
    pos = cal.get_indexer(pd.DatetimeIndex(dates))
    nxt, ext = cal[pos + 1], cal[pos + H]
    o = d.open.reindex(nxt).to_numpy(float)
    c = d.close.reindex(ext).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c / o - 1
    return pd.Series(np.where(np.isfinite(r) & (o > 0), r, np.nan), index=dates)


def build_panel(dates: list[pd.Timestamp]) -> pd.DataFrame:
    keep = ["ticker", "date", "market_cap", "enterprise_value", "earnings_yield", "sales_yield", "book_to_price",
            "ebitda_to_ev", "fcf_yield", "gross_margin", "roe", "roa", "accruals", "piotroski_f", "altman_z", "pe",
            "eps_growth", "sic2"]
    t = pq.read_table(paths.ARTIFACTS / "fundamentals" / "panel.parquet", columns=keep,
                      filters=[("date", "in", [d.to_pydatetime() for d in dates])])
    p = t.to_pandas()
    p["date"] = pd.to_datetime(p.date).astype("datetime64[ns]")
    assert p.date.max() <= SEAL_LAST
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    cal = sessions()
    cal = cal[cal <= SEAL_LAST]
    out = []
    for tk, g in p.groupby("ticker"):
        r = cm[cm.ticker == tk]
        if r.empty:
            continue
        sp = paths.ARTIFACTS / "fundamentals" / "snapshots" / f"CIK{int(r.cik.iloc[0]):010d}.parquet"
        if not sp.exists():
            continue
        s = pd.read_parquet(sp).sort_values("snap_date")
        s["snap_date"] = pd.to_datetime(s.snap_date).astype("datetime64[ns]")
        s = s[s.snap_date <= SEAL_LAST]
        g = g.sort_values("date")
        m = pd.merge_asof(g, s, left_on="date", right_on="snap_date", direction="backward")   # filed on/before t
        try:
            m["R"] = returns(tk, list(g.date), cal).to_numpy()
        except FileNotFoundError:
            continue
        out.append(m)
    df = pd.concat(out, ignore_index=True)
    return pd.concat([df, X.metrics(df)], axis=1)


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < MIN_STOCKS:
        return np.nan
    return float(a[ok].rank().corr(b[ok].rank()))


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dates = signal_dates()
    print(f"{len(dates)} signal dates, {dates[0].date()} .. {dates[-1].date()}", flush=True)
    df = build_panel(dates)
    print(f"panel {len(df):,} stock-months, {df.ticker.nunique()} tickers ({time.time() - t0:.0f}s)", flush=True)
    recs, cs_all = [], []
    for d, g in df.groupby("date"):
        g = g.set_index("ticker")
        cs = X.cross_section(g, sic2=g.sic2)
        cs["montier_safety"] = X._pct(g.montier_c, higher_is_better=False)
        cs["R"], cs["date"], cs["sic2"] = g.R, d, g.sic2
        cs["size_rank"] = g.market_cap.rank(pct=True)
        cs_all.append(cs.reset_index())
        rec = {"date": d, "n_R": int(g.R.notna().sum())}
        for c in PRIMARY + DESCRIPTIVE:
            rec[f"ic_{c}"] = spearman(cs[c], cs.R)
            rec[f"n_{c}"] = int((cs[c].notna() & cs.R.notna()).sum())
            # industry-neutral: rank inside two-digit SIC
            ind = cs.groupby("sic2")[c].rank(pct=True)
            rec[f"icind_{c}"] = spearman(ind, cs.R)
            # size removed: residual of the score on the market-cap rank
            ok = cs[c].notna() & cs.size_rank.notna()
            res = pd.Series(np.nan, index=cs.index)
            if ok.sum() >= MIN_STOCKS:
                beta = np.polyfit(cs.size_rank[ok], cs[c][ok], 1)
                res[ok] = cs[c][ok] - np.polyval(beta, cs.size_rank[ok])
            rec[f"icsize_{c}"] = spearman(res, cs.R)
            # top-minus-bottom quintile spread of R, equal-weighted, gross
            ok = cs[c].notna() & cs.R.notna()
            if ok.sum() >= MIN_STOCKS:
                q = pd.qcut(cs[c][ok].rank(method="first"), 5, labels=False)
                rec[f"spread_{c}"] = float(cs.R[ok][q == 4].mean() - cs.R[ok][q == 0].mean())
            else:
                rec[f"spread_{c}"] = np.nan
        recs.append(rec)
    M = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    M.to_csv(OUT / "e30_monthly_ic.csv", index=False, float_format="%.6g")
    CS = pd.concat(cs_all, ignore_index=True)

    res = {"registered": "docs/E30_PREREGISTRATION.md", "decision": "D-060", "signal_dates": len(M),
           "first": str(M.date.min().date()), "last": str(M.date.max().date()), "horizon_sessions": H,
           "block_months": BLOCK, "n_boot": N_BOOT, "seed": SEED, "primary": {}, "descriptive": {}}
    pv = {}
    for c in PRIMARY + DESCRIPTIVE:
        ic = M.set_index("date")[f"ic_{c}"]
        v = ic.dropna()
        p = block_p_value(v.reset_index(drop=True), block=BLOCK, n_boot=N_BOOT, seed=SEED) if len(v) > 12 else np.nan
        rng = np.random.default_rng(SEED)
        k = int(np.ceil(len(v) / BLOCK))
        boots = [np.concatenate([v.to_numpy()[s:s + BLOCK] for s in rng.integers(0, len(v) - BLOCK + 1, k)])[:len(v)].mean()
                 for _ in range(2000)]
        sub = {name: float(ic[(ic.index.year >= a) & (ic.index.year <= b)].mean()) for name, a, b in SUBPERIODS}
        d = {"mean_ic": float(v.mean()), "ic_sd": float(v.std()), "months": int(len(v)),
             "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
             "share_positive_months": float((v > 0).mean()), "p_one_sided": p,
             "mean_ic_industry_neutral": float(M[f"icind_{c}"].mean()),
             "mean_ic_size_removed": float(M[f"icsize_{c}"].mean()),
             "mean_q5_minus_q1_20d": float(M[f"spread_{c}"].mean()),
             "mean_stocks_per_month": float(M[f"n_{c}"].replace(0, np.nan).mean()), "subperiods": sub}
        if c in PUBLISHED:
            y = PUBLISHED[c]
            d["published"] = y
            d["mean_ic_after_publication"] = float(ic[ic.index.year > y].mean())
            d["mean_ic_up_to_publication"] = float(ic[ic.index.year <= y].mean()) if (ic.index.year <= y).any() else None
        (res["primary"] if c in PRIMARY else res["descriptive"])[c] = d
        if c in PRIMARY:
            pv[c] = p
    adj = holm(pv)
    for c in PRIMARY:
        res["primary"][c]["p_holm"] = adj[c]
        res["primary"][c]["replicates"] = bool(adj[c] < 0.05 and res["primary"][c]["mean_ic"] > 0)
    corr = CS.groupby("date")[PRIMARY].corr(method="spearman").groupby(level=1).mean()
    res["composite_correlation"] = corr.loc[PRIMARY, PRIMARY].round(3).to_dict()
    res["runtime_s"] = round(time.time() - t0)
    (OUT / "e30_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({c: {k: res["primary"][c][k] for k in ("mean_ic", "ci95", "p_holm", "replicates",
                                                            "mean_q5_minus_q1_20d", "mean_stocks_per_month")}
                      for c in PRIMARY}, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

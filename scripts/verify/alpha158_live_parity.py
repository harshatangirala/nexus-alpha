"""Live parity of Alpha158: Yahoo bars (what the nightly forecaster sees) against the vendor bars v3_alpha was trained on.

E25's use clause asks that live features "equal the development computation on overlapping sessions". Two vendors never
agree to the last digit, so the test was made concrete here, and in D-049, before it was run:

* Sample: 40 tickers drawn with seed 11 from the forward-test universe (the v2 frame's 500 stocks). Yahoo daily bars
  are taken from 2020-09-01 to 2022-12-30, with auto_adjust=False (split-adjusted, not dividend-adjusted, like the
  live fetch and the vendor). Features are compared on 2022 sessions only, after more than 60 warm-up sessions on
  both sides.
* Criterion 1, price features (every feature not built from volume): the median of the per-feature Pearson
  correlations (Yahoo against vendor, pooled stock-days) is at least 0.99, and at least 90% of them are at least 0.95.
* Criterion 2, end to end: on the frame's 2022 rows of the sample tickers, the v3_alpha bundle is given the vendor F0
  row with either A158 version. Its rank_score (p_up - p_down) agrees with Spearman correlation at least 0.95 at
  every horizon.
* Volume features (CORR, CORD, VMA, VSTD, WVMA, VSUMP/N/D) are reported only. Yahoo's consolidated daily volume differs
  from the vendor's sum of minute bars; criterion 2 measures what that does to the model.

    python scripts/verify/alpha158_live_parity.py [--bundle <v3_alpha bundle dir>]

Writes reports/verification/alpha158_live_parity.json and exits 0 only if both criteria hold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.indicators import alpha158 as A  # noqa: E402

N, SEED, START, END = 40, 11, "2020-09-01", "2022-12-31"
VOLUME = ("CORR", "CORD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD")


def yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    from nexus_alpha.inference.pipeline import yahoo_symbol
    d = yf.download(yahoo_symbol(ticker), start=START, end=END, interval="1d", auto_adjust=False, actions=False,
                    progress=False, multi_level_index=False, threads=False, timeout=60)
    if d is None or len(d) == 0:
        raise ValueError("no Yahoo rows")
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    return pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                         "low": d["Low"].values, "close": d["Close"].values, "volume": d["Volume"].values}).dropna()


def main() -> None:
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None, help="v3_alpha bundle dir (default: the newest v3_alpha refit)")
    a = ap.parse_args()
    bdir = Path(a.bundle) if a.bundle else sorted((paths.ARTIFACTS / "bundles").glob("*_v3_alpha_refit_*"))[-1]
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    rng = np.random.default_rng(SEED)
    tickers = sorted(rng.choice(sorted(fr.ticker.unique()), N, replace=False).tolist())
    feats_v, feats_y, failed, t0 = [], [], [], time.time()
    for t in tickers:
        try:
            y = yahoo(t)
        except Exception as e:  # noqa: BLE001  (counted and reported)
            failed.append((t, f"{type(e).__name__}: {e}"[:120]))
            continue
        v = B.read_daily(t)
        v = v[(v.date >= START) & (v.date < END)].reset_index(drop=True)
        fv = A.alpha158(v)
        fv.insert(0, "date", v.date.to_numpy())
        fy = A.alpha158(y)
        fy.insert(0, "date", y.date.to_numpy())
        for f, lst in ((fv, feats_v), (fy, feats_y)):
            f.insert(0, "ticker", t)
            lst.append(f[(f.date >= "2022-01-01")])
    V, Y = pd.concat(feats_v), pd.concat(feats_y)
    M = V.merge(Y, on=["ticker", "date"], suffixes=("_v", "_y"))
    per = {}
    for n in A.NAMES:
        c = A.PREFIX + n
        x, z = M[c + "_v"].to_numpy(float), M[c + "_y"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(z)
        r = float(np.corrcoef(x[ok], z[ok])[0, 1]) if ok.sum() > 10 and x[ok].std() > 0 and z[ok].std() > 0 else float("nan")
        per[n] = {"corr": round(r, 5), "rows": int(ok.sum()), "volume_based": n.startswith(VOLUME)}
    price = [v["corr"] for v in per.values() if not v["volume_based"] and np.isfinite(v["corr"])]
    vol = [v["corr"] for v in per.values() if v["volume_based"] and np.isfinite(v["corr"])]
    c1 = bool(np.median(price) >= 0.99 and np.mean(np.array(price) >= 0.95) >= 0.90)
    # criterion 2: the model's score with vendor A158 vs Yahoo A158, F0 from the frame (vendor), frame rows of 2022
    rows = fr[fr.ticker.isin(set(M.ticker)) & (fr.asof_date >= "2022-01-01")][["ticker", "asof_date"] +
                                                                              [c for c in meta["cols"] if not c.startswith(A.PREFIX)]]
    ends = {}
    for tag, src in (("vendor", V), ("yahoo", Y)):
        s = src.rename(columns={"date": "asof_date"})
        s["asof_date"] = pd.to_datetime(s.asof_date).astype("datetime64[ns]")
        ends[tag] = rows.merge(s, on=["ticker", "asof_date"], how="inner")
    both = ends["vendor"][["ticker", "asof_date"]].merge(ends["yahoo"][["ticker", "asof_date"]])
    by_h = {}
    for h in meta["horizons"]:
        bst = lgb.Booster(model_file=str(bdir / f"lgbm_{h}.txt"))
        sc = {}
        for tag, d in ends.items():
            d = both.merge(d, on=["ticker", "asof_date"])
            P = bst.predict(d[meta["cols"]].astype(float))
            sc[tag] = P[:, 2] - P[:, 0]
        rho = float(pd.Series(sc["vendor"]).rank().corr(pd.Series(sc["yahoo"]).rank()))
        by_h[h] = {"spearman_rank_score": round(rho, 5), "rows": int(len(both)),
                   "mean_abs_diff_up_minus_down": round(float(np.mean(np.abs(sc["vendor"] - sc["yahoo"]))), 6)}
    c2 = all(v["spearman_rank_score"] >= 0.95 for v in by_h.values())
    out = {"bundle": bdir.name, "tickers_requested": N, "tickers_compared": int(M.ticker.nunique()),
           "stock_days_2022": int(len(M)), "failed": failed, "criteria_written_before_run": "see the module docstring and D-049",
           "price_features": {"n": len(price), "median_corr": round(float(np.median(price)), 5),
                              "share_ge_0.95": round(float(np.mean(np.array(price) >= 0.95)), 4),
                              "min_corr": round(float(np.min(price)), 5)},
           "volume_features": {"n": len(vol), "median_corr": round(float(np.median(vol)), 5) if vol else None,
                               "min_corr": round(float(np.min(vol)), 5) if vol else None},
           "criterion_1_price_features": c1, "end_to_end_by_horizon": by_h, "criterion_2_end_to_end": c2,
           "passed": bool(c1 and c2), "lowest_price_features": sorted(
               [(n, v["corr"]) for n, v in per.items() if not v["volume_based"]], key=lambda x: x[1])[:6],
           "by_feature": per, "elapsed_s": round(time.time() - t0, 1)}
    dst = REPO / "reports" / "verification" / "alpha158_live_parity.json"
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("tickers_compared", "stock_days_2022", "price_features", "volume_features",
                                          "criterion_1_price_features", "criterion_2_end_to_end", "passed")}, indent=1))
    print({h: v["spearman_rank_score"] for h, v in by_h.items()}, "| failed:", failed[:5])
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()

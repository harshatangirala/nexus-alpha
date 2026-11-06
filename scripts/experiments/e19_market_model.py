"""E19 market-direction model (pre-registered in docs/V2_ARCHITECTURE.md section 4, decision D-029).

One row per session of the v2 frame. Label: SPY's own v2 label (same vol-scaled 3-class rule) per horizon. Features: the
date-level values the stock models saw. Nested sets P (training prior) < M1 SPY technicals < M2 + other market < M3 + GMD
< M4 + FRED, and a placebo (M4 with whole feature rows permuted across dates). Logistic (penalty chosen on inner 2016)
and small LightGBM (early stopping on inner). Keep rule as E11, per family, on validation, with a moving-block bootstrap
of max(1, ceil(h/21)) months and only horizons with >= 4 non-overlapping validation windows in the keep decision.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import etf_bars as EB  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2, make_labels_v2, purge_per_horizon  # noqa: E402

SEED = 11
PENALTIES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1,
              "num_leaves": 4, "min_data_in_leaf": 60, "learning_rate": 0.03, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 7, "deterministic": True,
              "force_row_wise": True, "num_threads": 2}
MIN_WINDOWS = 4


def date_frame() -> tuple[pd.DataFrame, dict]:
    import pyarrow.parquet as pq
    cols = pq.read_schema(paths.ARTIFACTS / "samples" / "v2_frame.parquet").names
    mkt = [c for c in cols if c.startswith("mkt_") and c not in L.NEVER_CARRY]
    gmd = [c for c in cols if c.startswith("gmd_") and c not in L.NEVER_CARRY]
    fred = [c for c in cols if c.startswith("fred_")]
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["asof_date", "block", *mkt, *gmd, *fred])
    D = fr.groupby("asof_date").first().reset_index()
    lab = make_labels_v2(EB.read_daily("SPY")).rename(columns={"date": "asof_date"})
    D = D.merge(lab, on="asof_date", how="left")
    D = purge_per_horizon(D)
    spy = [c for c in mkt if c.startswith("mkt_spy_")]
    other = [c for c in mkt if not c.startswith("mkt_spy_")]
    sets = {"M1": spy, "M2": spy + other, "M3": spy + other + gmd, "M4": spy + other + gmd + fred}
    return D, sets


def ll_rows(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def fit_logistic(X, y, tr, inn, lam=None):
    """Median-impute and standardise on train; choose the per-row L2 penalty on inner unless given."""
    med = X[tr].median().fillna(0.0)          # a column with no training values carries no information
    f = X[tr].fillna(med)
    mu, sd = f.mean(), f.std().replace(0, 1)
    Z = lambda m: ((X[m].fillna(med) - mu) / sd).clip(-8, 8).to_numpy(np.float64)
    fits = {}
    with threadpool_limits(limits=1, user_api="blas"):
        cands = [lam] if lam is not None else PENALTIES
        for p in cands:
            m = LogisticRegression(C=1.0 / (p * tr.sum()), max_iter=2000, tol=1e-6).fit(Z(tr), y[tr].astype(int))
            score = float(ll_rows(y[inn].astype(int), m.predict_proba(Z(inn))).mean()) if inn.sum() else float("nan")
            fits[p] = (score, m)
    best = lam if lam is not None else min(fits, key=lambda p: fits[p][0])
    return fits[best][1], best, Z, {str(p): s for p, (s, _) in fits.items()}


def block_boot(per: list[tuple[np.ndarray, np.ndarray]], L_months: int, rng, n_boot=1000):
    """Group-mean gain with a moving-block bootstrap over the month sequence (circular runs of L consecutive months,
    the same months for every horizon in the group, as E11 does with one-month blocks). per = [(months, d), ...]."""
    um = np.unique(np.concatenate([m for m, _ in per]))
    S = np.stack([pd.Series(d).groupby(m).sum().reindex(um).fillna(0).to_numpy() for m, d in per])
    C = np.stack([pd.Series(d).groupby(m).size().reindex(um).fillna(0).to_numpy() for m, d in per])
    n = len(um)
    k = math.ceil(n / L_months)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, k)
        ix = ((starts[:, None] + np.arange(L_months)[None, :]) % n).ravel()[:n]
        draws[b] = np.mean(S[:, ix].sum(1) / np.maximum(C[:, ix].sum(1), 1))
    point = float(np.mean(S.sum(1) / C.sum(1)))
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    t0 = time.time()
    D, sets = date_frame()
    allc = sets["M4"]
    perm = np.random.default_rng(SEED).permutation(len(D))
    PL = D[allc].iloc[perm].reset_index(drop=True)
    PL.columns = [f"PL_{c}" for c in allc]
    D = pd.concat([D.reset_index(drop=True), PL], axis=1)
    sets["M4_placebo"] = list(PL.columns)
    horizons = list(L.REP_HORIZONS)
    blk = D.block.to_numpy()
    month = (D.asof_date.dt.year * 12 + D.asof_date.dt.month).to_numpy()
    run = Run("e19_market_model", {"sets": {k: len(v) for k, v in sets.items()}, "horizons": horizons,
                                   "penalties": PENALTIES, "lgb": LGB_PARAMS, "min_windows": MIN_WINDOWS,
                                   "bootstrap": "moving blocks of max(1, ceil(h/21)) months, 1000 draws", "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pred_dir = paths.artifact("predictions", "e19", "x").parent
    res, rows, last = {}, {}, {}
    for h in horizons:
        y = D[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y)
        tr, inn, va, te = (lab & (blk == b) for b in ("train", "inner", "valid", "test"))
        prior = np.bincount(y[tr].astype(int), minlength=3) / tr.sum()
        n_win = va.sum() / HORIZONS_V2[h]
        res[h] = {"n": {b: int(m.sum()) for b, m in zip(("train", "inner", "valid", "test"), (tr, inn, va, te))},
                  "valid_windows": round(float(n_win), 2), "in_keep_decision": bool(n_win >= MIN_WINDOWS),
                  "prior": prior.round(4).tolist()}
        for split, m in (("valid", va), ("test", te)):
            l = ll_rows(y[m].astype(int), np.tile(prior, (m.sum(), 1)))
            for mdl in ("lgbm", "logistic"):
                rows[(h, "P", mdl, split)] = (l, month[m])
            res[h].setdefault("P", {})[f"{split}_ll"] = float(l.mean())
        for name, cols in sets.items():
            X = D[cols]
            out = {}
            # LightGBM
            dtr = lgb.Dataset(X[tr], y[tr].astype(int), free_raw_data=False)
            if inn.sum():
                bst = lgb.train(LGB_PARAMS, dtr, 2000, valid_sets=[lgb.Dataset(X[inn], y[inn].astype(int), reference=dtr)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                trees = int(bst.best_iteration)
            else:
                trees = last.get((name, "trees"), 100)
                bst = lgb.train(LGB_PARAMS, dtr, trees)
            last[(name, "trees")] = trees
            out["lgbm"] = {"P": {s: bst.predict(X[m], num_iteration=trees) for s, m in (("valid", va), ("test", te))},
                           "trees": trees}
            # logistic
            lr, lam, Z, inner_ll = fit_logistic(X, y, tr, inn, None if inn.sum() else last.get((name, "lam"), 1.0))
            last[(name, "lam")] = lam
            with threadpool_limits(limits=1, user_api="blas"):
                out["logistic"] = {"P": {s: lr.predict_proba(Z(m)) for s, m in (("valid", va), ("test", te))},
                                   "penalty": lam, "inner_ll_by_penalty": inner_ll}
            for mdl, o in out.items():
                r = {k: v for k, v in o.items() if k != "P"}
                for split, m in (("valid", va), ("test", te)):
                    yy = y[m].astype(int)
                    l = ll_rows(yy, o["P"][split])
                    rows[(h, name, mdl, split)] = (l, month[m])
                    r[f"{split}_ll"] = float(l.mean())
                    r[f"{split}_auroc_ud"] = L.auroc_ud(yy, o["P"][split])
                    np.save(pred_dir / f"{h}_{name}_{mdl}_{split}.npy", o["P"][split].astype(np.float32))
                res[h].setdefault(name, {})[mdl] = r
        print(f"{h}: prior valid {res[h]['P']['valid_ll']:.4f} | " + " | ".join(
            f"{n} {res[h][n]['lgbm']['valid_ll']:.4f}/{res[h][n]['logistic']['valid_ll']:.4f}" for n in sets)
              + f" | windows {n_win:.1f} ({time.time() - t0:.0f}s)", flush=True)
    # gains with horizon-length blocks
    rng = np.random.default_rng(SEED)
    comps = [("M1", "P"), ("M2", "M1"), ("M3", "M2"), ("M4", "M3"), ("M4", "P"), ("M4_placebo", "P")]
    gains = {}
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in L.GROUPS.items():
                    hs = [h for h in hs if h in horizons and (split == "test" or res[h]["in_keep_decision"])]
                    if not hs:
                        continue
                    blen = lambda hh: max(1, math.ceil(HORIZONS_V2[hh] / 21))
                    per, per_h = [], {}
                    for h in hs:
                        lo_, mo = rows[(h, old, mdl, split)]
                        ln_, _ = rows[(h, new, mdl, split)]
                        per.append((mo, lo_ - ln_))
                        pt, ci = block_boot([(mo, lo_ - ln_)], blen(h), rng)
                        per_h[h] = {"gain": pt, "ci95": ci, "block_months": blen(h)}
                    # group: joint resampling with the longest member's block length (conservative)
                    mean, ci = block_boot(per, max(blen(h) for h in hs), rng)
                    gains[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": mean, "ci95": ci, "horizons": hs,
                                                                  "block_months": max(blen(h) for h in hs),
                                                                  "per_horizon": per_h}
    keep = {f"{m}|{new}": L.keep_decision(gains, m, new, old)
            for m in ("lgbm", "logistic") for new, old in comps if not new.endswith("placebo")}
    placebo_ok = {m: not L.keep_decision(gains, m, "M4_placebo", "P") for m in ("lgbm", "logistic")}
    metrics = {"by_horizon": res, "gains": gains, "keep_rule_decisions": keep, "placebo_shows_no_gain": placebo_ok,
               "sets": sets, "elapsed_s": round(time.time() - t0, 1)}
    summ = "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", f"E19 market-direction model. Keep rule (validation): {summ}. Placebo shows no gain: "
                            f"{placebo_ok}", metrics)
    print(summ, "| placebo ok:", placebo_ok)
    print(json.dumps({k: (round(v["gain"], 5), [round(x, 5) for x in v["ci95"]]) for k, v in gains.items()
                      if "|valid|" in k}, indent=0)[:4000])


if __name__ == "__main__":
    main()

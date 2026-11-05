"""E11 macro ladder (pre-registered in docs/V2_ARCHITECTURE.md section 4).

Nested feature sets on the v2 frame, every v2 horizon, two model families (multinomial logistic, LightGBM):
  F0 v1.1 numeric (80) | F1 + rel_* | F2 + mkt_* | F3 + gmd_* | F4 + fred_* (only if present in the frame)
plus the control F3-placebo: whole-date permutation of the date-level blocks (mkt_*, gmd_*, fred_*), which must remove
any gain that comes from real market/macro information rather than from fitting date effects.

Training: block 'train' (2003-2015) only; LightGBM early-stops on 'inner' (2016) only; logistic L2 = 1e-3 per row.
Selection evidence: 'valid' (2017-2018). Diagnostic: 'test' (2020-2022, reused). Month-block bootstrap (1,000 draws) of
the mean log-loss gain per horizon group: 1D (intraday), 2D-20D (swing), 3M-1Y (long-term).
Keep rule: a block is kept if the group-mean validation gain over the previous rung has a CI lower bound > 0 in at least
one group and a CI upper bound >= 0 in every group.
"""
from __future__ import annotations

import argparse
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

import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

E07_PARAMS = {"learning_rate": 0.07826968053857598, "num_leaves": 7, "min_data_in_leaf": 195,
              "feature_fraction": 0.7580871652599147, "bagging_fraction": 0.8317004666751824,
              "lambda_l2": 0.03388212372550781}
LGB_BASE = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "bagging_freq": 1,
            "seed": 7, "deterministic": True, "force_row_wise": True}
GROUPS = {"intraday": ["1D"], "swing": [f"{h}D" for h in range(2, 21)], "long": ["3M", "6M", "1Y"]}
N_BOOT, SEED = 1000, 11


def ll_rows(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def auroc_ud(y, P):
    m = y != 1
    if m.sum() < 10:
        return float("nan")
    s, t = P[m, 2] - P[m, 0], (y[m] == 2)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = t.sum(), (~t).sum()
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--horizons", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    num = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    rel = [c for c in fr.columns if c.startswith("rel_")]
    mkt = [c for c in fr.columns if c.startswith("mkt_")]
    gmd = [c for c in fr.columns if c.startswith("gmd_") and c != "gmd_data_year"]
    fred = [c for c in fr.columns if c.startswith("fred_")]
    sets = {"F0": num, "F1": num + rel, "F2": num + rel + mkt, "F3": num + rel + mkt + gmd}
    if fred:
        sets["F4"] = sets["F3"] + fred
    date_level = mkt + gmd + fred
    # placebo: permute whole dates of the date-level blocks (same permutation in every block)
    dates = np.sort(fr.asof_date.unique())
    perm = dict(zip(dates, np.random.default_rng(SEED).permutation(dates)))
    dl = fr.groupby("asof_date")[date_level].first()
    plac = dl.loc[[perm[d] for d in fr.asof_date]].reset_index(drop=True)
    plac.columns = [f"PL_{c}" for c in date_level]
    fr = pd.concat([fr, plac], axis=1)
    top = "F4" if fred else "F3"
    sets[f"{top}_placebo"] = num + rel + [f"PL_{c}" for c in date_level]
    horizons = a.horizons or list(HORIZONS_V2)
    run = Run("e11_macro_ladder", {"sets": {k: len(v) for k, v in sets.items()}, "horizons": horizons,
                                   "lgb": E07_PARAMS, "logistic_l2_per_row": 1e-3, "n_boot": N_BOOT,
                                   "placebo": "whole-date permutation of date-level blocks"},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    blk = fr.block.to_numpy()
    month = (fr.asof_date.dt.year * 12 + fr.asof_date.dt.month).to_numpy()
    res, rows_ll = {}, {}
    pred_dir = paths.artifact("predictions", "e11", "x").parent
    for h in horizons:
        y_all = fr[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y_all)
        for name, cols in sets.items():
            X = fr[cols]
            tr, inn, va, te = (lab & (blk == b) for b in ("train", "inner", "valid", "test"))
            ytr, yin, yva, yte = (y_all[m].astype(int) for m in (tr, inn, va, te))
            out = {}
            # LightGBM
            cat = ["rel_sector_id"] if "rel_sector_id" in cols else "auto"
            dtr = lgb.Dataset(X[tr], ytr, free_raw_data=False, categorical_feature=cat)
            din = lgb.Dataset(X[inn], yin, reference=dtr)
            bst = lgb.train({**LGB_BASE, **E07_PARAMS, "num_threads": a.threads}, dtr, num_boost_round=3000,
                            valid_sets=[din], valid_names=["inner"], callbacks=[lgb.early_stopping(100, verbose=False)])
            P = {"valid": bst.predict(X[va], num_iteration=bst.best_iteration),
                 "test": bst.predict(X[te], num_iteration=bst.best_iteration)}
            out["lgbm"] = {"trees": int(bst.best_iteration), "P": P}
            # logistic (standardise on train; impute train medians; the sector id is categorical, so it is left out)
            X = X[[c for c in cols if c != "rel_sector_id"]]
            med = X[tr].median()
            mu, sd = X[tr].fillna(med).mean(), X[tr].fillna(med).std().replace(0, 1)
            Z = lambda m: ((X[m].fillna(med) - mu) / sd).clip(-8, 8).to_numpy(np.float64)
            lr = LogisticRegression(C=1.0 / (1e-3 * tr.sum()), max_iter=400, tol=1e-6)
            lr.fit(Z(tr), ytr)
            out["logistic"] = {"P": {"valid": lr.predict_proba(Z(va)), "test": lr.predict_proba(Z(te))}}
            for mdl, o in out.items():
                r = {}
                for split, yy, mm in (("valid", yva, va), ("test", yte, te)):
                    l = ll_rows(yy, o["P"][split])
                    rows_ll[(h, name, mdl, split)] = (l, month[mm])
                    r[f"{split}_ll"] = float(l.mean())
                    r[f"{split}_auroc_ud"] = auroc_ud(yy, o["P"][split])
                    np.save(pred_dir / f"{h}_{name}_{mdl}_{split}.npy", o["P"][split].astype(np.float32))
                if mdl == "lgbm":
                    r["trees"] = o["trees"]
                res.setdefault(h, {}).setdefault(name, {})[mdl] = r
            print(f"{h} {name}: lgbm valid {res[h][name]['lgbm']['valid_ll']:.5f} logit valid "
                  f"{res[h][name]['logistic']['valid_ll']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    # ladder gains with month-block bootstrap per horizon group
    order = [k for k in sets if not k.endswith("placebo")]
    comps = [(order[i], order[i - 1]) for i in range(1, len(order))] + [(f"{top}_placebo", "F1"), (top, "F0")]
    rng = np.random.default_rng(SEED)
    ladder = {}
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in GROUPS.items():
                    hs = [h for h in hs if h in horizons]
                    if not hs:
                        continue
                    # per-horizon month sums of the loss difference (old - new: positive = new is better)
                    per = []
                    for h in hs:
                        lo, mo = rows_ll[(h, old, mdl, split)]
                        ln, _ = rows_ll[(h, new, mdl, split)]
                        s = pd.DataFrame({"m": mo, "d": lo - ln}).groupby("m").d.agg(["sum", "count"])
                        per.append(s)
                    months = np.unique(np.concatenate([p.index.to_numpy() for p in per]))
                    S = np.stack([p["sum"].reindex(months).fillna(0).to_numpy() for p in per])
                    C = np.stack([p["count"].reindex(months).fillna(0).to_numpy() for p in per])
                    point = float(np.mean(S.sum(1) / C.sum(1)))
                    draws = []
                    for _ in range(N_BOOT):
                        ix = rng.integers(0, len(months), len(months))
                        draws.append(np.mean(S[:, ix].sum(1) / np.maximum(C[:, ix].sum(1), 1)))
                    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
                    ladder[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": point, "ci95": [float(lo95), float(hi95)],
                                                                    "n_months": int(len(months)), "horizons": len(hs)}
    keep = {}
    for mdl in ("lgbm", "logistic"):
        for i in range(1, len(order)):
            new, old = order[i], order[i - 1]
            g = [ladder.get(f"{mdl}|{new}_vs_{old}|valid|{gg}") for gg in GROUPS]
            g = [x for x in g if x]
            keep[f"{mdl}|{new}"] = bool(any(x["ci95"][0] > 0 for x in g) and all(x["ci95"][1] >= 0 for x in g))
    metrics = {"by_horizon": res, "ladder": ladder, "keep_rule_decisions": keep,
               "sets": {k: v for k, v in sets.items()}, "elapsed_s": round(time.time() - t0, 1)}
    summ = "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", f"E11 ladder over {len(horizons)} horizons. Keep rule (validation): {summ}", metrics)
    print(summ)


if __name__ == "__main__":
    main()

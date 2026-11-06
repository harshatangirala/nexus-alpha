"""E20 classic market-regime rules (pre-registered in docs/V2_ARCHITECTURE.md section 4; decision D-034).

E19's design and code (one row per session, SPY's own label, logistic with the penalty chosen on inner 2016, small
LightGBM, horizon-length block bootstrap, >= 4 validation windows, keep rule) with the rule block R from
`nexus_alpha.rules.market_regime` (distribution days, follow-through state, power trend; tradermonty/claude-trading-skills,
MIT). Comparisons: M2 + R vs M2, R vs P, placebo (R rows permuted across dates) vs P.
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
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

import e19_market_model as E19  # noqa: E402
from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import etf_bars as EB  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.rules.market_regime import regime_features  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEED = 11


def main() -> None:
    t0 = time.time()
    D, sets19 = E19.date_frame()
    R = regime_features(EB.read_daily("SPY"), EB.read_daily("QQQ"))
    D = D.merge(R.reset_index().rename(columns={"date": "asof_date"}), on="asof_date", how="left")
    rcols = list(R.columns)
    perm = np.random.default_rng(SEED).permutation(len(D))
    PL = D[rcols].iloc[perm].reset_index(drop=True)
    PL.columns = [f"PL_{c}" for c in rcols]
    D = pd.concat([D.reset_index(drop=True), PL], axis=1)
    sets = {"M2": sets19["M2"], "M2R": sets19["M2"] + rcols, "R": rcols, "R_placebo": list(PL.columns)}
    comps = [("M2R", "M2"), ("R", "P"), ("R_placebo", "P")]
    horizons = list(L.REP_HORIZONS)
    blk = D.block.to_numpy()
    month = (D.asof_date.dt.year * 12 + D.asof_date.dt.month).to_numpy()
    run = Run("e20_regime_rules", {"sets": {k: len(v) for k, v in sets.items()}, "rule_cols": rcols, "horizons": horizons,
                                   "penalties": E19.PENALTIES, "lgb": E19.LGB_PARAMS, "min_windows": E19.MIN_WINDOWS,
                                   "source": "tradermonty/claude-trading-skills d8e71ce (MIT)", "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pred_dir = paths.artifact("predictions", "e20", "x").parent
    res, rows, last = {}, {}, {}
    for h in horizons:
        y = D[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y)
        tr, inn, va, te = (lab & (blk == b) for b in ("train", "inner", "valid", "test"))
        prior = np.bincount(y[tr].astype(int), minlength=3) / tr.sum()
        n_win = va.sum() / HORIZONS_V2[h]
        res[h] = {"valid_windows": round(float(n_win), 2), "in_keep_decision": bool(n_win >= E19.MIN_WINDOWS)}
        for split, m in (("valid", va), ("test", te)):
            l = E19.ll_rows(y[m].astype(int), np.tile(prior, (m.sum(), 1)))
            for mdl in ("lgbm", "logistic"):
                rows[(h, "P", mdl, split)] = (l, month[m])
            res[h].setdefault("P", {})[f"{split}_ll"] = float(l.mean())
        for name, cols in sets.items():
            X = D[cols].astype(float)
            out = {}
            dtr = lgb.Dataset(X[tr], y[tr].astype(int), free_raw_data=False)
            if inn.sum():
                bst = lgb.train(E19.LGB_PARAMS, dtr, 2000, valid_sets=[lgb.Dataset(X[inn], y[inn].astype(int), reference=dtr)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                trees = int(bst.best_iteration)
            else:
                trees = last.get((name, "trees"), 100)
                bst = lgb.train(E19.LGB_PARAMS, dtr, trees)
            last[(name, "trees")] = trees
            out["lgbm"] = {s: bst.predict(X[m], num_iteration=trees) for s, m in (("valid", va), ("test", te))}
            lr, lam, Z, _ = E19.fit_logistic(X, y, tr, inn, None if inn.sum() else last.get((name, "lam"), 1.0))
            last[(name, "lam")] = lam
            with threadpool_limits(limits=1, user_api="blas"):
                out["logistic"] = {s: lr.predict_proba(Z(m)) for s, m in (("valid", va), ("test", te))}
            for mdl, P in out.items():
                r = {"penalty": lam} if mdl == "logistic" else {"trees": trees}
                for split, m in (("valid", va), ("test", te)):
                    yy = y[m].astype(int)
                    l = E19.ll_rows(yy, P[split])
                    rows[(h, name, mdl, split)] = (l, month[m])
                    r[f"{split}_ll"] = float(l.mean())
                    r[f"{split}_auroc_ud"] = L.auroc_ud(yy, P[split])
                    np.save(pred_dir / f"{h}_{name}_{mdl}_{split}.npy", P[split].astype(np.float32))
                res[h].setdefault(name, {})[mdl] = r
        print(f"{h}: prior {res[h]['P']['valid_ll']:.4f} | " + " | ".join(
            f"{n} {res[h][n]['lgbm']['valid_ll']:.4f}/{res[h][n]['logistic']['valid_ll']:.4f}" for n in sets), flush=True)
    rng = np.random.default_rng(SEED)
    gains = {}
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in L.GROUPS.items():
                    hs = [h for h in hs if h in horizons and (split == "test" or res[h]["in_keep_decision"])]
                    if not hs:
                        continue
                    per = [(rows[(h, old, mdl, split)][1], rows[(h, old, mdl, split)][0] - rows[(h, new, mdl, split)][0])
                           for h in hs]
                    blen = max(max(1, math.ceil(HORIZONS_V2[h] / 21)) for h in hs)
                    pt, ci = E19.block_boot(per, blen, rng)
                    gains[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": pt, "ci95": ci, "horizons": hs, "block_months": blen}
    keep = {f"{m}|{new}_vs_{old}": L.keep_decision(gains, m, new, old) for m in ("lgbm", "logistic")
            for new, old in comps if not new.endswith("placebo")}
    placebo_ok = {m: not L.keep_decision(gains, m, "R_placebo", "P") for m in ("lgbm", "logistic")}
    summ = "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", f"E20 regime rules. Keep rule (validation): {summ}. Placebo shows no gain: {placebo_ok}",
               {"by_horizon": res, "gains": gains, "keep_rule_decisions": keep, "placebo_shows_no_gain": placebo_ok,
                "sets": sets, "elapsed_s": round(time.time() - t0, 1)})
    print(summ, "| placebo ok:", placebo_ok)
    for k, v in gains.items():
        if "_placebo" not in k:
            print(k, round(v["gain"], 5), [round(x, 5) for x in v["ci95"]])


if __name__ == "__main__":
    main()

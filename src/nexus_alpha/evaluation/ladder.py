"""Shared engine for nested feature-set experiments (E11 macro ladder, E16 text encoders).

For each horizon and feature set: LightGBM (E07 parameters; trained on 'train', early-stopped on 'inner' only) and a
multinomial logistic regression (L2 1e-3 per row; train medians imputed; standardised on train). Scores on 'valid'
(selection) and 'test' (reused diagnostic). Gains between sets are averaged within horizon groups and given month-block
bootstrap 95% intervals. Predictions are saved for later blending.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

E07_PARAMS = {"learning_rate": 0.07826968053857598, "num_leaves": 7, "min_data_in_leaf": 195,
              "feature_fraction": 0.7580871652599147, "bagging_fraction": 0.8317004666751824,
              "lambda_l2": 0.03388212372550781}
LGB_BASE = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "bagging_freq": 1,
            "seed": 7, "deterministic": True, "force_row_wise": True}
GROUPS = {"intraday": ["1D"], "swing": [f"{h}D" for h in range(2, 21)], "long": ["3M", "6M", "1Y"]}


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


def fit_eval(fr: pd.DataFrame, sets: dict[str, list[str]], horizons: list[str], threads: int, pred_dir,
             log=print) -> tuple[dict, dict]:
    t0 = time.time()
    blk = fr.block.to_numpy()
    month = (fr.asof_date.dt.year * 12 + fr.asof_date.dt.month).to_numpy()
    res, rows_ll = {}, {}
    for h in horizons:
        y_all = fr[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y_all)
        tr, inn, va, te = (lab & (blk == b) for b in ("train", "inner", "valid", "test"))
        ytr, yin, yva, yte = (y_all[m].astype(int) for m in (tr, inn, va, te))
        for name, cols in sets.items():
            X = fr[cols]
            out = {}
            cat = ["rel_sector_id"] if "rel_sector_id" in cols else "auto"
            dtr = lgb.Dataset(X[tr], ytr, free_raw_data=False, categorical_feature=cat)
            din = lgb.Dataset(X[inn], yin, reference=dtr)
            bst = lgb.train({**LGB_BASE, **E07_PARAMS, "num_threads": threads}, dtr, num_boost_round=3000,
                            valid_sets=[din], valid_names=["inner"], callbacks=[lgb.early_stopping(100, verbose=False)])
            out["lgbm"] = {"trees": int(bst.best_iteration),
                           "P": {"valid": bst.predict(X[va], num_iteration=bst.best_iteration),
                                 "test": bst.predict(X[te], num_iteration=bst.best_iteration)}}
            Xl = X[[c for c in cols if c != "rel_sector_id"]]         # categorical id is not a linear feature
            med = Xl[tr].median()
            filled = Xl[tr].fillna(med)
            mu, sd = filled.mean(), filled.std().replace(0, 1)
            Z = lambda m: ((Xl[m].fillna(med) - mu) / sd).clip(-8, 8).to_numpy(np.float64)
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
                    r[f"{split}_n"] = int(len(yy))
                    np.save(pred_dir / f"{h}_{name}_{mdl}_{split}.npy", o["P"][split].astype(np.float32))
                if mdl == "lgbm":
                    r["trees"] = o["trees"]
                res.setdefault(h, {}).setdefault(name, {})[mdl] = r
            log(f"{h} {name}: lgbm valid {res[h][name]['lgbm']['valid_ll']:.5f} | logistic valid "
                f"{res[h][name]['logistic']['valid_ll']:.5f} ({time.time() - t0:.0f}s)")
    return res, rows_ll


def group_gains(rows_ll: dict, comps: list[tuple[str, str]], horizons: list[str], n_boot: int = 1000,
                seed: int = 11) -> dict:
    """Mean per-horizon log-loss gain (old - new; positive = new better) per horizon group, month-block bootstrap."""
    rng = np.random.default_rng(seed)
    out = {}
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in GROUPS.items():
                    hs = [h for h in hs if h in horizons]
                    if not hs:
                        continue
                    per = []
                    for h in hs:
                        lo, mo = rows_ll[(h, old, mdl, split)]
                        ln, _ = rows_ll[(h, new, mdl, split)]
                        per.append(pd.DataFrame({"m": mo, "d": lo - ln}).groupby("m").d.agg(["sum", "count"]))
                    months = np.unique(np.concatenate([p.index.to_numpy() for p in per]))
                    S = np.stack([p["sum"].reindex(months).fillna(0).to_numpy() for p in per])
                    C = np.stack([p["count"].reindex(months).fillna(0).to_numpy() for p in per])
                    point = float(np.mean(S.sum(1) / C.sum(1)))
                    draws = np.empty(n_boot)
                    for b in range(n_boot):
                        ix = rng.integers(0, len(months), len(months))
                        draws[b] = np.mean(S[:, ix].sum(1) / np.maximum(C[:, ix].sum(1), 1))
                    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
                    out[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": point, "ci95": [float(lo95), float(hi95)],
                                                                 "n_months": int(len(months)), "horizons": len(hs)}
    return out


def keep_decision(gains: dict, mdl: str, new: str, old: str) -> bool:
    """Pre-registered keep rule: CI lower bound > 0 in >= 1 group and CI upper bound >= 0 in every group (validation)."""
    g = [gains.get(f"{mdl}|{new}_vs_{old}|valid|{gg}") for gg in GROUPS]
    g = [x for x in g if x]
    return bool(g) and any(x["ci95"][0] > 0 for x in g) and all(x["ci95"][1] >= 0 for x in g)


def carried_feature_set(experiments_dir) -> tuple[list[str], str]:
    """Feature columns carried forward by the latest completed E11 run: F0 plus every block that passed the keep rule
    for a model family whose date-permutation placebo also passed (docs/V2_ARCHITECTURE.md, E11 clarification)."""
    import json
    from pathlib import Path
    for r in sorted(Path(experiments_dir).glob("*_e11_macro_ladder_*"), reverse=True):
        man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
        if man.get("status") != "COMPLETED":
            continue
        m = json.loads((r / "metrics.json").read_text(encoding="utf-8"))
        sets, keep, plac = m["sets"], m["keep_rule_decisions"], m["placebo_shows_no_gain"]
        order = [k for k in sets if not k.endswith("placebo")]
        cols = list(sets["F0"])
        for i in range(1, len(order)):
            if any(keep.get(f"{f}|{order[i]}") and plac.get(f) for f in ("lgbm", "logistic")):
                cols += [c for c in sets[order[i]] if c not in sets[order[i - 1]]]
        return cols, r.name
    raise FileNotFoundError("no completed E11 run")

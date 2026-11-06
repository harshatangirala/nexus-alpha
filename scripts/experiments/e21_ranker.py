"""E21 cross-sectional ranker (pre-registered in docs/V3_RESEARCH.md; decision D-037).

Candidates R1 (LightGBM regression, F0), R2 (regression, F0 + rel_*), R3 (lambdarank by date, F0 + rel_*) against the
baseline B0 (E18's selection-stage F0 LightGBM classifier score p_up - p_down). Target: within-date rank-Gaussianised
h-session return. Metric: mean daily Spearman rank IC on validation with a moving-block bootstrap; keep rule on the paired
daily IC difference vs B0 per horizon group. Economic diagnostic: within-date top-minus-bottom quintile spread.
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
from scipy.stats import norm

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

HZ = ["1D", "5D", "10D", "20D", "3M"]
GROUPS = {"intraday": ["1D"], "swing": ["5D", "10D", "20D"], "long": ["3M"]}
COST_PER_LEG = 0.0010
MIN_N = 10
SEED = 11


def rank_gauss(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("asof_date")[col]
    n = g.transform("count")
    r = g.rank(method="average")
    out = pd.Series(norm.ppf((r - 0.5) / n), index=df.index)
    return out.where(n >= MIN_N)


def daily_ic(dates, score, ret) -> pd.Series:
    d = pd.DataFrame({"d": dates, "s": score, "r": ret}).dropna()
    g = d.groupby("d")
    return g.apply(lambda x: x.s.rank().corr(x.r.rank()) if len(x) >= MIN_N else np.nan).dropna()


def block_ci(series_by_date: list[pd.Series], L_months: int, rng, n_boot=1000):
    """Mean over horizons of each horizon's mean daily value; moving blocks of months, same months for all."""
    months = sorted(set().union(*[set(s.index.year * 12 + s.index.month) for s in series_by_date]))
    um = np.array(months)
    S = np.stack([pd.Series(s.values).groupby(s.index.year * 12 + s.index.month).sum().reindex(um).fillna(0).values for s in series_by_date])
    C = np.stack([pd.Series(s.values).groupby(s.index.year * 12 + s.index.month).size().reindex(um).fillna(0).values for s in series_by_date])
    point = float(np.mean(S.sum(1) / np.maximum(C.sum(1), 1)))
    n, k = len(um), math.ceil(len(um) / L_months)
    dr = []
    for _ in range(n_boot):
        st = rng.integers(0, n, k)
        ix = ((st[:, None] + np.arange(L_months)[None, :]) % n).ravel()[:n]
        dr.append(np.mean(S[:, ix].sum(1) / np.maximum(C[:, ix].sum(1), 1)))
    return point, [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]


def quintile_spread(dates, score, ret) -> pd.Series:
    d = pd.DataFrame({"d": dates, "s": score, "r": ret}).dropna()

    def one(x):
        if len(x) < MIN_N:
            return np.nan
        q = x.s.rank(pct=True)
        return x.r[q > 0.8].mean() - x.r[q <= 0.2].mean() - 2 * COST_PER_LEG
    return d.groupby("d").apply(one).dropna()


def main() -> None:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    f0 = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    rel = [c for c in fr.columns if c.startswith("rel_")]
    sets = {"R1": f0, "R2": f0 + rel, "R3": f0 + rel}
    from nexus_alpha.evaluation.ladder import E07_PARAMS
    base = {"verbosity": -1, "seed": 7, "deterministic": True, "force_row_wise": True, "bagging_freq": 1,
            "num_threads": 8, **E07_PARAMS}
    blk = fr.block.to_numpy()
    run = Run("e21_ranker", {"horizons": HZ, "sets": {k: len(v) for k, v in sets.items()}, "lgb": base,
                             "cost_per_leg": COST_PER_LEG, "min_n": MIN_N, "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pred_dir = paths.artifact("predictions", "e21", "x").parent
    res, ics, spreads, trees = {}, {}, {}, {}
    for h in HZ:
        ret = fr[f"ret_{h}"]
        tgt = rank_gauss(fr.assign(_r=ret), "_r")
        lab = fr[f"y_{h}"].notna().to_numpy()
        tr, inn, va, te = (lab & tgt.notna().to_numpy() & (blk == b) for b in ("train", "inner", "valid", "test"))
        # B0 on the same rows: E18 predictions are stored for labelled valid/test rows in frame order
        b0 = {}
        for split, m in (("valid", va), ("test", te)):
            m_all = lab & (blk == split)
            P = np.load(paths.ARTIFACTS / "predictions" / "e18" / f"{h}_FINAL_lgbm_{split}.npy")
            s = pd.Series(np.nan, index=fr.index)
            s.iloc[np.flatnonzero(m_all)] = P[:, 2] - P[:, 0]
            b0[split] = s.to_numpy()
        scores = {"B0": b0}
        for name, cols in sets.items():
            X = fr[cols]
            cat = ["rel_sector_id"] if "rel_sector_id" in cols else "auto"
            if name == "R3":
                q = fr.groupby("asof_date")["_r" if False else f"ret_{h}"].rank(pct=True)
                rel_lab = np.clip((q * 5).apply(np.ceil) - 1, 0, 4)

                def ds(m, ref=None):
                    idx = np.flatnonzero(m)
                    order = idx[np.argsort(fr.asof_date.to_numpy()[idx], kind="stable")]
                    grp = fr.asof_date.iloc[order].value_counts(sort=False).reindex(pd.unique(fr.asof_date.iloc[order])).to_numpy()
                    return lgb.Dataset(X.iloc[order], rel_lab.iloc[order].astype(int), group=grp, categorical_feature=cat,
                                       reference=ref, free_raw_data=False)
                dtr = ds(tr)
                prm = {**base, "objective": "lambdarank", "metric": "ndcg", "eval_at": [10]}
                bst = lgb.train(prm, dtr, 3000, valid_sets=[ds(inn, dtr)], callbacks=[lgb.early_stopping(100, verbose=False)])
            else:
                dtr = lgb.Dataset(X[tr], tgt[tr], categorical_feature=cat, free_raw_data=False)
                prm = {**base, "objective": "regression", "metric": "l2"}
                bst = lgb.train(prm, dtr, 3000, valid_sets=[lgb.Dataset(X[inn], tgt[inn], reference=dtr)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            trees[(h, name)] = int(bst.best_iteration)
            s = np.full(len(fr), np.nan)
            for m in (va, te):
                s[m] = bst.predict(X[m], num_iteration=bst.best_iteration)
            scores[name] = {"valid": s, "test": s}
            np.save(pred_dir / f"{h}_{name}_valid_test.npy", s.astype(np.float32))
        res[h] = {}
        for name, sc in scores.items():
            for split, m in (("valid", va), ("test", te)):
                ic = daily_ic(fr.asof_date[m].to_numpy(), sc[split][m], ret[m].to_numpy())
                sp = quintile_spread(fr.asof_date[m].to_numpy(), sc[split][m], ret[m].to_numpy())
                ic.index, sp.index = pd.DatetimeIndex(ic.index), pd.DatetimeIndex(sp.index)
                ics[(h, name, split)], spreads[(h, name, split)] = ic, sp
                res[h].setdefault(name, {})[split] = {"mean_ic": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()),
                                                      "dates": int(len(ic)), "mean_spread_net": float(sp.mean())}
            if name != "B0":
                res[h][name]["trees"] = trees[(h, name)]
        print(f"{h}: " + " | ".join(f"{n} IC {res[h][n]['valid']['mean_ic']:+.4f}/{res[h][n]['test']['mean_ic']:+.4f}"
                                    for n in scores) + f"  ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    blen = lambda hs: max(max(1, math.ceil(HORIZONS_V2[h] / 21)) for h in hs)
    gains, abs_ic, spr = {}, {}, {}
    for name in ["R1", "R2", "R3"]:
        for split in ("valid", "test"):
            for g, hs in GROUPS.items():
                diffs = [(ics[(h, name, split)] - ics[(h, "B0", split)]).dropna() for h in hs]
                gains[f"{name}_vs_B0|{split}|{g}"] = dict(zip(("gain", "ci95"), block_ci(diffs, blen(hs), rng)))
    for name in ["B0", "R1", "R2", "R3"]:
        for split in ("valid", "test"):
            for h in HZ:
                L_ = max(1, math.ceil(HORIZONS_V2[h] / 21))
                abs_ic[f"{name}|{split}|{h}"] = dict(zip(("ic", "ci95"), block_ci([ics[(h, name, split)]], L_, rng)))
                spr[f"{name}|{split}|{h}"] = dict(zip(("spread_net", "ci95"), block_ci([spreads[(h, name, split)]], L_, rng)))
    keep = {n: L.keep_decision({k.replace(f"{n}_vs_B0", "lgbm|new_vs_old"): v for k, v in gains.items() if k.startswith(n)},
                               "lgbm", "new", "old") for n in ("R1", "R2", "R3")}
    passed = [n for n, v in keep.items() if v]
    chosen = max(passed, key=lambda n: np.mean([res[h][n]["valid"]["mean_ic"] for h in HZ])) if passed else "B0"
    summ = f"E21 ranker: keep vs B0 {keep}; chosen {chosen}"
    run.finish("COMPLETED", summ, {"by_horizon": res, "ic_gain_vs_B0": gains, "abs_ic": abs_ic, "quintile_spread": spr,
                                   "keep_vs_B0": keep, "chosen": chosen, "sets": sets,
                                   "trees": {f"{h}|{n}": t for (h, n), t in trees.items()}, "elapsed_s": round(time.time() - t0, 1)})
    print(summ)
    for k, v in gains.items():
        print(k, round(v["gain"], 4), [round(x, 4) for x in v["ci95"]])
    for k, v in spr.items():
        if "|valid|" in k or k.startswith(chosen):
            print("spread", k, round(v["spread_net"] * 1e4, 1), "bp", [round(x * 1e4, 1) for x in v["ci95"]])


if __name__ == "__main__":
    main()

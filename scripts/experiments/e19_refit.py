"""E19 deployment (decision D-031): refit the logistic M2 market model on 2003-2018 at the gated horizons and write a
bundle for the Trade Desk and the forward shadow test.

Gate (recomputed here from the saved E19 selection-stage predictions, and required to reproduce D-031's horizons): the
horizon entered E19's keep decision (>= 4 non-overlapping validation windows) and the validation up-vs-down AUROC 95%
interval (moving blocks of max(1, ceil(h/21)) months, 1000 draws, seed 11) lies above 0.5. Each refit keeps the penalty
chosen on inner 2016 in E19; medians and scaling are recomputed on the refit rows.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

import e19_market_model as E19  # noqa: E402
from nexus_alpha import paths  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2, LABEL_VERSION_V2  # noqa: E402

SET, FAMILY = "M2", "logistic"
D031_HORIZONS = ["20D"]


def latest_e19() -> tuple[str, dict]:
    for r in sorted((REPO / "experiments").glob("*_e19_market_model_*"), reverse=True):
        man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
        if man.get("status") == "COMPLETED":
            return r.name, json.loads((r / "metrics.json").read_text(encoding="utf-8"))
    raise FileNotFoundError("no completed E19 run")


def auroc_ci(y, P, months, L_months, rng, n=1000):
    um = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in um}
    k = math.ceil(len(um) / L_months)
    draws = []
    for _ in range(n):
        st = rng.integers(0, len(um), k)
        ix = ((st[:, None] + np.arange(L_months)[None, :]) % len(um)).ravel()[:len(um)]
        r = np.concatenate([idx[um[i]] for i in ix])
        a = L.auroc_ud(y[r], P[r])
        if a == a:
            draws.append(a)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else [float("nan")] * 2


def main() -> None:
    t0 = time.time()
    e19_id, m = latest_e19()
    D, sets = E19.date_frame()
    cols = sets[SET]
    assert cols == m["sets"][SET], "E19 M2 columns changed since the E19 run"
    blk = D.block.to_numpy()
    month = (D.asof_date.dt.year * 12 + D.asof_date.dt.month).to_numpy()
    rng = np.random.default_rng(11)
    evidence, gated = {}, []
    for h in L.REP_HORIZONS:
        y = D[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y)
        ev = {}
        for split in ("valid", "test"):
            mm = lab & (blk == split)
            P = np.load(paths.ARTIFACTS / "predictions" / "e19" / f"{h}_{SET}_{FAMILY}_{split}.npy").astype(float)
            yy = y[mm].astype(int)
            ev[f"{split}_auroc_ud"] = L.auroc_ud(yy, P)
            ev[f"{split}_auroc_ci95"] = auroc_ci(yy, P, month[mm], max(1, math.ceil(HORIZONS_V2[h] / 21)), rng)
            ev[f"{split}_ll"] = m["by_horizon"][h][SET][FAMILY][f"{split}_ll"]
            ev[f"{split}_ll_prior"] = m["by_horizon"][h]["P"][f"{split}_ll"]
        ev["in_keep_decision"] = m["by_horizon"][h]["in_keep_decision"]
        ev["direction_skill_validated"] = bool(ev["in_keep_decision"] and ev["valid_auroc_ci95"][0] > 0.5)
        evidence[h] = ev
        if ev["direction_skill_validated"]:
            gated.append(h)
        print(h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}, flush=True)
    if gated != D031_HORIZONS:
        sys.exit(f"gate gives {gated}, D-031 recorded {D031_HORIZONS}: stop and investigate")
    run = Run("e19_refit", {"e19_run": e19_id, "set": SET, "family": FAMILY, "horizons": gated,
                            "refit_blocks": ["train", "inner", "valid"]},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    models, prior = {}, {}
    for h in gated:
        y = D[f"y_{h}"].to_numpy()
        rows = ~np.isnan(y) & np.isin(blk, ["train", "inner", "valid"])
        X = D.loc[rows, cols]
        med = X.median().fillna(0.0)
        f = X.fillna(med)
        mu, sd = f.mean(), f.std().replace(0, 1)
        Z = ((f - mu) / sd).clip(-8, 8).to_numpy(np.float64)
        pen = float(m["by_horizon"][h][SET][FAMILY]["penalty"])
        with threadpool_limits(limits=1, user_api="blas"):
            lr = LogisticRegression(C=1.0 / (pen * rows.sum()), max_iter=2000, tol=1e-6).fit(Z, y[rows].astype(int))
        models[h] = {"model": lr, "cols": cols, "median": med, "mu": mu, "sd": sd, "penalty": pen,
                     "n_refit": int(rows.sum())}
        prior[h] = (np.bincount(y[rows].astype(int), minlength=3) / rows.sum()).tolist()
        print(f"{h}: refit on {rows.sum()} sessions, penalty {pen}", flush=True)
    with open(bdir / "market_model.pkl", "wb") as fh:
        pickle.dump(models, fh)
    meta = {"bundle_id": run.run_id, "kind": "market", "recipe": "E19 logistic M2, refit 2003-2018 (D-031)",
            "e19_run": e19_id, "target": "SPY v2 label (vol-scaled 3-class)", "label_version": LABEL_VERSION_V2,
            "horizons": {h: HORIZONS_V2[h] for h in gated}, "trained_through_block": "valid", "cols": cols,
            "refit_class_prior": prior, "evidence": {h: evidence[h] for h in gated}, "all_horizon_evidence": evidence}
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    run.finish("COMPLETED", f"E19 market model refit for {gated} (D-031)", {"evidence": evidence, "bundle": run.run_id,
                                                                          "elapsed_s": round(time.time() - t0, 1)})
    print("bundle", run.run_id)


if __name__ == "__main__":
    main()

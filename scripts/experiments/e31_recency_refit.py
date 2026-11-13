"""E31 (D-062): v1.2 = the frozen v1.1 recipe refit on 2003 to 2021-11-30 and calibrated on 2022. Nothing is chosen:
every setting is read from the v1.1 bundle and its source runs (E06, E07, E08). Pre-registered in
docs/E31_PREREGISTRATION.md; judged only in the forward test.

    .venv/Scripts/python -I scripts/experiments/e31_recency_refit.py

Writes artifacts/bundles/<run>/ (same layout as v1.1, so the inference pipeline loads it unchanged) and
experiments/<run>/metrics.json. Reads nothing on or after the seal (the s500 frame ends 2022-12-01).
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import ARCHITECTURE_VERSION, paths  # noqa: E402
from nexus_alpha.calibration.scalers import fit_best  # noqa: E402
from nexus_alpha.evaluation.metrics import log_loss  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.multimodal.dataset import build, load_tables  # noqa: E402
from nexus_alpha.multimodal.fusion import FusionConfig  # noqa: E402
from nexus_alpha.multimodal.trainer import predict, train  # noqa: E402
from nexus_alpha.targets import HORIZONS  # noqa: E402
from nexus_alpha.training.tabular import LGB_PARAMS  # noqa: E402

H = list(HORIZONS)
FULL = ("img_W", "img_D", "img_H", "num", "news", "sec")
SEEDS = (7, 11, 23, 42, 101)
LAM = 1e-3
V11 = "20261106T020411_e08_final_model_5c4327"
REFIT_END = pd.Timestamp("2021-11-30")
CAL_START, CAL_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")


def main() -> int:
    torch.set_num_threads(5)
    t0 = time.time()
    m11 = json.loads((paths.ARTIFACTS / "bundles" / V11 / "bundle_meta.json").read_text(encoding="utf-8"))
    e08 = json.loads((REPO / "experiments" / V11 / "manifest.json").read_text(encoding="utf-8"))["config"]
    e06m = json.loads((REPO / "experiments" / e08["e06"] / "metrics.json").read_text(encoding="utf-8"))
    e07m = json.loads((REPO / "experiments" / e08["e07"] / "metrics.json").read_text(encoding="utf-8"))
    comp = m11["components"]
    wl, wf, wg = comp["lgbm_weight"], comp["fusion_weight"], comp["logistic_weight"]
    assert m11["recipe"] == "blend_l0.1_f0.6_g0.3" and (wl, wf, wg) == (0.1, 0.6, 0.3), m11["recipe"]

    fr = load_tables()
    blk = fr.block.to_numpy()
    dates = pd.to_datetime(fr.asof_date).to_numpy()
    assert pd.Timestamp(dates.max()) < pd.Timestamp("2023-01-03"), "the frame must end before the seal"
    full, pre = build(fr, blk == "train", FULL, "convnext_tiny")
    Y = full.y
    labelled = np.isin(blk, ["train", "inner", "valid", "calib", "test"])
    refit_rows = labelled & (dates <= np.datetime64(REFIT_END))
    cal_rows = labelled & (dates >= np.datetime64(CAL_START)) & (dates <= np.datetime64(CAL_END))
    run = Run("e31_recency_refit", {"recipe": m11["recipe"], "source_bundle": V11, "e06": e08["e06"], "e07": e08["e07"],
                                     "refit_end": str(REFIT_END.date()), "calibration": "2022", "preregistration": "docs/E31_PREREGISTRATION.md"})
    print(f"refit rows {refit_rows.sum():,} (to {REFIT_END.date()}), calibration rows {cal_rows.sum():,} (2022)", flush=True)
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    components = {"lgbm_weight": wl, "fusion_weight": wf, "logistic_weight": wg, "fusion_modalities": list(FULL),
                  "fusion_seeds": [], "lgbm": {}}
    P = np.zeros((len(fr), len(H), 3))
    # ---- convex logistic (E09-certified objective, identical to E08)
    from sklearn.linear_model import LogisticRegression
    Xn = full.x["num"].astype(np.float64)
    log_models = {}
    for j, h in enumerate(H):
        m = refit_rows & (Y[:, j] >= 0)
        lr_ = LogisticRegression(C=1.0 / (LAM * m.sum()), max_iter=3000, tol=1e-8)
        lr_.fit(Xn[m], Y[m, j])
        P[:, j] += wg * lr_.predict_proba(Xn)
        log_models[h] = lr_
    with open(bdir / "logistic.pkl", "wb") as f:
        pickle.dump(log_models, f)
    components["logistic"] = {"file": "logistic.pkl", "lambda": LAM}
    print(f"logistic done ({time.time() - t0:.0f}s)", flush=True)
    # ---- LightGBM (E07 parameters; rounds scaled by rows from E07's inner-stopped iteration, as E08)
    feats = pre.num_cols
    for j, h in enumerate(H):
        m = refit_rows & (Y[:, j] >= 0)
        n_old = int(((blk == "train") & (Y[:, j] >= 0)).sum())
        rounds = max(1, int(round(e07m["curves"][h]["best_iteration_inner"] * m.sum() / n_old)))
        bst = lgb.train({**LGB_PARAMS, "num_threads": 5, **e07m["best_params"]}, lgb.Dataset(fr.loc[m, feats], Y[m, j]),
                        num_boost_round=rounds)
        P[:, j] += wl * bst.predict(fr[feats])
        bst.save_model(str(bdir / f"lgbm_{h}.txt"))
        components["lgbm"][h] = {"file": f"lgbm_{h}.txt", "rounds": rounds}
    print(f"lightgbm done ({time.time() - t0:.0f}s)", flush=True)
    # ---- gated fusion, 5 seeds, E06 settings and epoch counts (as E08)
    kw = e06m["selection"]["kw"]
    names = [m for m in full.names if m in FULL]
    d = type(full)({k: full.x[k] for k in names}, full.mask[:, [full.names.index(k) for k in names]], Y, full.z, names)
    Pf = []
    for s in SEEDS:
        ep = e06m[f"best_seed{s}"]["best_epoch"] + 1
        cfg = FusionConfig(modality_dims={k: v.shape[1] for k, v in d.x.items()}, variant="gated", d_model=kw.get("d_model", 64),
                           hidden=2 * kw.get("d_model", 64), dropout=kw.get("dropout", 0.1), modality_dropout=0.15)
        model, _ = train(cfg, d.subset(np.flatnonzero(refit_rows)), d.subset(np.flatnonzero(cal_rows)),
                         paths.ARTIFACTS / "models" / run.run_id / f"refit_seed{s}", epochs=ep, batch=2048, lr=kw["lr"],
                         weight_decay=kw.get("wd", 1e-4), patience=10 ** 6, seed=s, log=lambda x: None, schedule="cosine",
                         restore_best=False)
        P_s, _, _ = predict(model, d)
        Pf.append(P_s)
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, bdir / f"fusion_seed{s}.pt")
        components["fusion_seeds"].append({"file": f"fusion_seed{s}.pt", "seed": s, "epochs": ep})
        print(f"fusion seed {s}: {ep} epochs ({time.time() - t0:.0f}s)", flush=True)
    P += wf * np.mean(Pf, axis=0)
    # ---- calibration on 2022 (E08's fit_best: none / temperature / vector per horizon)
    cal, cal_info, Pc = {}, {}, P.copy()
    for j, h in enumerate(H):
        m = cal_rows & (Y[:, j] >= 0)
        c, inf = fit_best(P[m, j], Y[m, j], np.argsort(dates[m], kind="stable"))
        cal[h], cal_info[h] = c, inf["chosen"]
        Pc[:, j] = c.transform(P[:, j])
    # descriptive only: in-sample fit on 2022 (the calibration year) and class shares of 2019 vs 2022
    desc = {}
    for j, h in enumerate(H):
        m = cal_rows & (Y[:, j] >= 0)
        m19 = labelled & (pd.DatetimeIndex(dates).year == 2019) & (Y[:, j] >= 0)
        desc[h] = {"calibration_year_log_loss_raw": float(log_loss(Y[m, j], P[m, j])),
                   "calibration_year_log_loss_calibrated": float(log_loss(Y[m, j], Pc[m, j])),
                   "class_share_2019": np.bincount(Y[m19, j], minlength=3).tolist(),
                   "class_share_2022": np.bincount(Y[m, j], minlength=3).tolist()}
    prior = {h: (np.bincount(Y[refit_rows & (Y[:, j] >= 0), j], minlength=3)
                 / max(1, (refit_rows & (Y[:, j] >= 0)).sum())).tolist() for j, h in enumerate(H)}
    with open(bdir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open(bdir / "calibrators.pkl", "wb") as f:
        pickle.dump(cal, f)
    for fn in ("news_tfidf_svd.pkl", "sec_tfidf_svd.pkl"):
        shutil.copy2(paths.ARTIFACTS / "features" / fn, bdir / fn)
    shutil.copy2(bdir / components["fusion_seeds"][0]["file"], bdir / "fusion.pt")
    skill = {h: {"valid_up_vs_down_auroc": None, "ci95": None, "validated": False,
                 "note": "no out-of-sample period left (E31, D-062): direction gate closed"} for h in H}
    meta = {"bundle_id": run.run_id, "architecture_version": ARCHITECTURE_VERSION + "+v1.2", "recipe": m11["recipe"],
            "components": components, "modalities": list(FULL), "backbone": "convnext_tiny", "horizons": HORIZONS,
            "trained_through_block": "refit 2003-2021-11-30, calibrated 2022 (E31)", "calibration": cal_info,
            "direction_skill": skill, "source_recipe_bundle": V11, "refit_class_prior": prior}
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    res = {"refit_rows": int(refit_rows.sum()), "calibration_rows": int(cal_rows.sum()), "calibration": cal_info,
           "descriptive": desc, "components": components}
    run.finish("COMPLETED", f"v1.2 recipe refit to {REFIT_END.date()}, calibrated on 2022; bundle {run.run_id}", res)
    print(f"bundle {run.run_id} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

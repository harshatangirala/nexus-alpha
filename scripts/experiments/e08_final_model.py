"""E08: final model assembly (post-registration v1.1; selection on validation, calibration on 2019, test scored once more as
a historical diagnostic).

Candidates (uncalibrated, trained on 2003-2015): E03 gated (1 seed), E06 best fusion 5-seed ensemble, E06 numbers-only
5-seed ensemble, E07 tuned LightGBM, and blends w * LightGBM + (1 - w) * fusion ensemble with w in {0, 0.1, ..., 1}.
Selection: lowest mean validation (2017-18) log loss over six horizons.
Calibration: none / temperature / vector per horizon on 2019 (static), compared with walk-forward recalibration (each test
year recalibrated on the previous year's samples whose labels had ended).
Deployed variant: the selected recipe refitted on 2003-2018 (LightGBM trees scaled by rows; fusion seeds trained for their
validation-stage best epoch count), calibrated on 2019. Valid/calib predictions for downstream threshold selection come
from the 2003-2015 models (the refit has seen 2017-18).
Outputs: experiments/<run>/metrics.json, artifacts/bundles/<run>/ (the app's bundle), predictions for E04 v1.1.
Run: .venv/Scripts/python -I scripts/experiments/e08_final_model.py <e03_run> <e06_run> <e07_run>
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from nexus_alpha import ARCHITECTURE_VERSION, paths
from nexus_alpha.calibration.scalers import fit_best
from nexus_alpha.evaluation.metrics import classification_report, log_loss
from nexus_alpha.evaluation.stats import block_bootstrap, month_blocks
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.multimodal.dataset import build, load_tables
from nexus_alpha.multimodal.fusion import FusionConfig
from nexus_alpha.multimodal.trainer import predict, train
from nexus_alpha.targets import HORIZONS
from nexus_alpha.training.tabular import LGB_PARAMS

H = list(HORIZONS)
FULL = ("img_W", "img_D", "img_H", "num", "news", "sec")
SEEDS = (7, 11, 23, 42, 101)
LAM = 1e-3                                   # convex logistic head (certified in E09)


def mean_ll(Y, P, rows):
    v = []
    for j in range(Y.shape[1]):
        m = rows & (Y[:, j] >= 0)
        if m.any():
            v.append(log_loss(Y[m, j], P[m, j]))
    return float(np.mean(v))


def calibrate(P, Y, fit_rows, order_dates):
    cal, info, Pc = {}, {}, P.copy()
    for j, h in enumerate(H):
        m = fit_rows & (Y[:, j] >= 0)
        c, inf = fit_best(P[m, j], Y[m, j], np.argsort(order_dates[m], kind="stable"))
        cal[h], info[h] = c, inf["chosen"]
        Pc[:, j] = c.transform(P[:, j])
    return Pc, cal, info


def main() -> int:
    e03, e06, e07 = sys.argv[1:4]
    torch.set_num_threads(5)
    t0 = time.time()
    fr = load_tables()
    blk = fr.block.to_numpy()
    dates = fr.asof_date.to_numpy()
    full, pre = build(fr, blk == "train", FULL, "convnext_tiny")
    Y = full.y
    va, ca, te = blk == "valid", blk == "calib", blk == "test"
    pr = lambda run, name: np.load(paths.ARTIFACTS / "predictions" / run / f"{name}__probs.npy").astype(np.float64)
    e06m = json.loads((paths.REPO_ROOT / "experiments" / e06 / "metrics.json").read_text(encoding="utf-8"))
    e07m = json.loads((paths.REPO_ROOT / "experiments" / e07 / "metrics.json").read_text(encoding="utf-8"))
    run = Run("e08_final_model", {"e03": e03, "e06": e06, "e07": e07, "blend_grid": "simplex over (lgbm, fusion ensemble, convex logistic), step 0.1",
                                  "e06_best_kw": e06m["selection"]["kw"], "e07_best_params": e07m["best_params"]})
    Xn = full.x["num"].astype(np.float64)

    def fit_logistic(rows):
        from sklearn.linear_model import LogisticRegression
        out, models = np.zeros((len(fr), len(H), 3)), {}
        for j, h in enumerate(H):
            m = rows & (Y[:, j] >= 0)
            lr_ = LogisticRegression(C=1.0 / (LAM * m.sum()), max_iter=3000, tol=1e-8)   # = mean CE + LAM/2 ||w||^2
            lr_.fit(Xn[m], Y[m, j])
            out[:, j] = lr_.predict_proba(Xn)
            models[h] = lr_
        return out, models
    P_log, _ = fit_logistic(blk == "train")
    base = {"lgbm": pr(e07, "lgbm_tuned"), "fusion": pr(e06, "best_seed_ensemble"), "logistic": P_log}
    cands = {"e03_gated_1seed": pr(e03, "variant_gated"), "e06_numbers_ens": pr(e06, "num_seed_ensemble")}
    grid = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
    for wl, wf, wg in grid:
        cands[f"blend_l{wl:.1f}_f{wf:.1f}_g{wg:.1f}"] = wl * base["lgbm"] + wf * base["fusion"] + wg * base["logistic"]
    res = {"candidates": {n: {"valid": mean_ll(Y, P, va), "calib": mean_ll(Y, P, ca), "test": mean_ll(Y, P, te)} for n, P in cands.items()},
           "n_candidates": len(cands)}
    sel = min(cands, key=lambda n: res["candidates"][n]["valid"])
    res["selected"] = sel
    print("selected:", sel, res["candidates"][sel], flush=True)
    if sel.startswith("blend_"):
        wl, wf, wg = (float(t[1:]) for t in sel.replace("blend_", "").split("_"))
    else:
        raise SystemExit(f"selected a non-blend reference model {sel}; deployment for it is not implemented")
    w_l = wl
    use_fusion = wf > 0
    fusion_mods = FULL
    # ---- calibration on the selection-stage (2003-2015) predictions
    P_sel = cands[sel]
    Pc_sel, cal_sel, cal_info = calibrate(P_sel, Y, ca, dates)
    res["calibration_selection_stage"] = cal_info
    # walk-forward recalibration of the test years (each year on the previous year's ended labels)
    P_roll = P_sel.copy()
    for y in (2020, 2021, 2022):
        fit_rows = (pd.to_datetime(dates) >= pd.Timestamp(f"{y - 1}-01-01")) & (pd.to_datetime(dates) <= pd.Timestamp(f"{y - 1}-11-30"))
        tgt = (pd.to_datetime(dates) >= pd.Timestamp(f"{y}-01-01")) & (pd.to_datetime(dates) <= pd.Timestamp(f"{y}-12-31")) & te
        for j, h in enumerate(H):
            m = fit_rows & (Y[:, j] >= 0)
            c, _ = fit_best(P_sel[m, j], Y[m, j], np.argsort(dates[m], kind="stable"))
            P_roll[tgt, j] = c.transform(P_sel[tgt, j])
    res["calibration_compare_test"] = {"none": mean_ll(Y, P_sel, te), "static_2019": mean_ll(Y, Pc_sel, te),
                                       "walk_forward_previous_year": mean_ll(Y, P_roll, te)}
    print("calibration on test:", res["calibration_compare_test"], flush=True)
    # ---- refit on 2003-2018
    refit_rows = np.isin(blk, ["train", "inner", "valid"])
    P_ref = np.zeros_like(P_sel)
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    components = {"lgbm_weight": w_l, "fusion_weight": wf, "logistic_weight": wg,
                  "fusion_modalities": list(fusion_mods) if use_fusion else [], "fusion_seeds": [], "lgbm": {}}
    if wg > 0:
        P_logr, log_models = fit_logistic(refit_rows)
        P_ref += wg * P_logr
        with open(bdir / "logistic.pkl", "wb") as f:
            pickle.dump(log_models, f)
        components["logistic"] = {"file": "logistic.pkl", "lambda": LAM}
    if w_l > 0:
        feats = pre.num_cols
        best = e07m["best_params"]
        Pl = np.zeros_like(P_sel)
        for j, h in enumerate(H):
            m = refit_rows & (Y[:, j] >= 0)
            n_old = int(((blk == "train") & (Y[:, j] >= 0)).sum())
            rounds = max(1, int(round(e07m["curves"][h]["best_iteration_inner"] * m.sum() / n_old)))
            bst = lgb.train({**LGB_PARAMS, "num_threads": 5, **best}, lgb.Dataset(fr.loc[m, feats], Y[m, j]), num_boost_round=rounds)
            Pl[:, j] = bst.predict(fr[feats])
            bst.save_model(str(bdir / f"lgbm_{h}.txt"))
            components["lgbm"][h] = {"file": f"lgbm_{h}.txt", "rounds": rounds}
        P_ref += w_l * Pl
    if use_fusion:
        kw = e06m["selection"]["kw"]
        names = [m for m in full.names if m in fusion_mods]
        d = type(full)({k: full.x[k] for k in names}, full.mask[:, [full.names.index(k) for k in names]], Y, full.z, names)
        Pf = []
        for s in SEEDS:
            grp = "best"
            ep = e06m[f"{grp}_seed{s}"]["best_epoch"] + 1
            cfg = FusionConfig(modality_dims={k: v.shape[1] for k, v in d.x.items()}, variant="gated", d_model=kw.get("d_model", 64),
                               hidden=2 * kw.get("d_model", 64), dropout=kw.get("dropout", 0.1),
                               modality_dropout=0.15 if len(names) > 1 else 0.0)
            ref_idx = np.flatnonzero(refit_rows)
            model, _ = train(cfg, d.subset(ref_idx), d.subset(np.flatnonzero(ca)), paths.ARTIFACTS / "models" / run.run_id / f"refit_seed{s}",
                             epochs=ep, batch=2048, lr=kw["lr"], weight_decay=kw.get("wd", 1e-4), patience=10 ** 6, seed=s,
                             log=lambda x: None, schedule="cosine", restore_best=False)
            P_s, _, _ = predict(model, d)
            Pf.append(P_s)
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, bdir / f"fusion_seed{s}.pt")
            components["fusion_seeds"].append({"file": f"fusion_seed{s}.pt", "seed": s, "epochs": ep})
            print(f"refit fusion seed {s}: {ep} epochs ({time.time() - t0:.0f} s)", flush=True)
        P_ref += wf * np.mean(Pf, axis=0)
    Pc_ref, cal_ref, cal_ref_info = calibrate(P_ref, Y, ca, dates)
    res["calibration_refit"] = cal_ref_info
    # ---- final test evaluation (deployed = refit + 2019 calibration)
    final = {"test_mean_log_loss": mean_ll(Y, Pc_ref, te), "selection_stage_test_mean_log_loss": mean_ll(Y, Pc_sel, te), "by_horizon": {}}
    prior_counts = {h: np.bincount(Y[(blk == "train") & (Y[:, j] >= 0), j], minlength=3) for j, h in enumerate(H)}
    e03P = np.load(paths.ARTIFACTS / "predictions" / e03 / "selected_calibrated__probs.npy").astype(np.float64)
    tb = month_blocks(fr.asof_date[te])
    for j, h in enumerate(H):
        m = te & (Y[:, j] >= 0)
        rep = classification_report(Y[m, j], Pc_ref[m, j])
        prior = np.tile(prior_counts[h] / prior_counts[h].sum(), (m.sum(), 1))
        bm = month_blocks(fr.asof_date[m])
        rep["gain_vs_prior"] = block_bootstrap(lambda yy, a, b: log_loss(yy, b) - log_loss(yy, a), (Y[m, j], Pc_ref[m, j], prior), bm, n_boot=500)
        rep["gain_vs_e03"] = block_bootstrap(lambda yy, a, b: log_loss(yy, b) - log_loss(yy, a), (Y[m, j], Pc_ref[m, j], e03P[m, j]), bm, n_boot=500)
        nd = m & (Y[:, j] != 1)
        rep["auroc_up_down_ci"] = block_bootstrap(lambda a, b: roc_auc_score(a, b), ((Y[nd, j] == 2).astype(int),
                                                  Pc_ref[nd, j, 2] - Pc_ref[nd, j, 0]), month_blocks(fr.asof_date[nd]), n_boot=500)
        rep["by_year"] = {}
        for y in (2020, 2021, 2022):
            my = m & (pd.to_datetime(dates).year == y)
            if my.sum() > 100:
                ry = classification_report(Y[my, j], Pc_ref[my, j])
                rep["by_year"][y] = {k: ry.get(k) for k in ("log_loss", "auroc_up_vs_down", "auroc_SIDEWAYS", "ece_top", "n")}
        final["by_horizon"][h] = rep
        print(f"{h}: test LL {rep['log_loss']:.5f} gain vs prior {rep['gain_vs_prior']['estimate']:+.5f} "
              f"[{rep['gain_vs_prior']['ci95'][0]:+.5f},{rep['gain_vs_prior']['ci95'][1]:+.5f}] up/down {rep.get('auroc_up_vs_down', float('nan')):.3f}", flush=True)
    res["final"] = final
    # quantile heads (E03 selected model): coverage of the 10-90% band for z_h
    q = np.load(paths.ARTIFACTS / "predictions" / e03 / "variant_gated__quantiles.npy")
    res["quantile_coverage_e03"] = {}
    for j, h in enumerate(H):
        for name, rows in (("valid", va), ("test", te)):
            m = rows & np.isfinite(full.z[:, j])
            z = full.z[m, j]
            res["quantile_coverage_e03"].setdefault(h, {})[name] = {"coverage_10_90": float(((z >= q[m, j, 0]) & (z <= q[m, j, 2])).mean()),
                                                                    "nominal": 0.8, "median_abs_err": float(np.median(np.abs(z - q[m, j, 1])))}
    # ---- direction skill (validation, selection-stage calibrated probabilities) and bundle
    skill = {}
    for j, h in enumerate(H):
        m = va & (Y[:, j] >= 0) & (Y[:, j] != 1)
        bs = block_bootstrap(lambda a, b: roc_auc_score(a, b), ((Y[m, j] == 2).astype(int), Pc_sel[m, j, 2] - Pc_sel[m, j, 0]),
                             month_blocks(fr.asof_date[m]), n_boot=500)
        skill[h] = {"valid_up_vs_down_auroc": bs["estimate"], "ci95": bs["ci95"], "validated": bool(bs["ci95"][0] > 0.5)}
    with open(bdir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)
    with open(bdir / "calibrators.pkl", "wb") as f:
        pickle.dump(cal_ref, f)
    for fn in ("news_tfidf_svd.pkl", "sec_tfidf_svd.pkl"):
        shutil.copy2(paths.ARTIFACTS / "features" / fn, bdir / fn)
    if components["fusion_seeds"]:
        shutil.copy2(bdir / components["fusion_seeds"][0]["file"], bdir / "fusion.pt")      # first seed = model template
    meta = {"bundle_id": run.run_id, "architecture_version": ARCHITECTURE_VERSION + "+v1.1", "recipe": sel, "components": components,
            "modalities": list(fusion_mods) if use_fusion else ["num"], "backbone": "convnext_tiny", "horizons": HORIZONS,
            "trained_through_block": "refit 2003-2018", "calibration": cal_ref_info, "direction_skill": skill,
            "selection_valid_mean_log_loss": res["candidates"][sel]["valid"], "test_mean_log_loss_calibrated": final["test_mean_log_loss"]}
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # predictions for E04 v1.1: valid/calib from the selection-stage model, test from the deployed refit
    P_bt = Pc_sel.copy()
    P_bt[te] = Pc_ref[te]
    out = paths.ARTIFACTS / "predictions" / run.run_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "final_for_backtest__probs.npy", P_bt.astype(np.float32))
    np.save(out / "final_deployed_calibrated__probs.npy", Pc_ref.astype(np.float32))
    res["skill"] = skill
    summ = (f"selected {sel} (valid {res['candidates'][sel]['valid']:.5f}); deployed refit test {final['test_mean_log_loss']:.5f} vs "
            f"selection-stage {final['selection_stage_test_mean_log_loss']:.5f}; calibration on test {res['calibration_compare_test']}")
    run.finish("COMPLETED", summ, res)
    print(summ)
    return 0


if __name__ == "__main__":
    sys.exit(main())

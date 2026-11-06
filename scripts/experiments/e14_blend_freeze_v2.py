"""E14: v2 blend, refit on 2003-2018, calibration on 2019, quantile coverage, skill gate, and the v2 bundle (pre-registered,
docs/V2_ARCHITECTURE.md section 4). One blend search; no other tuning.

Components (selection-stage validation predictions): LightGBM and logistic on the tabular set chosen by E11/E16, the E12
fusion ensemble, the E13 TCN and GRU ensembles. For each horizon group (1D | 2D-20D | 3M-1Y) the simplex weights (grid
step 0.1) minimising the group's mean validation log loss are chosen. Every component with positive weight is refitted
on 2003-2018 with its selection-stage length fixed (trees scaled by rows; epochs = best epoch + 1; final weights kept;
2019 is never used to stop training). Calibration per horizon (none / temperature / vector) is chosen on chronological
halves of 2019 and fitted on 2019. Quantile ranges come from the fusion heads and are shown only where the 10-90% band
covered 0.80 +/- 0.08 on validation. The direction-skill gate uses the validation blend (month-block bootstrap CI of the
up-vs-down AUROC above 0.5). Test 2020-2022 is reported as a reused diagnostic only.
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, seqmodels as SM  # noqa: E402
from nexus_alpha.calibration.scalers import fit_best  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.evaluation.stats import block_bootstrap, month_blocks  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.multimodal.fusion import FusionConfig  # noqa: E402
from nexus_alpha.multimodal.trainer import predict as fpredict, train as ftrain  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

H = list(HORIZONS_V2)
EXP = REPO / "experiments"


def latest(prefix):
    for r in sorted(EXP.glob(f"*_{prefix}_*"), reverse=True):
        man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
        if man.get("status") == "COMPLETED":
            return r.name, json.loads((r / "metrics.json").read_text(encoding="utf-8"))
    return None, None


def tabular_choice(final: bool = True):
    """(feature columns, prediction dir, set name, source run, per-horizon results) from E16's keep decisions on top of
    E11's carried set. With final=True and a completed E18 run, its all-horizon predictions are used."""
    if final:
        e18_id, e18 = latest("e18_tabular_final")
        if e18:
            return e18["cols"], paths.ARTIFACTS / "predictions" / "e18", "FINAL", e18_id, e18["by_horizon"]
    e11_id, e11 = latest("e11_macro_ladder")
    base, _ = L.carried_feature_set(EXP)
    e16_id, e16 = latest("e16_text_encoders")
    if e16:
        k = e16["keep_rule_decisions"]
        kept = lambda new, old: any(k.get(f"{m}|{new}_vs_{old}") for m in ("lgbm", "logistic"))
        name = "B"
        if kept("B_finbert", "B") and kept("B_newsv1", "B") and kept("B_both", "B_newsv1"):
            name = "B_both"
        elif kept("B_newsv1", "B"):
            name = "B_newsv1"
        elif kept("B_finbert", "B"):
            name = "B_finbert"
        return e16["sets"][name], paths.ARTIFACTS / "predictions" / "e16", name, e16_id, e16["by_horizon"]
    # no E16: use E11's top carried rung predictions
    order = [s for s in e11["sets"] if not s.endswith("placebo")]
    top = [s for s in order if set(e11["sets"][s]) == set(base)]
    return base, paths.ARTIFACTS / "predictions" / "e11", (top[0] if top else order[0]), e11_id, e11["by_horizon"]


def simplex(n, step=0.1):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) == k:
            yield np.array(c, float) / k


def main() -> None:
    t0 = time.time()
    cols, tab_dir, tab_set, tab_src, tab_res = tabular_choice()
    e12_id, e12 = latest("e12_fusion_v2")
    e13_id, e13 = latest("e13_sequence")
    print(f"tabular set {tab_set} from {tab_src} ({len(cols)} cols); E12 {e12_id}; E13 {e13_id}", flush=True)
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "calib", "test"])].reset_index(drop=True)
    blk = fr.block.to_numpy()
    Y = np.stack([fr[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in H], 1)
    # validation predictions of every component, aligned to (valid rows with a label) per horizon
    va = np.flatnonzero(blk == "valid")
    comp_names = ["lgbm", "logistic"] + (["fusion"] if e12 else []) + [a for a in ("tcn", "gru") if e13]
    Pv = {}
    for j, h in enumerate(H):
        m = Y[va, j] >= 0
        Pv[(h, "lgbm")] = np.load(tab_dir / f"{h}_{tab_set}_lgbm_valid.npy")
        Pv[(h, "logistic")] = np.load(tab_dir / f"{h}_{tab_set}_logistic_valid.npy")
        if e12:
            Pv[(h, "fusion")] = np.load(Path(e12["pred_dir"]) / "ens_valid.npy")[m, j]
        for a_ in ("tcn", "gru"):
            if e13:
                Pv[(h, a_)] = np.load(Path(e13["pred_dir"]) / f"{a_}_ens_valid.npy")[m, j]
        assert all(len(Pv[(h, c)]) == m.sum() for c in comp_names), f"alignment {h}"
    groups = {g: [h for h in hs if h in H] for g, hs in L.GROUPS.items()}
    weights, sel = {}, {}
    for g, hs in groups.items():
        best = (np.inf, None)
        for w in simplex(len(comp_names)):
            ll = np.mean([L.ll_rows(Y[va, H.index(h)][Y[va, H.index(h)] >= 0],
                                    sum(wi * Pv[(h, c)] for wi, c in zip(w, comp_names))).mean() for h in hs])
            if ll < best[0]:
                best = (ll, w)
        weights[g] = dict(zip(comp_names, best[1].tolist()))
        sel[g] = {"valid_mean_ll": best[0],
                  "single_best": {c: float(np.mean([L.ll_rows(Y[va, H.index(h)][Y[va, H.index(h)] >= 0], Pv[(h, c)]).mean()
                                                     for h in hs])) for c in comp_names}}
        print(g, weights[g], round(best[0], 5), flush=True)
    gw = {h: weights[g] for g, hs in groups.items() for h in hs}
    used = {c for w in weights.values() for c, x in w.items() if x > 0}
    run = Run("e14_blend_freeze_v2", {"components": comp_names, "weights": weights, "tabular": [tab_src, tab_set],
                                      "e12": e12_id, "e13": e13_id, "grid_step": 0.1},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    ref = np.isin(blk, ["train", "inner", "valid"])
    ca, te = blk == "calib", blk == "test"
    out_rows = np.flatnonzero(ca | te)
    P_ref = np.zeros((len(out_rows), len(H), 3))
    comp_meta = {"tabular_cols": cols, "tabular_set": tab_set}
    # ---- tabular refits
    if {"lgbm", "logistic"} & used:
        text = paths.ARTIFACTS / "features"
        X = fr
        if any(c.startswith("fb_") for c in cols):
            fb = pd.read_parquet(text / "finbert_features.parquet")
            fb["fb_present"] = 1.0
            X = X.merge(fb, on="sample_id", how="left")
            X["fb_present"] = X.fb_present.fillna(0.0)
        if any(c.startswith("news_") for c in cols):
            X = X.merge(pd.read_parquet(text / "news_features.parquet"), on="sample_id", how="left")
        X = X[cols]
        n_tr = None
        lg_models, lr_models = {}, {}
        for j, h in enumerate(H):
            m = ref & (Y[:, j] >= 0)
            w = gw[h]
            if w.get("lgbm", 0) > 0:
                trees = tab_res[h][tab_set]["lgbm"]["trees"]
                n_tr = ((blk == "train") & (Y[:, j] >= 0)).sum()
                rounds = max(1, int(round(trees * m.sum() / n_tr)))
                cat = ["rel_sector_id"] if "rel_sector_id" in cols else "auto"
                bst = lgb.train({**L.LGB_BASE, **L.E07_PARAMS, "num_threads": 8},
                                lgb.Dataset(X[m], Y[m, j], categorical_feature=cat), num_boost_round=rounds)
                bst.save_model(str(bdir / f"lgbm_{h}.txt"))
                lg_models[h] = rounds
                P_ref[:, j] += w["lgbm"] * bst.predict(X.iloc[out_rows])
            if w.get("logistic", 0) > 0:
                lc = [c for c in cols if c != "rel_sector_id"]
                Xl = X[lc]
                med = Xl[m].median()
                f_ = Xl[m].fillna(med)
                mu, sd = f_.mean(), f_.std().replace(0, 1)
                Z = lambda r: ((Xl.iloc[r].fillna(med) - mu) / sd).clip(-8, 8).to_numpy(np.float64)
                lr = LogisticRegression(C=1.0 / (1e-3 * m.sum()), max_iter=400, tol=1e-6).fit(Z(np.flatnonzero(m)), Y[m, j])
                lr_models[h] = {"model": lr, "cols": lc, "median": med, "mu": mu, "sd": sd}
                P_ref[:, j] += w["logistic"] * lr.predict_proba(Z(out_rows))
        with open(bdir / "logistic.pkl", "wb") as f:
            pickle.dump(lr_models, f)
        comp_meta["lgbm_rounds"] = lg_models
        del X
        print(f"tabular refit done ({time.time() - t0:.0f}s)", flush=True)
    # ---- fusion refit (5 seeds, epochs fixed)
    Q_ref = None
    if "fusion" in used or e12:
        import e12_fusion_v2 as E12
        from nexus_alpha.multimodal.dataset import build
        from nexus_alpha.multimodal.trainer import ModalityData
        bm = json.loads((Path(paths.ARTIFACTS / "bundles") / json.loads(
            (REPO / "registry" / "frozen_models" / "v1.1.json").read_text())["bundle_id"] / "bundle_meta.json").read_text())
        frf = E12.load_frame()
        d, pre = build(frf, ref, modalities=tuple(bm["modalities"]))
        e12cfg = json.loads((EXP / e12_id / "manifest.json").read_text())["config"]
        ctx_cols = [c for c in L.carried_feature_set(EXP)[0] if c not in pre.num_cols]
        x, names, mask = dict(d.x), list(d.names), d.mask
        ctx_mu = ctx_sd = None
        if ctx_cols:
            x["ctx"], ctx_mu, ctx_sd = E12.ctx_block(frf, ctx_cols, np.flatnonzero(ref), return_stats=True)
            names.append("ctx")
            mask = np.concatenate([mask, np.ones((len(frf), 1), np.float32)], 1)
        z = np.stack([frf[f"z_{h}"].to_numpy(np.float32) for h in H], 1)
        data = ModalityData(x, mask, Y, z, names)
        Pf, Qf, seeds = [], [], []
        for s, info in e12["per_seed"].items():
            ep = int(info["best_epoch"]) + 1
            cfg = FusionConfig(modality_dims={k: v.shape[1] for k, v in x.items()}, n_horizons=len(H), variant="gated",
                               d_model=64, hidden=128, dropout=0.3, modality_dropout=0.15)
            model, _ = ftrain(cfg, data.subset(np.flatnonzero(ref)), data.subset(np.flatnonzero(ca)),
                              paths.ARTIFACTS / "models" / run.run_id / f"fusion_seed{s}", epochs=ep, batch=2048, lr=3e-4,
                              weight_decay=1e-4, patience=10 ** 6, seed=int(s), log=lambda m: None, schedule="cosine",
                              restore_best=False)
            P_s, Q_s, _ = fpredict(model, data.subset(out_rows))
            Pf.append(P_s)
            Qf.append(Q_s)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, bdir / f"fusion_seed{s}.pt")
            seeds.append({"seed": int(s), "epochs": ep})
        with open(bdir / "fusion_preprocessor.pkl", "wb") as f:
            pickle.dump({"pre": pre, "ctx_cols": ctx_cols, "names": names, "ctx_mu": ctx_mu, "ctx_sd": ctx_sd}, f)
        comp_meta["fusion_seeds"] = seeds
        Q_ref = np.sort(np.mean(Qf, 0), axis=-1)
        if "fusion" in used:
            for j, h in enumerate(H):
                P_ref[:, j] += gw[h].get("fusion", 0) * np.mean(Pf, 0)[:, j]
        print(f"fusion refit done ({time.time() - t0:.0f}s)", flush=True)
    # ---- sequence refits
    if {"tcn", "gru"} & used:
        import e13_sequence as E13
        Xs = E13.build_X(fr)
        e13cfg = json.loads((EXP / e13_id / "manifest.json").read_text())["config"]
        mu, sd = np.array(e13cfg["channel_mean"]), np.array(e13cfg["channel_sd"])
        Xs = np.clip(np.nan_to_num((Xs - mu) / sd, nan=0.0), -8, 8).astype(np.float32)
        torch.set_num_threads(8)
        seq_meta = {}
        for a_ in ("tcn", "gru"):
            if a_ not in used:
                continue
            Ps = []
            for key, info in e13["per_seed"].items():
                if not key.startswith(a_):
                    continue
                s = int(key.split("seed")[1])
                torch.manual_seed(s)
                mdl = SM.TCN(Xs.shape[2], len(H)) if a_ == "tcn" else SM.GRUNet(Xs.shape[2], len(H))
                mdl, _ = SM.fit(mdl, Xs[ref], Y[ref], epochs=int(info["best_epoch"]) + 1, seed=s, log=lambda m: None,
                                restore_best=False)
                torch.save(mdl.state_dict(), bdir / f"{a_}_seed{s}.pt")
                Ps.append(SM.predict(mdl, Xs[out_rows]))
                seq_meta.setdefault(a_, []).append({"seed": s, "epochs": int(info["best_epoch"]) + 1})
            for j, h in enumerate(H):
                P_ref[:, j] += gw[h].get(a_, 0) * np.mean(Ps, 0)[:, j]
        comp_meta["sequence"] = {"seeds": seq_meta, "channel_mean": mu.tolist(), "channel_sd": sd.tolist(),
                                 "window": SM.WINDOW, "channels": SM.CHANNELS}
        print(f"sequence refit done ({time.time() - t0:.0f}s)", flush=True)
    # ---- calibration on 2019 (chronological halves choose the method), applied to test
    dates = fr.asof_date.to_numpy()
    ca_o, te_o = ca[out_rows], te[out_rows]
    Yo = Y[out_rows]
    cal, cal_info, Pc = {}, {}, P_ref.copy()
    for j, h in enumerate(H):
        m = ca_o & (Yo[:, j] >= 0)
        if m.sum() < 500:            # pre-registered: 1Y has no 2019 labels after the purge -> left uncalibrated
            from nexus_alpha.calibration.scalers import Identity
            cal[h], cal_info[h] = Identity(), "none (no 2019 labels)"
            Pc[:, j] = P_ref[:, j]
            continue
        c, inf = fit_best(P_ref[m, j], Yo[m, j], np.argsort(dates[out_rows][m], kind="stable"))
        cal[h], cal_info[h] = c, inf["chosen"]
        Pc[:, j] = c.transform(P_ref[:, j])
    with open(bdir / "calibrators.pkl", "wb") as f:
        pickle.dump(cal, f)
    # deployed-model predictions for every 2019-2022 row (E15 backtest input)
    pd_dir = paths.artifact("predictions", run.run_id, "x").parent
    fr.iloc[out_rows][["sample_id", "ticker", "asof_date", "block"]].reset_index(drop=True).to_parquet(pd_dir / "rows.parquet")
    np.save(pd_dir / "probs_calibrated.npy", Pc.astype(np.float32))
    if Q_ref is not None:
        np.save(pd_dir / "quantiles.npy", Q_ref.astype(np.float32))
    # ---- skill gate from the validation blend; quantile coverage of the selection-stage fusion on validation
    skill, cover = {}, {}
    mb = month_blocks(fr.asof_date.iloc[va])
    for j, h in enumerate(H):
        m = Y[va, j] >= 0
        Pb = sum(gw[h][c] * Pv[(h, c)] for c in comp_names)
        yy = Y[va, j][m]
        ud = yy != 1
        bs = block_bootstrap(lambda a_, b_: roc_auc_score(a_, b_), ((yy[ud] == 2).astype(int), (Pb[ud, 2] - Pb[ud, 0])),
                             mb[m][ud], n_boot=500)
        skill[h] = {"valid_up_vs_down_auroc": bs["estimate"], "ci95": bs["ci95"], "validated": bool(bs["ci95"][0] > 0.5)}
        if e12:
            Qv = np.load(Path(e12["pred_dir"]) / "ens_valid_quantiles.npy")[m, j]
            zv = fr[f"z_{h}"].to_numpy()[va][m]
            cv = float(np.mean((zv >= Qv[:, 0]) & (zv <= Qv[:, 2])))
            cover[h] = {"valid_coverage_10_90": cv, "show_in_app": bool(abs(cv - 0.8) <= 0.08)}
    # ---- test diagnostics (reused block)
    prior = {h: [float((Y[ref & (Y[:, j] >= 0), j] == c).mean()) for c in range(3)] for j, h in enumerate(H)}
    test = {}
    for j, h in enumerate(H):
        m = te_o & (Yo[:, j] >= 0)
        if not m.any():
            continue
        yy = Yo[m, j]
        test[h] = {"n": int(m.sum()), "ll": float(L.ll_rows(yy, Pc[m, j]).mean()),
                   "ll_uncal": float(L.ll_rows(yy, P_ref[m, j]).mean()),
                   "ll_prior": float(-np.log(np.array(prior[h])[yy]).mean()), "auroc_ud": L.auroc_ud(yy, Pc[m, j])}
        if Q_ref is not None:
            zt = fr[f"z_{h}"].to_numpy()[out_rows][m]
            test[h]["coverage_10_90"] = float(np.mean((zt >= Q_ref[m, j, 0]) & (zt <= Q_ref[m, j, 2])))
    meta = {"bundle_id": run.run_id, "architecture_version": "nexus-alpha-0.2+v2", "horizons": HORIZONS_V2,
            "groups": groups, "weights": weights, "components": comp_meta, "calibration": cal_info,
            "direction_skill": skill, "quantile_coverage": cover, "train_class_prior": prior,
            "trained_through_block": "refit 2003-2018", "sources": {"tabular": tab_src, "e12": e12_id, "e13": e13_id},
            "selection": sel, "test_diagnostic": test}
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    run.finish("COMPLETED", f"E14 v2 bundle {run.run_id}: weights {weights}; test mean LL "
                            f"{np.mean([t['ll'] for t in test.values()]):.5f} vs prior "
                            f"{np.mean([t['ll_prior'] for t in test.values()]):.5f}", meta)
    print(json.dumps({h: test[h] for h in list(test)[:6]}, indent=1))


if __name__ == "__main__":
    main()

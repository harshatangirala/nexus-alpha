"""E03: multimodal fusion comparison, ablations, calibration, registered hypotheses H1-H5 and the inference bundle.

Protocol (docs/PROJECT_SPECIFICATION.md, registered in commit 1b17168 before this script ran):
  A. five fusion variants on all modalities (img_W, img_D, img_H, num, news, sec), frozen ConvNeXt-T embeddings;
     the variant with the lowest mean validation log loss (six horizons) is selected (ties within 0.0005 -> simpler).
  B. ablations with the selected variant (master spec 11.6): single views, tri-frame, numbers, news, SEC and their
     combinations; no modality dropout; plus rule features; plus the chart-fine-tuned ConvNeXt-T backbone.
  C. calibration per horizon on the calib block (none / temperature / vector chosen on chronological halves).
  D. hypotheses H1-H5: paired month-block bootstrap of mean six-horizon log loss on the test block, Holm-corrected.
  E. missing-modality stress tests and the exported inference bundle for the app.
Run: .venv/Scripts/python -I scripts/experiments/e03_fusion.py [--quick]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from nexus_alpha import ARCHITECTURE_VERSION, paths
from nexus_alpha.calibration.scalers import fit_best
from nexus_alpha.cv import store as ST
from nexus_alpha.evaluation.metrics import classification_report, log_loss
from nexus_alpha.evaluation.stats import block_bootstrap, holm, month_blocks
from nexus_alpha.experiments import Run
from nexus_alpha.multimodal.dataset import build, load_tables
from nexus_alpha.multimodal.fusion import FusionConfig
from nexus_alpha.multimodal.trainer import predict, train
from nexus_alpha.targets import HORIZONS

H = list(HORIZONS)
FULL = ("img_W", "img_D", "img_H", "num", "news", "sec")
VARIANTS = ("late", "concat", "attention", "gated", "hybrid")
SIMPLICITY = {"late": 0, "concat": 1, "gated": 2, "attention": 3, "hybrid": 4}
ABLATIONS = {
    "a01_img_W": ("img_W",), "a02_img_D": ("img_D",), "a03_img_H": ("img_H",), "a04_img_WDH": ("img_W", "img_D", "img_H"),
    "a05_num": ("num",), "a06_img_num": ("img_W", "img_D", "img_H", "num"), "a07_news": ("news",), "a08_sec": ("sec",),
    "a09_news_sec": ("news", "sec"), "a10_img_news": ("img_W", "img_D", "img_H", "news"),
    "a11_img_sec": ("img_W", "img_D", "img_H", "sec"), "a12_img_news_sec": ("img_W", "img_D", "img_H", "news", "sec"),
    "a13_full": FULL, "a17_full_rules": FULL + ("rules",),
}


def mean_ll(Y: np.ndarray, P: np.ndarray) -> float:
    vals = []
    for j in range(Y.shape[1]):
        m = Y[:, j] >= 0
        if m.any():
            vals.append(log_loss(Y[m, j], P[m, j]))
    return float(np.mean(vals))


def evaluate(Y, P, rows, label):
    out = {"mean_log_loss": mean_ll(Y[rows], P[rows])}
    for j, h in enumerate(H):
        m = rows & (Y[:, j] >= 0)
        out[h] = classification_report(Y[m, j], P[m, j])
    return out


def main() -> int:
    quick = "--quick" in sys.argv
    t0 = time.time()
    fr = load_tables()
    blocks = fr.block.to_numpy()
    tr_m, es_m = blocks == "train", blocks == "inner"
    val_m, cal_m, te_m = blocks == "valid", blocks == "calib", blocks == "test"
    epochs = 3 if quick else 20
    cfg = {"variants": VARIANTS, "ablations": {k: list(v) for k, v in ABLATIONS.items()}, "backbone": "convnext_tiny",
           "pca_per_view": 64, "d_model": 64, "epochs_max": epochs, "patience": 3, "batch": 2048, "lr": 1e-3,
           "modality_dropout": 0.15, "seed": 7, "registered_in_commit": "1b17168", "quick": quick}
    run = Run("e03_fusion" + ("_quick" if quick else ""), cfg)
    pred_dir = paths.ARTIFACTS / "predictions" / run.run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    results, probs, gates = {}, {}, {}
    cache = {}

    def data_for(mods, backbone="convnext_tiny"):
        """Each modality is preprocessed independently, so slicing the full build is identical to building a subset."""
        if backbone not in cache:
            cache[backbone] = build(fr, tr_m, FULL + ("rules",), backbone)
        full, pre = cache[backbone]
        names = [m for m in full.names if m in mods]
        sub = type(full)({k: full.x[k] for k in names}, full.mask[:, [full.names.index(k) for k in names]],
                         full.y, full.z, names)
        return sub, pre

    def fit_eval(name, mods, variant, backbone="convnext_tiny", mdrop=0.15):
        d, pre = data_for(mods, backbone)
        fc = FusionConfig(modality_dims={k: v.shape[1] for k, v in d.x.items()}, variant=variant, d_model=64, hidden=128,
                          modality_dropout=mdrop if len(d.names) > 1 else 0.0)
        ck = paths.ARTIFACTS / "models" / run.run_id / name
        model, hist = train(fc, d.subset(np.flatnonzero(tr_m)), d.subset(np.flatnonzero(es_m)), ck, epochs=epochs,
                            batch=2048, patience=3, seed=7, log=lambda s: None)
        P, Q, diag = predict(model, d)
        probs[name] = P
        if "gates" in diag:
            gates[name] = {"names": d.names, "mean_test": diag["gates"][te_m].mean(0).tolist(),
                           "mean_valid": diag["gates"][val_m].mean(0).tolist()}
        res = {"modalities": list(d.names), "variant": variant, "backbone": backbone, "epochs_run": len(hist),
               "best_inner_ce": min(h["inner_ce"] for h in hist), "valid": evaluate(d.y, P, val_m, "valid"),
               "test_raw": evaluate(d.y, P, te_m, "test")}
        results[name] = res
        np.save(pred_dir / f"{name}__probs.npy", P.astype(np.float32))
        np.save(pred_dir / f"{name}__quantiles.npy", Q.astype(np.float32))
        print(f"{name:<28} valid {res['valid']['mean_log_loss']:.5f} test {res['test_raw']['mean_log_loss']:.5f} "
              f"epochs {len(hist)} ({time.time() - t0:.0f} s)", flush=True)
        return model, d, pre, fc

    try:
        fr[["sample_id", "ticker", "asof_date", "block"]].to_parquet(pred_dir / "rows.parquet", index=False)
        Y = data_for(FULL)[0].y
        # ---------- A. fusion variants on all modalities
        models = {}
        for v in VARIANTS:
            models[v] = fit_eval(f"variant_{v}", FULL, v)
        best = min(VARIANTS, key=lambda v: (round(results[f"variant_{v}"]["valid"]["mean_log_loss"] / 0.0005), SIMPLICITY[v]))
        sel_name = f"variant_{best}"
        print("selected variant:", best, flush=True)
        # ---------- B. ablations with the selected variant
        for name, mods in ABLATIONS.items():
            if name == "a13_full":
                results[name] = results[sel_name]; probs[name] = probs[sel_name]
                continue
            fit_eval(name, mods, best)
        fit_eval("a15_full_no_modality_dropout", FULL, best, mdrop=0.0)
        ST.assert_eval_allowed("ft_convnext_tiny", fr.asof_date[val_m | cal_m | te_m])
        fit_eval("b01_full_ft_convnext", FULL, best, backbone="ft_convnext_tiny")
        # ---------- C. calibration of the selected model (calib block only)
        P = probs[sel_name]
        cal_order = np.argsort(fr.asof_date.to_numpy()[cal_m], kind="stable")
        calibrators, cal_info = {}, {}
        Pc = P.copy()
        for j, h in enumerate(H):
            m = cal_m & (Y[:, j] >= 0)
            c, info = fit_best(P[m, j], Y[m, j], np.argsort(fr.asof_date.to_numpy()[m], kind="stable"))
            calibrators[h], cal_info[h] = c, info
            Pc[:, j] = c.transform(P[:, j])
        probs["selected_calibrated"] = Pc
        results["selected_calibrated"] = {"test": evaluate(Y, Pc, te_m, "test"), "valid": evaluate(Y, Pc, val_m, "valid"),
                                          "calibration": cal_info}
        np.save(pred_dir / "selected_calibrated__probs.npy", Pc.astype(np.float32))
        # ---------- D. registered hypotheses (paired month-block bootstrap on the test block)
        tb = month_blocks(fr.asof_date[te_m])
        Yt = Y[te_m]
        def diff(a, b, rows=None):
            r = np.ones(len(Yt), bool) if rows is None else rows
            return block_bootstrap(lambda yy, pa, pb: mean_ll(yy, pb) - mean_ll(yy, pa),
                                   (Yt[r], probs[a][te_m][r], probs[b][te_m][r]), tb[r], n_boot=1000)
        single = min(("a01_img_W", "a02_img_D", "a03_img_H"), key=lambda n: results[n]["valid"]["mean_log_loss"])
        news_or_sec = ((fr.news_present.fillna(0) > 0) | (fr.sec_covered.fillna(0) > 0)).to_numpy()[te_m]
        hyp = {"H1_triframe": diff("a04_img_WDH", single), "H2_nlp_matched": diff("a13_full", "a06_img_num", news_or_sec),
               "H3_indicators": diff("a06_img_num", "a04_img_WDH"),
               "H4_fusion_vs_late": diff(sel_name, "variant_late") if best != "late" else None,
               "H5_calibration": diff("selected_calibrated", sel_name)}
        hyp = {k: v for k, v in hyp.items() if v is not None}
        for v in hyp.values():
            # one-sided p for "improvement > 0" = share of bootstrap resamples <= 0, recovered from the two-sided tail
            p2 = v["p_two_sided_vs_0"]
            v["p_one_sided"] = p2 / 2 if v["estimate"] > 0 else 1 - p2 / 2
        hyp_holm = holm({k: v["p_one_sided"] for k, v in hyp.items()})
        results["hypotheses"] = {"tests": hyp, "holm": hyp_holm, "single_view_reference": single,
                                 "h2_rows": int(news_or_sec.sum()), "metric": "mean six-horizon log loss, positive = improvement"}
        # ---------- E. missing-modality stress tests on the selected model (test block)
        model, d, pre, fc = models[best]
        stress = {}
        for drop in ("news", "sec", "img_H", "img_W", "num"):
            if drop not in d.names:
                continue
            d2 = d.subset(np.flatnonzero(te_m))
            d2.mask = d2.mask.copy()
            d2.mask[:, d.names.index(drop)] = 0
            Pm, _, _ = predict(model, d2)
            stress[f"without_{drop}"] = {"mean_log_loss": mean_ll(d2.y, Pm),
                                         "delta_vs_full": mean_ll(d2.y, Pm) - mean_ll(d2.y, P[te_m])}
        results["missing_modality_stress"] = stress
        # ---------- bundle for the app
        bdir = paths.ARTIFACTS / "bundles" / run.run_id
        bdir.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({"model": model.state_dict(), "cfg": asdict(fc)}, bdir / "fusion.pt")
        with open(bdir / "preprocessor.pkl", "wb") as f:
            pickle.dump(pre, f)
        with open(bdir / "calibrators.pkl", "wb") as f:
            pickle.dump(calibrators, f)
        import shutil
        for fn in ("news_tfidf_svd.pkl", "sec_tfidf_svd.pkl"):
            src = paths.ARTIFACTS / "features" / fn
            if src.exists():
                shutil.copy2(src, bdir / fn)
        skill = {}
        for j, h in enumerate(H):
            m = val_m & (Y[:, j] >= 0) & (Y[:, j] != 1)
            yy = (Y[m, j] == 2).astype(int)
            sc = Pc[m, j, 2] - Pc[m, j, 0]
            from sklearn.metrics import roc_auc_score
            bs = block_bootstrap(lambda a, b: roc_auc_score(a, b), (yy, sc), month_blocks(fr.asof_date[m]), n_boot=500)
            skill[h] = {"valid_up_vs_down_auroc": bs["estimate"], "ci95": bs["ci95"], "validated": bool(bs["ci95"][0] > 0.5)}
        meta = {"bundle_id": run.run_id, "architecture_version": ARCHITECTURE_VERSION, "variant": best,
                "modalities": d.names, "backbone": "convnext_tiny", "horizons": HORIZONS, "trained_through_block": "train 2003-2015",
                "calibration": {h: cal_info[h]["chosen"] for h in H}, "direction_skill": skill,
                "selection_valid_mean_log_loss": results[sel_name]["valid"]["mean_log_loss"],
                "test_mean_log_loss_calibrated": results["selected_calibrated"]["test"]["mean_log_loss"]}
        (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        results["bundle"] = meta
        run.save("gates.json", gates)
        summ = (f"selected {best}; valid {results[sel_name]['valid']['mean_log_loss']:.5f}; test calibrated "
                f"{results['selected_calibrated']['test']['mean_log_loss']:.5f}; H: "
                + ", ".join(f"{k} {v['estimate']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] holm p {hyp_holm[k]['p_holm']:.3f}" for k, v in hyp.items()))
        run.finish("COMPLETED", summ, results)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

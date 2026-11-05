"""E02: chart-vision branch on the reused TriFrame s500 embeddings.

Part A (images only, linear): per backbone x view set {W, D, H, WDH}: train-fitted standardisation + PCA(64 per view),
multinomial logistic regression (C=0.05) per horizon. Backbones: frozen ConvNeXt-T, ResNet-18, DINOv2-S, a random-weight
ResNet-18 control and the chart-fine-tuned ConvNeXt-T (evaluated on 2017+ only; governance rule in cv.store).
Part B (images + numbers, trees): LightGBM on structured + TriFrame-33 + PCA(32 per view, WDH) for the frozen backbone
with the best Part A validation log loss, and for the fine-tuned ConvNeXt-T.

Run: .venv/Scripts/python -I scripts/experiments/e02_cv_branch.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nexus_alpha import paths
from nexus_alpha.cv import store as ST
from nexus_alpha.cv.features import ViewPCA
from nexus_alpha.evaluation.metrics import classification_report, log_loss
from nexus_alpha.evaluation.stats import month_blocks, paired_difference
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.targets import HORIZONS
from nexus_alpha.training.tabular import LGB_PARAMS, Prior, fit_lgbm

FRAME = paths.ARTIFACTS / "samples" / "s500_frame.parquet"
EVAL_BLOCKS = ("valid", "calib", "test")
FROZEN = ["convnext_tiny", "resnet18", "vit_small_patch14_dinov2", "resnet18_random_s0"]
FT = ["ft_convnext_tiny"]
VIEWSETS = {"W": ("W",), "D": ("D",), "H": ("H",), "WDH": ("W", "D", "H")}


def lr_probs(Ztr, ytr, Zev):
    m = LogisticRegression(C=0.05, max_iter=500)
    m.fit(Ztr, ytr)
    return m.predict_proba(Zev)


def main() -> int:
    fr = pd.read_parquet(FRAME)
    x33 = [c for c in fr.columns if c.startswith("x_")]
    num = list(STRUCTURED_ALL) + x33
    cfg = {"frame": str(FRAME), "backbones": FROZEN + FT, "viewsets": list(VIEWSETS), "pca_linear": 64,
           "pca_trees": 32, "lr_C": 0.05, "lgb_params": LGB_PARAMS, "horizons": HORIZONS}
    run = Run("e02_cv_branch", cfg, inputs={"frame": FRAME, "emb_manifest": ST.STORE / "manifest.json"})
    pred_dir = paths.ARTIFACTS / "predictions" / run.run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    tr_all = fr[fr.block == "train"]
    es_all = fr[fr.block == "inner"]
    ev = fr[fr.block.isin(EVAL_BLOCKS)].reset_index(drop=True)
    ST.assert_eval_allowed("ft_convnext_tiny", ev.asof_date)
    metrics, extra = {}, {"pca_explained": {}}
    try:
        # ---------------- Part A: images only
        for b in FROZEN + FT:
            vp = ViewPCA(b, ("W", "D", "H"), 64).fit(tr_all.row.to_numpy())
            extra["pca_explained"][b] = vp.explained()
            Ztr_all = vp.transform(tr_all.row.to_numpy())
            Zev_all = vp.transform(ev.row.to_numpy())
            for vs, views in VIEWSETS.items():
                cols = [c for c in Ztr_all.columns if any(f"_{v}_" in c for v in views)]
                name = f"img_lr__{b}__{vs}"
                pr = ev[["sample_id", "block"]].copy()
                for h in HORIZONS:
                    yc = f"y_{h}"
                    mtr = tr_all[yc].notna().to_numpy()
                    p = lr_probs(Ztr_all.loc[mtr, cols].to_numpy(), tr_all.loc[mtr, yc].to_numpy(int), Zev_all[cols].to_numpy())
                    for k, cls in enumerate(("down", "side", "up")):
                        pr[f"p_{cls}_{h}"] = p[:, k]
                    for blk in EVAL_BLOCKS:
                        mk = (ev.block == blk).to_numpy() & ev[yc].notna().to_numpy()
                        metrics.setdefault(h, {}).setdefault(blk, {})[name] = classification_report(ev.loc[mk, yc].to_numpy(int), p[mk])
                pr.to_parquet(pred_dir / f"{name}.parquet", index=False)
                print(name, {h: round(metrics[h]["valid"][name]["log_loss"], 4) for h in HORIZONS}, flush=True)
        # ---------------- Part B: images + numbers (trees), best frozen backbone by mean valid log loss (WDH) + ft
        best_frozen = min([b for b in FROZEN if "random" not in b],
                          key=lambda b: np.mean([metrics[h]["valid"][f"img_lr__{b}__WDH"]["log_loss"] for h in HORIZONS]))
        extra["best_frozen_backbone_by_valid"] = best_frozen
        for b in (best_frozen, "ft_convnext_tiny"):
            vp = ViewPCA(b, ("W", "D", "H"), 32).fit(tr_all.row.to_numpy())
            def X(df):
                return pd.concat([df[num].reset_index(drop=True), vp.transform(df.row.to_numpy())], axis=1)
            Xtr, Xes, Xev = X(tr_all), X(es_all), X(ev)
            name = f"lgbm_num_img__{b}"
            pr = ev[["sample_id", "block"]].copy()
            for h in HORIZONS:
                yc = f"y_{h}"
                mtr, mes = tr_all[yc].notna().to_numpy(), es_all[yc].notna().to_numpy()
                f, _, info = fit_lgbm(Xtr[mtr], tr_all.loc[mtr, yc].to_numpy(int), Xes[mes], es_all.loc[mes, yc].to_numpy(int))
                p = f(Xev)
                extra[f"{name}|{h}"] = info
                for k, cls in enumerate(("down", "side", "up")):
                    pr[f"p_{cls}_{h}"] = p[:, k]
                for blk in EVAL_BLOCKS:
                    mk = (ev.block == blk).to_numpy() & ev[yc].notna().to_numpy()
                    metrics[h][blk][name] = classification_report(ev.loc[mk, yc].to_numpy(int), p[mk])
            pr.to_parquet(pred_dir / f"{name}.parquet", index=False)
            print(name, {h: round(metrics[h]["test"][name]["log_loss"], 4) for h in HORIZONS}, flush=True)
        run.save("extra.json", extra)
        summ = (f"best frozen backbone (valid, WDH, linear): {best_frozen}; images-only valid log loss WDH vs prior-free "
                + ", ".join(f"{h} {metrics[h]['valid'][f'img_lr__{best_frozen}__WDH']['log_loss']:.4f}" for h in HORIZONS))
        run.finish("COMPLETED", summ, metrics)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

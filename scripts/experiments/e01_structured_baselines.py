"""E01: baselines and the structured (indicator) branch on the primary split, six horizons.

Models (each per horizon): class prior | momentum logistic (d_ret1, d_ret5, d_ret21, d_ret63) | LightGBM on NEXUS
multi-timeframe indicators | LightGBM on TriFrame's 33 visible-window features | LightGBM on both | both + market context.
Fit on `train` (2003-2015), early stopping on `inner` (2016), scored on `valid` (2017-18), `calib` (2019) and `test`
(2020-22, historical diagnostic). Out-of-sample probabilities are saved for stacking/fusion.

Run: .venv/Scripts/python -I scripts/experiments/e01_structured_baselines.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import classification_report, log_loss
from nexus_alpha.evaluation.stats import month_blocks, paired_difference
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.indicators.market import MARKET_FEATURES
from nexus_alpha.targets import HORIZONS
from nexus_alpha.training.tabular import LGB_PARAMS, Prior, fit_lgbm, fit_logistic

FRAME = paths.ARTIFACTS / "samples" / "s500_frame.parquet"
EVAL_BLOCKS = ("valid", "calib", "test")


def main() -> int:
    fr = pd.read_parquet(FRAME)
    x33 = [c for c in fr.columns if c.startswith("x_")]
    struct = [c for c in STRUCTURED_ALL]
    mkt = [c for c in MARKET_FEATURES if c != "mkt_n"]
    sets = {"momentum_lr": ["d_ret1", "d_ret5", "d_ret21", "d_ret63"], "lgbm_structured": struct, "lgbm_x33": x33,
            "lgbm_struct_x33": struct + x33, "lgbm_struct_x33_mkt": struct + x33 + mkt}
    cfg = {"frame": str(FRAME), "feature_sets": {k: len(v) for k, v in sets.items()}, "lgb_params": LGB_PARAMS,
           "horizons": HORIZONS, "eval_blocks": EVAL_BLOCKS, "bootstrap": "month blocks, 500 resamples"}
    run = Run("e01_structured_baselines", cfg, inputs={"frame": FRAME})
    pred_dir = paths.ARTIFACTS / "predictions" / run.run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    keep = fr[fr.block.isin(EVAL_BLOCKS)][["sample_id", "ticker", "asof_date", "block"]].reset_index(drop=True)
    preds = {m: keep.copy() for m in ["prior"] + list(sets)}
    metrics, fitinfo = {}, {}
    try:
        for hname in HORIZONS:
            yc = f"y_{hname}"
            tr = fr[(fr.block == "train") & fr[yc].notna()]
            es = fr[(fr.block == "inner") & fr[yc].notna()]
            ev = fr[fr.block.isin(EVAL_BLOCKS)].reset_index(drop=True)
            ytr = tr[yc].to_numpy(int)
            P = {"prior": Prior().fit(ytr).predict_proba(len(ev))}
            P["momentum_lr"] = fit_logistic(tr[sets["momentum_lr"]], ytr)(ev[sets["momentum_lr"]])
            for m in ("lgbm_structured", "lgbm_x33", "lgbm_struct_x33", "lgbm_struct_x33_mkt"):
                f, _, info = fit_lgbm(tr[sets[m]], ytr, es[sets[m]], es[yc].to_numpy(int))
                P[m] = f(ev[sets[m]])
                fitinfo[f"{m}|{hname}"] = info
            for m, p in P.items():
                for k, cls in enumerate(("down", "side", "up")):
                    preds[m][f"p_{cls}_{hname}"] = p[:, k]
            for b in EVAL_BLOCKS:
                mask = (ev.block == b).to_numpy() & ev[yc].notna().to_numpy()
                y = ev.loc[mask, yc].to_numpy(int)
                blocks = month_blocks(ev.loc[mask, "asof_date"])
                for m, p in P.items():
                    rep = classification_report(y, p[mask])
                    if b == "test" and m != "prior":
                        rep["dlogloss_vs_prior"] = paired_difference(log_loss, y, p[mask], P["prior"][mask], blocks, n_boot=500)
                    metrics.setdefault(hname, {}).setdefault(b, {})[m] = rep
            print(hname, {m: round(metrics[hname]["test"][m]["log_loss"], 4) for m in P},
                  "| valid:", {m: round(metrics[hname]["valid"][m]["log_loss"], 4) for m in P}, flush=True)
        for m, df in preds.items():
            df.to_parquet(pred_dir / f"{m}.parquet", index=False)
        run.save("fit_info.json", fitinfo)
        best = min(sets, key=lambda m: np.mean([metrics[h]["valid"][m]["log_loss"] for h in HORIZONS]))
        summ = (f"valid-selected best structured model: {best}; test log loss by horizon "
                + ", ".join(f"{h} {metrics[h]['test'][best]['log_loss']:.4f} vs prior {metrics[h]['test']['prior']['log_loss']:.4f}" for h in HORIZONS))
        run.finish("COMPLETED", summ, metrics)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

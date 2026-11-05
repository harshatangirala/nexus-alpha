"""E05: walk-forward evaluation of the E01 validation-selected structured model (LightGBM on NEXUS indicators + TriFrame
33 features) with annual test folds 2008-2022, expanding training windows from 2003.

Fold Y: train = as-of 2003..Y-2, early stopping on Y-1, test = Y; every boundary purged by the 21-session label end plus a
5-session embargo (nexus_alpha.splits.walk_forward_blocks). Horizons 1D, 5D, 20D (CPU budget). Per fold: log loss of the
model and of the train-fitted class prior, up-vs-down AUROC, sideways AUROC. Inference across folds uses the 15 annual
values (each year is one observation).
Run: .venv/Scripts/python -I scripts/experiments/e05_walk_forward.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import classification_report
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.splits import WALK_FORWARD_TEST_YEARS, assign_blocks, walk_forward_blocks
from nexus_alpha.training.tabular import Prior, fit_lgbm

HORIZONS = ("1D", "5D", "20D")


def main() -> int:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    feats = list(STRUCTURED_ALL) + [c for c in fr.columns if c.startswith("x_")]
    run = Run("e05_walk_forward", {"model": "lgbm_struct_x33", "horizons": HORIZONS, "folds": WALK_FORWARD_TEST_YEARS,
                                   "num_threads": 2})
    rows = []
    try:
        for y in WALK_FORWARD_TEST_YEARS:
            blk = assign_blocks(fr.asof_date, walk_forward_blocks(y)).to_numpy()
            for h in HORIZONS:
                yc = f"y_{h}"
                tr = fr[(blk == "train") & fr[yc].notna().to_numpy()]
                va = fr[(blk == "valid") & fr[yc].notna().to_numpy()]
                te = fr[(blk == "test") & fr[yc].notna().to_numpy()]
                f, _, info = fit_lgbm(tr[feats], tr[yc].to_numpy(int), va[feats], va[yc].to_numpy(int), {"num_threads": 2})
                p = f(te[feats])
                pr = Prior().fit(tr[yc].to_numpy(int)).predict_proba(len(te))
                yt = te[yc].to_numpy(int)
                rm, rp = classification_report(yt, p), classification_report(yt, pr)
                rows.append({"test_year": y, "horizon": h, "n_train": len(tr), "n_test": len(te), "best_iter": info["best_iteration"],
                             "logloss": rm["log_loss"], "logloss_prior": rp["log_loss"],
                             "d_logloss": rp["log_loss"] - rm["log_loss"], "auc_up_vs_down": rm.get("auroc_up_vs_down"),
                             "auc_sideways": rm.get("auroc_SIDEWAYS"), "macro_f1": rm["macro_f1"]})
                print(f"{y} {h}: dLL {rows[-1]['d_logloss']:+.4f} up/down {rows[-1]['auc_up_vs_down']:.3f} "
                      f"side {rows[-1]['auc_sideways']:.3f}", flush=True)
        df = pd.DataFrame(rows)
        summ = {}
        for h in HORIZONS:
            g = df[df.horizon == h]
            t = sps.ttest_1samp(g.d_logloss, 0.0)
            tu = sps.ttest_1samp(g.auc_up_vs_down - 0.5, 0.0)
            summ[h] = {"mean_d_logloss": g.d_logloss.mean(), "years_improved": int((g.d_logloss > 0).sum()),
                       "t_years": float(t.statistic), "p_years": float(t.pvalue),
                       "mean_auc_up_vs_down": g.auc_up_vs_down.mean(), "years_up_vs_down_above_0.5": int((g.auc_up_vs_down > 0.5).sum()),
                       "t_up_vs_down": float(tu.statistic), "p_up_vs_down": float(tu.pvalue),
                       "mean_auc_sideways": g.auc_sideways.mean()}
        out = paths.REPO_ROOT / "reports" / "walk_forward"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "e05_fold_metrics.csv", index=False)
        run.save("fold_metrics.json", rows)
        txt = "; ".join(f"{h}: dLL {v['mean_d_logloss']:+.4f} ({v['years_improved']}/15 yrs), up-vs-down AUC "
                        f"{v['mean_auc_up_vs_down']:.3f} (p {v['p_up_vs_down']:.2f})" for h, v in summ.items())
        run.finish("COMPLETED", txt, {"summary": summ})
        print(txt)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

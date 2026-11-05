"""E10: negative and positive controls for the modelling pipeline (LightGBM, E07 best parameters, 1D/5D/20D).

* Placebo: labels permuted WITHIN each as-of date (class mix per date kept, any feature-label link destroyed), fitted with
  the same protocol. Expected: validation and test log loss no better than the class prior.
* Global placebo: labels permuted across ALL training rows (dates and stocks). Expected: no gain at all; the gap between
  the two placebos measures how much of the real gain is date-level (market-wide) timing rather than leakage.
* Canary (planted leak): one extra feature = the realised future return of the horizon plus noise (signal-to-noise 0.5).
  Expected: a large gain. If it were not found, the pipeline could not be trusted to find real signal either.
Output: experiments/<run>/metrics.json
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import log_loss
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.training.tabular import LGB_PARAMS


def main() -> int:
    e07 = sys.argv[1]
    best = json.loads((paths.REPO_ROOT / "experiments" / e07 / "metrics.json").read_text(encoding="utf-8"))["best_params"]
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    feats = list(STRUCTURED_ALL) + [c for c in fr.columns if c.startswith("x_")]
    blk = fr.block.to_numpy()
    run = Run("e10_controls", {"e07": e07, "params": best})
    rng = np.random.default_rng(7)
    res = {}
    for h in ("1D", "5D", "20D"):
        yc = f"y_{h}"
        ok = fr[yc].notna().to_numpy()
        y = fr[yc].to_numpy()
        y_perm = y.copy()
        for _, ix in fr[ok].groupby("asof_date").indices.items():
            rows = np.flatnonzero(ok)[ix]
            y_perm[rows] = rng.permutation(y[rows])
        y_glob = y.copy()
        y_glob[ok] = rng.permutation(y[ok])                 # full shuffle across dates: the true null
        canary = fr[f"ret_{h}"].to_numpy() + rng.normal(0, 2 * np.nanstd(fr[f"ret_{h}"]), len(fr))
        out = {}
        for name, labels, extra in (("real", y, None), ("placebo_within_date_shuffle", y_perm, None), ("placebo_global_shuffle", y_glob, None),
                                    ("canary_planted_leak", y, canary)):
            X = fr[feats].copy()
            if extra is not None:
                X["leak"] = extra
            tr, es = (blk == "train") & ok, (blk == "inner") & ok
            prm = {**LGB_PARAMS, "num_threads": 4, **best}
            dtr = lgb.Dataset(X[tr], labels[tr].astype(int))
            bst = lgb.train(prm, dtr, 3000, valid_sets=[lgb.Dataset(X[es], labels[es].astype(int), reference=dtr)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            prior = np.bincount(labels[tr].astype(int), minlength=3) / tr.sum()
            for b in ("valid", "test"):
                m = (blk == b) & ok
                p = bst.predict(X[m], num_iteration=bst.best_iteration)
                yy = y[m].astype(int)                      # always scored against the TRUE labels
                out.setdefault(name, {})[b] = {"log_loss": log_loss(yy, p), "prior_log_loss": log_loss(yy, np.tile(prior, (m.sum(), 1))),
                                               "best_iteration": int(bst.best_iteration)}
        res[h] = out
        print(h, {k: round(v["test"]["log_loss"] - v["test"]["prior_log_loss"], 4) for k, v in out.items()}, flush=True)
    run.finish("COMPLETED", "; ".join(f"{h}: test LL minus prior real {v['real']['test']['log_loss'] - v['real']['test']['prior_log_loss']:+.4f}, "
                                       f"placebo {v['placebo_within_date_shuffle']['test']['log_loss'] - v['placebo_within_date_shuffle']['test']['prior_log_loss']:+.4f}, "
                                       f"canary {v['canary_planted_leak']['test']['log_loss'] - v['canary_planted_leak']['test']['prior_log_loss']:+.4f}"
                                       for h, v in res.items()), res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

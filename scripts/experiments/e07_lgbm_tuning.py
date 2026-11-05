"""E07: LightGBM (numbers only: NEXUS indicators + TriFrame-33) hyperparameter search, curves and refit (post-registration,
exploratory; selection on validation only).

1. Optuna TPE, 25 trials, objective = mean validation (2017-18) log loss over 1D/5D/20D, each model early-stopped on the
   2016 inner block; the full trial table is kept (search size counts for multiple testing).
2. Final per-horizon models with the best parameters on all six horizons. The number of trees comes from early stopping
   on the 2016 inner block ONLY; a second fit with that number + 100 fixed rounds (no early stopping) records the
   per-iteration log loss on the training rows, the inner block and the validation block (validation curves).
   Audit fix A-21: v1 of this script recorded the curves in the early-stopped fit itself, and LightGBM's early stopping
   watches every non-training set, so the 2017-18 block could end the run (it did at 1D and 2D) and cap the tree count.
3. Learning curves: 10/25/50/100% of the training rows (time-ordered subsampling of stocks) vs validation log loss.
4. Gain importance per horizon.
5. Refit variant: train on 2003-2018 (train+inner+valid) with the early-stopped number of trees scaled by the row ratio;
   calibrate on 2019 later (E08); scored on 2020-22 only as the historical diagnostic.
Run: .venv/Scripts/python -I scripts/experiments/e07_lgbm_tuning.py [--reuse-search <earlier E07 run id>]
     (--reuse-search takes the best parameters from an earlier run's Optuna search instead of searching again; the search
     itself only ever early-stopped on the inner block, so it is unaffected by A-21)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import log_loss
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.core import STRUCTURED_ALL
from nexus_alpha.targets import HORIZONS
from nexus_alpha.training.tabular import LGB_PARAMS

H = list(HORIZONS)
TUNE_H = ("1D", "5D", "20D")
THREADS = 3


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-search", default=None)
    args = ap.parse_args()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    feats = list(STRUCTURED_ALL) + [c for c in fr.columns if c.startswith("x_")]
    blk = fr.block.to_numpy()
    run = Run("e07_lgbm_tuning", {"features": len(feats), "trials": 25, "tune_horizons": TUNE_H, "threads": THREADS,
                                  "reuse_search": args.reuse_search, "stopping": "inner 2016 only; curves from a fixed-rounds refit"})
    out = paths.ARTIFACTS / "predictions" / run.run_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def data(h, block_mask):
        m = block_mask & fr[f"y_{h}"].notna().to_numpy()
        return fr.loc[m, feats], fr.loc[m, f"y_{h}"].to_numpy(int)

    def fit(params, h, train_mask, rounds=None, record=False):
        Xtr, ytr = data(h, train_mask)
        Xes, yes = data(h, blk == "inner")
        Xva, yva = data(h, blk == "valid")
        prm = {**LGB_PARAMS, "num_threads": THREADS, **params}
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        sets = [lgb.Dataset(Xes, yes, reference=dtr)]
        names = ["inner"]
        if record:
            sets = [dtr, sets[0], lgb.Dataset(Xva, yva, reference=dtr)]
            names = ["train", "inner", "valid"]
        ev = {}
        cbs = [lgb.record_evaluation(ev)]
        if rounds is None:
            cbs.append(lgb.early_stopping(100, verbose=False))
        bst = lgb.train(prm, dtr, num_boost_round=rounds or 3000, valid_sets=sets, valid_names=names, callbacks=cbs)
        return bst, ev

    def objective(trial):
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 3000, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 50.0, log=True)}
        ll = []
        for h in TUNE_H:
            bst, _ = fit(params, h, blk == "train")
            Xva, yva = data(h, blk == "valid")
            ll.append(log_loss(yva, bst.predict(Xva, num_iteration=bst.best_iteration)))
        return float(np.mean(ll))

    try:
        if args.reuse_search:
            prev = json.loads((paths.REPO_ROOT / "experiments" / args.reuse_search / "metrics.json").read_text(encoding="utf-8"))
            best, best_value, default_value, n_trials = prev["best_params"], prev["best_value"], prev["default_value"], prev["n_trials"]
            print(f"reusing the search of {args.reuse_search}: {best_value:.5f} (default {default_value:.5f}) {best}", flush=True)
        else:
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
            study.enqueue_trial({k: LGB_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                                                           "bagging_fraction", "lambda_l2")})   # trial 0 = the E01 default
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=25)
            trials = study.trials_dataframe()[["number", "value"] + [c for c in study.trials_dataframe().columns if c.startswith("params_")]]
            trials.to_csv(paths.REPO_ROOT / "reports" / "e07_optuna_trials.csv", index=False)
            best, best_value, default_value, n_trials = study.best_params, study.best_value, study.trials[0].value, len(study.trials)
            print(f"best trial {study.best_trial.number}: {best_value:.5f} (default {default_value:.5f}) {best} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        res = {"best_params": best, "best_value": best_value, "default_value": default_value, "search_from": args.reuse_search or "this run",
               "n_trials": n_trials, "curves": {}, "importance": {}, "learning_curves": {}, "scores": {}}
        P = {k: np.full((len(fr), len(H), 3), np.nan, np.float32) for k in ("tuned", "refit")}
        for j, h in enumerate(H):
            bst0, _ = fit(best, h, blk == "train")                          # early stopping on the 2016 inner block only
            bi = int(bst0.best_iteration)
            bst, ev = fit(best, h, blk == "train", rounds=bi + 100, record=True)   # fixed rounds: curves only, no stopping
            assert int(np.argmin(ev["inner"]["multi_logloss"])) + 1 == bi, "inner minimum moved between identical fits"
            res["curves"][h] = {"train": ev["train"]["multi_logloss"], "inner": ev["inner"]["multi_logloss"],
                                "valid": ev["valid"]["multi_logloss"], "best_iteration_inner": bi,
                                "best_iteration_valid": int(np.argmin(ev["valid"]["multi_logloss"])) + 1}
            P["tuned"][:, j] = bst.predict(fr[feats], num_iteration=bi)
            gain = pd.Series(bst.feature_importance("gain", iteration=bi), index=feats)
            res["importance"][h] = (gain / gain.sum()).sort_values(ascending=False).round(5).to_dict()
            n_old = int((blk == "train").sum())
            n_new = int(np.isin(blk, ["train", "inner", "valid"]).sum())
            bst_r, _ = fit(best, h, np.isin(blk, ["train", "inner", "valid"]), rounds=max(1, int(round(bi * n_new / n_old))))
            P["refit"][:, j] = bst_r.predict(fr[feats])
            for name in ("tuned", "refit"):
                for bl in ("valid", "calib", "test"):
                    m = (blk == bl) & fr[f"y_{h}"].notna().to_numpy()
                    res["scores"].setdefault(name, {}).setdefault(h, {})[bl] = log_loss(fr.loc[m, f"y_{h}"].to_numpy(int), P[name][m, j])
            print(f"{h}: best iter {bi}, valid {res['scores']['tuned'][h]['valid']:.5f}, test tuned "
                  f"{res['scores']['tuned'][h]['test']:.5f} refit {res['scores']['refit'][h]['test']:.5f} ({time.time() - t0:.0f} s)", flush=True)
        # learning curves (stocks subsampled, all their training rows kept: no within-stock time leakage)
        tick = np.array(sorted(fr.ticker.unique()))
        rng = np.random.default_rng(7)
        order = rng.permutation(tick)
        for frac in (0.1, 0.25, 0.5, 1.0):
            keep = set(order[: max(1, int(round(frac * len(tick))))])
            msk = (blk == "train") & fr.ticker.isin(keep).to_numpy()
            for h in TUNE_H:
                bst, _ = fit(best, h, msk)
                Xva, yva = data(h, blk == "valid")
                res["learning_curves"].setdefault(h, {})[str(frac)] = {
                    "train_rows": int((msk & fr[f"y_{h}"].notna().to_numpy()).sum()),
                    "valid_log_loss": log_loss(yva, bst.predict(Xva, num_iteration=bst.best_iteration))}
        for name, arr in P.items():
            np.save(out / f"lgbm_{name}__probs.npy", arr)
        fr[["sample_id", "block"]].to_parquet(out / "rows.parquet", index=False)
        mv = {n: {bl: float(np.mean([res["scores"][n][h][bl] for h in H])) for bl in ("valid", "calib", "test")} for n in ("tuned", "refit")}
        res["mean_scores"] = mv
        summ = (f"Optuna {n_trials} trials{' (reused ' + args.reuse_search + ')' if args.reuse_search else ''}: mean valid LL (1D/5D/20D) {best_value:.5f} vs default {default_value:.5f}; "
                f"six-horizon mean: tuned valid {mv['tuned']['valid']:.5f} test {mv['tuned']['test']:.5f}; refit test {mv['refit']['test']:.5f}")
        run.finish("COMPLETED", summ, res)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

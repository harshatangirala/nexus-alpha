"""Tabular branch models: class prior, momentum logistic baseline, LightGBM multiclass (per horizon).

Fitting rules: every scaler, imputer and model is fitted on the training block only; LightGBM early-stops on the
`inner` block (2016), never on the selection (`valid`) or later blocks. Settings are fixed in configs/training.yaml and
not tuned on evaluation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 5.0, "verbose": -1, "deterministic": True, "force_row_wise": True, "seed": 7,
              "num_threads": 6}
MAX_ROUNDS = 1500
EARLY_STOP = 100


class Prior:
    def fit(self, y):
        self.p = np.bincount(y.astype(int), minlength=3) / len(y)
        return self

    def predict_proba(self, n):
        return np.tile(self.p, (n, 1))


class ImputeScale:
    """Train-median imputation plus missing indicators, then standardisation (train statistics only)."""

    def fit(self, X: pd.DataFrame):
        self.cols = list(X.columns)
        self.med = X.median()
        self.miss_cols = [c for c in self.cols if X[c].isna().any()]
        Z = self._z(X)
        self.sc = StandardScaler().fit(Z)
        return self

    def _z(self, X):
        Z = X[self.cols].fillna(self.med)
        for c in self.miss_cols:
            Z[f"{c}__missing"] = X[c].isna().astype(float)
        return Z.to_numpy(float)

    def transform(self, X):
        return np.clip(self.sc.transform(self._z(X)), -8, 8)


def fit_logistic(Xtr: pd.DataFrame, ytr: np.ndarray, C: float = 0.1):
    pre = ImputeScale().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(pre.transform(Xtr), ytr.astype(int))
    return lambda X: m.predict_proba(pre.transform(X))


def fit_lgbm(Xtr: pd.DataFrame, ytr: np.ndarray, Xes: pd.DataFrame, yes: np.ndarray, params: dict | None = None):
    import lightgbm as lgb
    prm = {**LGB_PARAMS, **(params or {})}
    dtr = lgb.Dataset(Xtr, label=ytr.astype(int), free_raw_data=False)
    des = lgb.Dataset(Xes, label=yes.astype(int), reference=dtr, free_raw_data=False)
    booster = lgb.train(prm, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[des], valid_names=["inner"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    info = {"best_iteration": int(booster.best_iteration), "inner_logloss": float(booster.best_score["inner"]["multi_logloss"])}
    predict = lambda X: booster.predict(X, num_iteration=booster.best_iteration)
    return predict, booster, info

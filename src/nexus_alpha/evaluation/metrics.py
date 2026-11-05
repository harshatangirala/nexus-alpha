"""Classification, probability and return metrics (master spec 11.2-11.3). Class order is (DOWN, SIDEWAYS, UP).

Calibration error: top-label ECE with 15 equal-width confidence bins, plus class-wise ECE (mean over classes of the
equal-width-binned |mean predicted - observed frequency|). Both are documented in docs/VALIDATION_AND_BACKTESTING.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

CLASSES = ("DOWN", "SIDEWAYS", "UP")
EPS = 1e-12
ECE_BINS = 15


def check_probs(p: np.ndarray, tol: float = 1e-6) -> None:
    p = np.asarray(p, float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"probabilities must be (n, 3), got {p.shape}")
    if not np.isfinite(p).all() or (p < -tol).any() or (p > 1 + tol).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.abs(p.sum(1) - 1).max() > tol:
        raise ValueError(f"probabilities must sum to 1 within {tol}")


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    oh = np.eye(3)[y.astype(int)]
    return float(np.mean(((np.asarray(p, float) - oh) ** 2).sum(1)))


def ece_top_label(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    conf = p.max(1)
    pred = p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def ece_classwise(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for k in range(3):
        pk = p[:, k]
        yk = (y == k).astype(float)
        idx = np.clip(np.digitize(pk, edges[1:-1]), 0, bins - 1)
        e = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                e += m.mean() * abs(yk[m].mean() - pk[m].mean())
        out.append(e)
    return float(np.mean(out))


def reliability_table(y: np.ndarray, p: np.ndarray, k: int, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    pk = p[:, k]
    idx = np.clip(np.digitize(pk, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                     "mean_pred": float(pk[m].mean()) if m.any() else np.nan,
                     "obs_freq": float((y[m] == k).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def classification_report(y: np.ndarray, p: np.ndarray) -> dict:
    """All per-horizon classification and probability metrics for one model on one evaluation set."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    check_probs(p)
    pred = p.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
           "log_loss": log_loss(y, p), "brier": brier(y, p), "ece_top": ece_top_label(y, p), "ece_classwise": ece_classwise(y, p),
           "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
    present = [k for k in range(3) if 0 < (y == k).sum() < len(y)]
    aucs, aps = {}, {}
    for k in range(3):
        name = CLASSES[k]
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = float(prec[k]), float(rec[k]), float(f1[k])
        out[f"support_{name}"] = int(sup[k])
        out[f"pred_count_{name}"] = int((pred == k).sum())
        if k in present:
            aucs[name] = float(roc_auc_score((y == k).astype(int), p[:, k]))
            aps[name] = float(average_precision_score((y == k).astype(int), p[:, k]))
            out[f"auroc_{name}"], out[f"prauc_{name}"] = aucs[name], aps[name]
            out[f"base_rate_{name}"] = float((y == k).mean())
    out["auroc_macro"] = float(np.mean(list(aucs.values()))) if aucs else np.nan
    out["prauc_macro"] = float(np.mean(list(aps.values()))) if aps else np.nan
    # direction among non-sideways outcomes: does P(UP) - P(DOWN) rank UP above DOWN?
    nd = y != 1
    if nd.sum() > 10 and 0 < (y[nd] == 2).mean() < 1:
        out["auroc_up_vs_down"] = float(roc_auc_score((y[nd] == 2).astype(int), p[nd, 2] - p[nd, 0]))
    return out


def rank_ic_by_date(dates, score, realized) -> pd.Series:
    """Spearman correlation between score and realized return within each date (dates with >= 20 names)."""
    df = pd.DataFrame({"d": pd.to_datetime(dates), "s": score, "r": realized}).dropna()
    def ic(g):
        if len(g) < 20:
            return np.nan
        return g.s.rank().corr(g.r.rank())
    return df.groupby("d")[["s", "r"]].apply(ic).dropna()


# ------------------------------------------------------------------------------------------------ return heads
def regression_report(y: np.ndarray, yhat: np.ndarray, train_mean: float) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    resid = y - yhat
    sse = float((resid ** 2).sum())
    return {"n": int(len(y)), "mae": float(np.abs(resid).mean()), "rmse": float(np.sqrt((resid ** 2).mean())),
            "r2_oos_vs_train_mean": float(1 - sse / ((y - train_mean) ** 2).sum()),
            "r2_vs_eval_mean": float(1 - sse / ((y - y.mean()) ** 2).sum()),
            "spearman": float(pd.Series(y).rank().corr(pd.Series(yhat).rank())),
            "sign_agreement": float((np.sign(y) == np.sign(yhat)).mean())}


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_report(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, nominal: float) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {"nominal": nominal, "coverage": float(inside.mean()), "mean_width": float(np.mean(hi - lo))}

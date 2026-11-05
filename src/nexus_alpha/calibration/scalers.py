"""Multiclass calibrators working on log-probabilities.

* TemperatureScaler: p' = softmax(log p / T), one T > 0 per horizon (keeps the argmax).
* VectorScaler: p' = softmax(a * log p + b) with per-class a and b (Guo et al. vector scaling), lightly L2-regularised
  towards the identity.
`fit_best` chooses between none / temperature / vector by log loss on a held-out half of the calibration block
(chronological halves), then refits the chosen method on the whole calibration block. Isotonic is not used: TriFrame's
audit found it worsened log loss (0.590 -> 0.604) by producing exact 0/1 probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..evaluation.metrics import log_loss

EPS = 1e-9


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TemperatureScaler:
    name = "temperature"

    def fit(self, p: np.ndarray, y: np.ndarray):
        lp = np.log(np.clip(p, EPS, 1))
        f = lambda t: log_loss(y, _softmax(lp / np.exp(t[0])))
        self.T = float(np.exp(minimize(f, [0.0], method="Nelder-Mead").x[0]))
        return self

    def transform(self, p):
        return _softmax(np.log(np.clip(p, EPS, 1)) / self.T)


class VectorScaler:
    name = "vector"

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2

    def fit(self, p, y):
        lp = np.log(np.clip(p, EPS, 1))
        def f(w):
            a, b = w[:3], w[3:]
            return log_loss(y, _softmax(a * lp + b)) + self.l2 * (((a - 1) ** 2).sum() + (b ** 2).sum())
        w = minimize(f, np.r_[np.ones(3), np.zeros(3)], method="L-BFGS-B").x
        self.a, self.b = w[:3], w[3:]
        return self

    def transform(self, p):
        return _softmax(self.a * np.log(np.clip(p, EPS, 1)) + self.b)


class Identity:
    name = "none"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return p


def fit_best(p_cal: np.ndarray, y_cal: np.ndarray, order: np.ndarray | None = None) -> tuple[object, dict]:
    """Choose a calibrator on chronological halves of the calibration block; refit the winner on all of it."""
    n = len(y_cal)
    order = np.arange(n) if order is None else order
    a, b = order[: n // 2], order[n // 2:]
    scores = {}
    for C in (Identity, TemperatureScaler, VectorScaler):
        c = C().fit(p_cal[a], y_cal[a])
        scores[C.name] = log_loss(y_cal[b], c.transform(p_cal[b]))
    best = min(scores, key=scores.get)
    cls = {"none": Identity, "temperature": TemperatureScaler, "vector": VectorScaler}[best]
    return cls().fit(p_cal, y_cal), {"half_split_logloss": scores, "chosen": best}

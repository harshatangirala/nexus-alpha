"""E26 forward benchmark statistics (pre-registered in docs/V3_RESEARCH.md): NEXUS's P(move) against option-implied
volatility on the binary label "move" (the 3-class label is not sideways).

Input: one row per scored, on-time forecast with an implied volatility of the same session: asof_session, p_side
(NEXUS), iv30, real_tau (the label rule's threshold), real_y (0 down, 1 sideways, 2 up), and the horizon length.
* Endpoint 1 (descriptive): paired per-row binary log loss, IV minus NEXUS, date-clustered bootstrap.
* Endpoint 2 (decides): a logistic on logit(P_IV) and logit(P_NEXUS), fitted on the first half of the sessions and
  scored on the second half, against the IV-only logistic fitted and scored the same way; paired log-loss difference,
  date-clustered bootstrap over the second half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..options import p_sideways

EPS = 1e-6
MIN_SESSIONS = 120


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _draws(dates: np.ndarray, d: np.ndarray, n_boot: int = 1000, seed: int = 11) -> np.ndarray:
    """Date-clustered bootstrap of the mean: whole dates are resampled."""
    g = pd.DataFrame({"t": dates, "d": d}).groupby("t").d.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    return np.array([s[ix].sum() / c[ix].sum() for ix in (rng.integers(0, len(s), len(s)) for _ in range(n_boot))])


def _boot(dates: np.ndarray, d: np.ndarray, n_boot: int = 1000, seed: int = 11) -> list[float]:
    x = _draws(dates, d, n_boot, seed)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure (D-051): which hypotheses are rejected at family-wise level alpha."""
    order = sorted(pvals, key=lambda k: pvals[k])
    out, m = {k: False for k in pvals}, len(pvals)
    for i, k in enumerate(order):
        if pvals[k] <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def prepare(rows: pd.DataFrame, sessions: int) -> pd.DataFrame:
    r = rows.dropna(subset=["iv30", "real_tau", "real_y", "p_side"]).copy()
    r["p_iv"] = [1 - p_sideways(iv, tau, sessions) for iv, tau in zip(r.iv30, r.real_tau, strict=True)]
    r["p_nx"] = 1 - r.p_side.astype(float)
    r["move"] = (r.real_y.astype(int) != 1).astype(float)
    return r.dropna(subset=["p_iv"])


def evaluate(rows: pd.DataFrame, sessions: int, min_sessions: int = MIN_SESSIONS) -> dict:
    r = prepare(rows, sessions)
    dates = np.sort(r.asof_session.unique())
    out = {"rows": int(len(r)), "sessions": int(len(dates)), "move_rate": float(r.move.mean()) if len(r) else None}
    if len(dates) < 2:
        out["decision"] = f"insufficient: {len(dates)} scored sessions (needs {min_sessions})"
        return out
    d1 = _ll(r.p_iv.to_numpy(), r.move.to_numpy()) - _ll(r.p_nx.to_numpy(), r.move.to_numpy())
    out["endpoint1_ll_iv_minus_nexus"] = {"mean": float(d1.mean()), "ci95": _boot(r.asof_session.to_numpy(), d1)}
    half = dates[: len(dates) // 2]
    tr, te = r[r.asof_session.isin(half)], r[~r.asof_session.isin(half)]
    if tr.move.nunique() < 2 or len(te) == 0:
        out["decision"] = "insufficient: one class only in the first half"
        return out
    Xi = lambda d: _logit(d.p_iv.to_numpy())[:, None]                                 # noqa: E731
    Xb = lambda d: np.c_[_logit(d.p_iv.to_numpy()), _logit(d.p_nx.to_numpy())]       # noqa: E731
    mi = LogisticRegression(C=1e6, max_iter=1000).fit(Xi(tr), tr.move)
    mb = LogisticRegression(C=1e6, max_iter=1000).fit(Xb(tr), tr.move)
    d2 = _ll(mi.predict_proba(Xi(te))[:, 1], te.move.to_numpy()) - _ll(mb.predict_proba(Xb(te))[:, 1], te.move.to_numpy())
    draws = _draws(te.asof_session.to_numpy(), d2)
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    out["endpoint2_ll_ivonly_minus_both"] = {"mean": float(d2.mean()), "ci95": ci,
                                            "p_one_sided": float(np.mean(draws <= 0)), "train_sessions": int(len(half)),
                                            "test_sessions": int(len(dates) - len(half)),
                                            "coef_both": [float(x) for x in mb.coef_[0]]}
    if len(dates) < min_sessions:
        out["decision"] = f"insufficient: {len(dates)} scored sessions (needs {min_sessions}); figures are progress only"
    else:
        out["decision"] = ("NEXUS adds to implied volatility" if ci[0] > 0
                           else "NEXUS's volatility skill is already priced")
    return out

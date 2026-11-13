"""E29a: prior-shift (label-shift) recalibration of a frozen model's probabilities, point in time (D-058).

Audit finding A-85: calibration fitted on 2019 does not transfer, because the share of DOWN / SIDEWAYS / UP outcomes
moves between regimes. The standard correction for a change in class priors (Saerens, Latinne and Decaestecker, 2002)
reweights each probability by the ratio of the new to the old class prior and renormalises:

    p'_k  proportional to  p_k * pi_new_k / pi_old_k

Here, with nothing fitted and nothing chosen after seeing an outcome:
* pi_new = the realised class shares of the model's own forward rows of the same horizon whose outcome was known
  before this forecast's decision time (the label's exit session closed, 16:00 New York, before the decision time),
  over the trailing WINDOW_SESSIONS decision dates;
* pi_old = the mean predicted probabilities of those same rows (the prior the model implied there);
* with fewer than MIN_ROWS such rows, no correction (the raw probabilities).

Every input is logged before the decision time or matured before it, so the corrected forecast is computable from the
forward logs at any later time without look-ahead. The constants are fixed by the pre-registration (docs/E29_PREREGISTRATION.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SESSIONS = 60
MIN_ROWS = 2000
CLIP = (0.25, 4.0)            # a single ratio never scales a class by more than 4x (pre-registered guard)


def prior_shift(p: np.ndarray, pi_old: np.ndarray, pi_new: np.ndarray) -> np.ndarray:
    """p: (n, 3) probabilities (DOWN, SIDEWAYS, UP). Returns the prior-shifted, renormalised probabilities."""
    w = np.clip(np.asarray(pi_new, float) / np.clip(np.asarray(pi_old, float), 1e-9, None), *CLIP)
    q = np.asarray(p, float) * w[None, :]
    return q / q.sum(1, keepdims=True)


def label_known_utc(end_session) -> pd.Series:
    """When a label is known: the close (16:00 New York) of its exit session, in UTC."""
    d = pd.to_datetime(pd.Series(end_session)).dt.strftime("%Y-%m-%d")
    return pd.to_datetime(d + " 16:00").dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def trailing_priors(scored: pd.DataFrame, decision_utc: pd.Timestamp, horizon: str):
    """(pi_old, pi_new, n_rows) from rows of `horizon` known before `decision_utc`, or None if too few.
    scored: forward rows with p_down, p_side, p_up, real_y (0/1/2), real_end_session, decision_utc, horizon."""
    s = scored[(scored.horizon == horizon) & scored.real_y.notna()]
    if s.empty:
        return None
    dec = pd.Timestamp(decision_utc)
    dec = dec.tz_localize("UTC") if dec.tzinfo is None else dec.tz_convert("UTC")
    known = (label_known_utc(s.real_end_session) < dec).to_numpy()
    s = s[known]
    if s.empty:
        return None
    dates = sorted(pd.to_datetime(s.decision_utc, utc=True).unique())[-WINDOW_SESSIONS:]
    s = s[pd.to_datetime(s.decision_utc, utc=True).isin(dates)]
    if len(s) < MIN_ROWS:
        return None
    y = s.real_y.astype(int).to_numpy()
    pi_new = np.bincount(y, minlength=3) / len(y)
    pi_old = s[["p_down", "p_side", "p_up"]].to_numpy(float).mean(0)
    return pi_old, pi_new, len(s)


def corrected(scored: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """E29a probabilities for `rows` (forward forecasts), each corrected only with labels known before its decision time.
    Returns rows with p_down_ls, p_side_ls, p_up_ls and n_prior_rows (0 = uncorrected)."""
    out = rows.copy()
    P = out[["p_down", "p_side", "p_up"]].to_numpy(float)
    Q = P.copy(); n = np.zeros(len(out), int)
    for (dec, h), idx in out.groupby(["decision_utc", "horizon"]).indices.items():
        pri = trailing_priors(scored, pd.Timestamp(dec), h)
        if pri is not None:
            Q[idx] = prior_shift(P[idx], pri[0], pri[1]); n[idx] = pri[2]
    out[["p_down_ls", "p_side_ls", "p_up_ls"]] = Q
    out["n_prior_rows"] = n
    return out

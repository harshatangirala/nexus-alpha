"""Kronos for one stock (Trade Desk): sampled future paths and label-rule probabilities, with the E24 configuration.

The configuration is the single one pre-registered for E24 (docs/V3_RESEARCH.md); tests/test_kronos.py checks that
this module, the E24 script and the frozen `kronos_zs` bundle use the same values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine as KE, weights as W

CONFIG = {"model": "Kronos-small", "lookback": 400, "paths": 32, "T": 1.0, "top_p": 1.0, "horizon_sessions": 20}
HORIZONS = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "15D": 15, "20D": 20}
PRETRAIN_END = pd.Timestamp("2024-06-30")       # paper, p. 14: "pre-training data ... extends up to June 2024"
_MODEL: dict = {}


def model():
    if "m" not in _MODEL:
        _MODEL["m"] = W.load(CONFIG["model"])
    return _MODEL["m"]


def forecast(ticker: str, daily: pd.DataFrame) -> dict | None:
    """daily: bars up to and including the as-of session (its last row). Returns None when history is too short."""
    from ..data.calendar import offset_session
    from ..indicators.core import ewma_sigma
    d = daily.sort_values("date").reset_index(drop=True)
    k = len(d) - 1
    c = KE.context(d, k, CONFIG["lookback"], CONFIG["horizon_sessions"])
    if c is None:
        return None
    m, tok, mc = model()
    asof = pd.Timestamp(d.date.iloc[k])
    P = KE.sample_paths(m, tok, [c], [KE.series_seed(ticker, asof)], CONFIG["paths"], CONFIG["horizon_sessions"], mc,
                        T=CONFIG["T"], top_p=CONFIG["top_p"])
    sigma = float(ewma_sigma(d.close).iloc[k])
    pr = KE.label_probabilities(P, np.array([sigma]), HORIZONS)
    fut = pd.DatetimeIndex([offset_session(pd.DatetimeIndex([asof]), i)[0] for i in range(1, CONFIG["horizon_sessions"] + 1)])
    return {"asof": asof, "dates": fut, "paths": P[0], "sigma": sigma, "close": float(d.close.iloc[k]),
            "probs": {h: {k_: (v[0] if np.ndim(v) else v) for k_, v in o.items()} for h, o in pr.items()},
            "history": d.tail(60)[["date", "close"]], "in_pretraining": bool(asof <= PRETRAIN_END)}

"""Multi-horizon targets (docs/LABEL_ENGINEERING.md, pre-registered as D-005).

For a sample whose last completed session is t (decision 09:00 New York on session t+1, entry at the open of t+1):

    R_h   = close[t+h] / open[t+1] - 1                       (price only, split-adjusted; dividends not included)
    tau_h = max(0.5 * sigma_t * sqrt(h), 10 bps)             (sigma_t = EWMA daily return vol, lambda 0.94, known at t)
    y_h   = UP if R_h > tau_h, DOWN if R_h < -tau_h, else SIDEWAYS
    end_h = session t+h (the label window is [open t+1, close t+h])

for h in {1, 2, 5, 10, 20, 21} sessions ("1M" = 21 sessions). The rule is TriFrame's frozen 5-session rule (tf_data.labels)
generalised to all horizons; it was fixed before any NEXUS model was fitted and is never tuned for class balance.
Also stored: z_h = R_h / (sigma_t sqrt h), the volatility-scaled return used by regression/quantile heads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.calendar import offset_session
from .indicators.core import ewma_sigma

HORIZONS = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "20D": 20, "1M": 21}
MAX_H = max(HORIZONS.values())
CLASSES = ("DOWN", "SIDEWAYS", "UP")          # class index 0, 1, 2
COST_FLOOR = 0.0010
K_SIGMA = 0.5
LABEL_VERSION = "nexus-labels-v1"


def make_labels(daily: pd.DataFrame) -> pd.DataFrame:
    """Labels for every session of one stock (rows aligned with `daily` sorted by date). NaN where the window is
    incomplete, including every window that would need bars on/after the seal (they are not in the sealed bars)."""
    d = daily.sort_values("date").reset_index(drop=True)
    c, o = d.close.astype(float), d.open.astype(float)
    sigma = ewma_sigma(c)
    o1 = o.shift(-1)
    out = pd.DataFrame({"date": d.date.values, "sigma_ewma": sigma.values, "entry_open": o1.values})
    for name, h in HORIZONS.items():
        r = c.shift(-h) / o1 - 1
        tau = np.maximum(K_SIGMA * sigma * np.sqrt(h), COST_FLOOR)
        y = np.select([r > tau, r < -tau], [2, 0], default=1).astype(float)
        y[r.isna().values] = np.nan
        out[f"ret_{name}"] = r.values
        out[f"tau_{name}"] = tau.values
        out[f"y_{name}"] = y
        out[f"z_{name}"] = (r / (sigma * np.sqrt(h))).values
        # label-window end from the exchange calendar (defined even where the sealed bars stop)
        out[f"end_{name}"] = offset_session(d.date, h).values
    return out


def label_columns(kind: str = "y") -> list[str]:
    return [f"{kind}_{k}" for k in HORIZONS]

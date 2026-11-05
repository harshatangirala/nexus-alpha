"""v2 targets: the dashboard's trade types (intraday, swing of 2-20 sessions, long-term 4 weeks to 1 year).

v1 targets (targets.py) are frozen with the v1.1 model and are not changed. v2 keeps the same rule and adds horizons:

* 1D ... 20D:      R_h = close[t+h] / open[t+1] - 1, threshold max(0.5 * sigma_t * sqrt(h), 10 bps) (the v1 rule).
                   1D is the intraday trade (entry at the open of t+1, exit at the close of the same session); a swing of
                   N trading days is ND; "4 weeks" in the dashboard is 20D. (A separate intraday label would duplicate 1D:
                   it was dropped on 2026-11-06 before any fit.)
* 3M, 6M, 1Y:      the same rule with h = 63, 126, 252 sessions.

Purging: v1 purges whole rows using the longest v1 window (21 sessions). The long horizons would cost years of rows under
that rule, so v2 purges per horizon instead: in every block except the last, a label is set to NaN when its window end
plus the 5-session embargo reaches the next block's first session (`purge_per_horizon`). Labels needing bars on/after
the seal are NaN because the sealed readers never return those bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.calendar import offset_session
from .indicators.core import ewma_sigma
from .splits import EMBARGO_SESSIONS, PRIMARY, _first_session_on_or_after
from .targets import COST_FLOOR, K_SIGMA

SWING = {f"{h}D": h for h in range(1, 21)}
LONG = {"3M": 63, "6M": 126, "1Y": 252}
HORIZONS_V2 = {**SWING, **LONG}                  # value = sessions until the exit session
LABEL_VERSION_V2 = "nexus-labels-v2"
TRADE_TYPES = {"Intraday": ["1D"], "Swing": [f"{h}D" for h in range(2, 21)], "Long-term": ["20D", "3M", "6M", "1Y"]}


def make_labels_v2(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True)
    c, o = d.close.astype(float), d.open.astype(float)
    sigma = ewma_sigma(c)
    o1 = o.shift(-1)
    out = {"date": d.date.values, "sigma_ewma": sigma.values}
    for name, h in HORIZONS_V2.items():
        r = c.shift(-h) / o1 - 1
        scale = sigma * np.sqrt(h)
        tau = np.maximum(K_SIGMA * scale, COST_FLOOR)
        y = np.select([r > tau, r < -tau], [2, 0], default=1).astype(float)
        y[r.isna().values] = np.nan
        out[f"y_{name}"] = y
        out[f"z_{name}"] = (r / scale).values
        out[f"ret_{name}"] = r.values
        out[f"end_{name}"] = offset_session(d.date, h).values
    return pd.DataFrame(out)


def purge_per_horizon(frame: pd.DataFrame, asof_col: str = "asof_date", block_col: str = "block",
                      blocks=PRIMARY, embargo: int = EMBARGO_SESSIONS) -> pd.DataFrame:
    """NaN-out labels whose window (+embargo) reaches the next block. Returns a copy; counts in .attrs['purged']."""
    f = frame.copy()
    purged = {}
    for i, b in enumerate(blocks[:-1]):
        nxt = _first_session_on_or_after(blocks[i + 1].start)
        rows = f[block_col] == b.name
        asof = pd.to_datetime(f.loc[rows, asof_col])
        for name, h in HORIZONS_V2.items():
            limit = offset_session(pd.Series(asof.values), h + embargo)   # window end + embargo, from the calendar
            bad = rows.copy()
            bad.loc[rows] = pd.isna(limit) | (limit >= nxt)
            n = int((bad & f[f"y_{name}"].notna()).sum())
            if n:
                purged[f"{b.name}:{name}"] = n
            f.loc[bad, [f"y_{name}", f"z_{name}", f"ret_{name}"]] = np.nan
    f.attrs["purged"] = purged
    return f

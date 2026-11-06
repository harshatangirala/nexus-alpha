"""Classic market-regime rule features (E20, pre-registered in docs/V2_ARCHITECTURE.md), causal per session.

Definitions follow tradermonty/claude-trading-skills (MIT): `ibd-distribution-day-monitor` (config `ibd_dd_v1.0`) and
`ftd-detector` (its `rally_tracker.py` is vendored unchanged in `rules/vendor/`). Every value at session t uses bars up
to t only: distribution days are counted from history[:t+1]; the follow-through state machine runs on the 60 sessions
ending at t (exactly the window the skill analyses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .vendor import tradermonty_rally_tracker as RT

DD_MIN_DECLINE = -0.002          # close at least 0.2% below the previous close
DD_EXPIRY = 25                   # removed when age > 25 sessions
DD_INVALIDATION_GAIN = 0.05      # removed once a post-DD high (within the window) is 5% above the DD close
RISK = {"NORMAL": 0, "CAUTION": 1, "HIGH": 2, "SEVERE": 3}
STATES = [s.value for s in RT.MarketState]


def distribution_days(d: pd.DataFrame) -> pd.DataFrame:
    """d: daily bars (date, high, close, volume). Per session: active distribution-day counts d5/d15/d25, the
    21-EMA/50-SMA filter and the risk class (0 NORMAL .. 3 SEVERE)."""
    d = d.sort_values("date").reset_index(drop=True)
    c, h, v = d.close.to_numpy(float), d.high.to_numpy(float), d.volume.to_numpy(float)
    chg = np.r_[np.nan, c[1:] / c[:-1] - 1]
    is_dd = (chg <= DD_MIN_DECLINE) & (np.r_[False, v[1:] > v[:-1]])
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().to_numpy()
    sma50 = pd.Series(c).rolling(50).mean().to_numpy()
    below = (c < ema21) | (c < np.nan_to_num(sma50, nan=-np.inf))
    dd_idx = np.flatnonzero(is_dd)
    rows = []
    for t in range(len(d)):
        cnt = {5: 0, 15: 0, 25: 0}
        for i in dd_idx[(dd_idx <= t) & (dd_idx >= t - DD_EXPIRY)]:
            age = t - i
            post = h[i + 1:t + 1]
            if len(post) and post.max() >= c[i] * (1 + DD_INVALIDATION_GAIN):
                continue
            for n in cnt:
                if age <= n:
                    cnt[n] += 1
        d5, d15, d25 = cnt[5], cnt[15], cnt[25]
        if d25 >= 6 or d15 >= 4 or (below[t] and d25 >= 5):
            risk = 3
        elif d25 >= 5 or d15 >= 3 or d5 >= 2:
            risk = 2
        elif d25 >= 3:
            risk = 1
        else:
            risk = 0
        rows.append((d5, d15, d25, float(below[t]), risk))
    return pd.DataFrame(rows, columns=["dd_d5", "dd_d15", "dd_d25", "below_21ema_or_50sma", "dd_risk"],
                        index=pd.DatetimeIndex(d.date, name="date"))


def _hist(d: pd.DataFrame) -> list[dict]:
    return [{"date": str(r.date.date()), "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "volume": float(r.volume)} for r in d.itertuples()]


def ftd_states(spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    """Combined S&P 500 / Nasdaq follow-through-day state per session (the skill's get_market_state on the 60 sessions
    ending at t), with each index's own state."""
    spy = spy.sort_values("date").reset_index(drop=True)
    qqq = qqq.set_index("date").reindex(spy.date).dropna(subset=["close"]).reset_index()
    hs, hq = _hist(spy), _hist(qqq)
    qpos = {r["date"]: i for i, r in enumerate(hq)}
    rows = []
    for t in range(len(hs)):
        ws = hs[max(0, t - 59):t + 1]
        j = qpos.get(hs[t]["date"])
        wq = hq[max(0, j - 59):j + 1] if j is not None else []
        st = RT.get_market_state(list(reversed(ws)), list(reversed(wq)))     # the skill expects newest first
        rows.append((st["combined_state"], st["sp500"]["state"], st["nasdaq"]["state"] if wq else None,
                     float(bool(st["dual_confirmation"]))))
    return pd.DataFrame(rows, columns=["ftd_state", "ftd_state_spy", "ftd_state_qqq", "ftd_dual"],
                        index=pd.DatetimeIndex(spy.date, name="date"))


def power_trend(d: pd.DataFrame) -> pd.Series:
    d = d.sort_values("date")
    c = d.close.astype(float).reset_index(drop=True)
    ema21 = c.ewm(span=21, adjust=False).mean()
    sma50 = c.rolling(50).mean()
    pt = (ema21 > sma50) & (sma50 > sma50.shift(5)) & (c > ema21)
    return pd.Series(pt.astype(float).to_numpy(), index=pd.DatetimeIndex(d.date, name="date"), name="power_trend")


def regime_features(spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    """The E20 rule block R (date-level): distribution days for SPY and QQQ, follow-through states (one-hot), power
    trend for SPY and QQQ."""
    ds = distribution_days(spy).add_prefix("rr_spy_")
    dq = distribution_days(qqq).add_prefix("rr_qqq_")
    fs = ftd_states(spy, qqq)
    oh = pd.get_dummies(fs.ftd_state).reindex(columns=STATES, fill_value=0).astype(float).add_prefix("rr_ftd_")
    out = ds.join(dq, how="left").join(oh, how="left")
    out["rr_ftd_dual"] = fs.ftd_dual
    out["rr_spy_power_trend"] = power_trend(spy)
    out["rr_qqq_power_trend"] = power_trend(qqq).reindex(out.index)
    return out

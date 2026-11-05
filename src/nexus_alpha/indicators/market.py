"""Causal market-context features from the cross-section of the 979-stock universe (equal-weighted).

The universe holds survivors only, so this proxy over-represents firms that did well; it is an ablation feature group,
not a benchmark (the economic benchmark is buy-and-hold of the same universe, see evaluation). Values at date t use
closes up to t only. A naive version of market features hurt TriFrame's volatility R2 (cqo_review 08), so the group is
tested by ablation rather than assumed useful.
"""
from __future__ import annotations

import concurrent.futures as cf

import numpy as np
import pandas as pd

from ..data import bars as B

MARKET_FEATURES = ["mkt_ret1", "mkt_ret5", "mkt_ret21", "mkt_rv20", "mkt_breadth_sma50", "mkt_dispersion20", "mkt_n"]


def _one(t: str) -> pd.DataFrame:
    d = B.read_daily(t)[["date", "close"]]
    c = d.close.astype(float)
    return pd.DataFrame({"date": d.date, "ticker": t, "r1": c.pct_change().values,
                         "above50": (c > c.rolling(50).mean()).astype(float).where(c.rolling(50).count() == 50).values})


def market_context(tickers: list[str] | None = None, workers: int = 6) -> pd.DataFrame:
    tickers = tickers or B.available_tickers()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        panel = pd.concat(list(ex.map(_one, tickers)), ignore_index=True)
    r = panel.pivot(index="date", columns="ticker", values="r1").sort_index()
    a = panel.pivot(index="date", columns="ticker", values="above50").reindex(r.index)
    ew = r.mean(axis=1)
    lvl = (1 + ew.fillna(0)).cumprod()
    out = pd.DataFrame(index=r.index)
    out["mkt_ret1"] = ew
    out["mkt_ret5"] = lvl / lvl.shift(5) - 1
    out["mkt_ret21"] = lvl / lvl.shift(21) - 1
    out["mkt_rv20"] = ew.rolling(20).std(ddof=0) * np.sqrt(252)
    out["mkt_breadth_sma50"] = a.mean(axis=1)
    out["mkt_dispersion20"] = r.std(axis=1).rolling(20).mean()
    out["mkt_n"] = r.notna().sum(axis=1)
    out.index.name = "date"
    return out

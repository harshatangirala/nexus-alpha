"""Select, for every sample, the news articles that were available at its decision time (vectorised as-of join).

Window: articles of the sample's ticker with decision_utc - LOOKBACK < available_utc <= decision_utc. Development use is
sealed: articles available on/after the seal are dropped before any join.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..seal import SEAL_UTC

LOOKBACK = pd.Timedelta(days=5)
MAX_ARTICLES = 7                         # the app accepts 1-7 articles; history uses the 7 most recent, after de-dup


def load_index(dev_only: bool = True) -> pd.DataFrame:
    idx = pd.read_parquet(paths.ARTIFACTS / "news" / "news_index.parquet")
    idx["available_utc"] = pd.to_datetime(idx.available_utc, utc=True)
    if dev_only:
        idx = idx[idx.available_utc < SEAL_UTC]
    else:
        from .. import seal as _seal
        _seal.require_open("news index beyond the seal")
    return idx.reset_index(drop=True)


def select_for_samples(samples: pd.DataFrame, idx: pd.DataFrame, lookback: pd.Timedelta = LOOKBACK) -> pd.DataFrame:
    """Return (sample_id, article_key, ...) pairs for articles inside each sample's window."""
    out = []
    s = samples[["sample_id", "ticker", "decision_utc"]].copy()
    s["decision_utc"] = pd.to_datetime(s.decision_utc, utc=True)
    a = idx[idx.ticker.isin(set(s.ticker))]
    for t, g in s.groupby("ticker"):
        ga = a[a.ticker == t].sort_values("available_utc")
        if ga.empty:
            continue
        at = ga.available_utc.to_numpy(dtype="datetime64[ns]")
        dt = g.decision_utc.to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(at, dt, side="right")
        lo = np.searchsorted(at, dt - np.timedelta64(int(lookback.total_seconds()), "s"), side="right")
        for sid, l_, h_ in zip(g.sample_id.to_numpy(), lo, hi):
            if h_ > l_:
                part = ga.iloc[l_:h_].copy()
                part.insert(0, "sample_id", sid)
                out.append(part)
    if not out:
        return pd.DataFrame(columns=["sample_id"] + list(idx.columns))
    res = pd.concat(out, ignore_index=True)
    return res

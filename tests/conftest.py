from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from nexus_alpha import paths
from nexus_alpha.data.calendar import sessions


def have_source_data() -> bool:
    return (paths.OHLCV_FINAL / "1day" / "AAPL.parquet").exists()


source_data = pytest.mark.skipif(not have_source_data(), reason="TriFrame source data store not available")


def synthetic_daily(n: int = 400, start: str = "2010-01-04", seed: int = 0, vol: float = 0.02) -> pd.DataFrame:
    """Geometric random walk on real XNYS sessions with consistent OHLC."""
    rng = np.random.default_rng(seed)
    s = sessions()
    s = s[s >= pd.Timestamp(start)][:n]
    r = rng.normal(0.0003, vol, len(s))
    c = 50 * np.exp(np.cumsum(r))
    o = c * np.exp(rng.normal(0, vol / 3, len(s)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, len(s))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, len(s))))
    v = rng.lognormal(13, 0.4, len(s))
    return pd.DataFrame({"date": s, "open": o, "high": h, "low": l, "close": c, "volume": v})


def synthetic_hourly(daily: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Seven regular-session bars per day (09:30 ... 15:30 New York) whose aggregate equals the daily bar."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, d in daily.iterrows():
        path = np.r_[d.open, d.open + np.cumsum(rng.normal(0, 1, 6)) * (d.high - d.low) / 10, d.close]
        path = np.clip(path, d.low, d.high)
        for j in range(7):
            ts = (d.date + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=j)).tz_localize("America/New_York")
            o_, c_ = path[j], path[j + 1]
            rows.append((ts.tz_convert("UTC"), o_, max(o_, c_), min(o_, c_), c_, d.volume / 7))
    h = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return h


@pytest.fixture
def daily_df():
    return synthetic_daily()


@pytest.fixture
def tmp_governance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GOVERNANCE", tmp_path / "gov")
    for k in ("NEXUS_HOLDOUT_CONFIG", "NEXUS_HOLDOUT_UNLOCK", "NEXUS_HOLDOUT_REUSE_REASON"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path

import numpy as np
import pandas as pd
import pytest

from nexus_alpha import trade_desk as TD


def test_horizon_mapping_v11():
    c = TD.choose_horizon("Intraday")
    assert c.served_by == "1D" and c.exact
    c = TD.choose_horizon("Swing", swing_days=5)
    assert c.served_by == "5D" and c.exact
    c = TD.choose_horizon("Swing", swing_days=4)
    assert c.served_by == "5D" and not c.exact and "approximation" in c.note
    c = TD.choose_horizon("Swing", swing_days=15)
    assert c.served_by in ("10D", "20D") and not c.exact
    c = TD.choose_horizon("Long-term", long_term="4 weeks")
    assert c.served_by == "20D" and c.exact
    for lt in ("3 months", "6 months", "1 year"):
        assert TD.choose_horizon("Long-term", long_term=lt).served_by is None
    with pytest.raises(ValueError):
        TD.choose_horizon("Swing", swing_days=1)


def test_legend_check():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 50)))
    u = rng.uniform(0.005, 0.03, (50, 3))
    d = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=50), "open": c * (1 - u[:, 0] + 0.01),
                      "high": c * (1 + u[:, 1]), "low": c * (1 - u[:, 2]), "close": c})
    d["high"] = d[["open", "high", "close"]].max(axis=1)
    d["low"] = d[["open", "low", "close"]].min(axis=1)
    bar = d.iloc[40]
    leg = {k: float(bar[k]) for k in ("open", "high", "low", "close")}
    r = TD.check_legend(d, leg)
    assert r["verdict"] == "match" and r["bars_from_end"] == 9
    r = TD.check_legend(d, {k: v * 0.9 for k, v in leg.items()})
    assert r["verdict"] == "basis_mismatch" and r["bars_from_end"] == 9
    assert TD.check_legend(d, {"close": 141})["verdict"] == "insufficient"

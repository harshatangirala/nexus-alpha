"""SPC rules of the forward-test monitor (Six Sigma control)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("spc", Path(__file__).resolve().parents[1] / "scripts" / "shadow" / "spc_monitor.py")
SPC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SPC)


def test_p_chart_flags_a_bad_session():
    c = pd.DataFrame({"asof_session": [f"d{i}" for i in range(20)], "failed": [2] * 19 + [60]})
    r = SPC.p_chart(c, 500)
    assert r["out_of_control"] == ["d19"]


def test_xmr_rules():
    x = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 30), 9.0], index=range(31))
    assert 30 in [int(i) for i in SPC.xmr(x)["rule1_beyond_3sigma"]]
    y = pd.Series(np.r_[np.zeros(10) - 1, np.ones(12)] + np.random.default_rng(1).normal(0, 0.01, 22))
    assert SPC.xmr(y)["rule2_nine_same_side"]
    assert "not yet defined" in SPC.xmr(pd.Series([1.0, 2.0]))["status"]

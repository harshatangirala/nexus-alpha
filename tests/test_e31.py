"""E31: the forward comparison of v1.2 against v1.1 (matched on-time rows, log loss difference per session)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(REPO / "scripts" / "shadow"))
    spec = importlib.util.spec_from_file_location("e31_score", REPO / "scripts" / "shadow" / "e31_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(p_up, late=False, horizon="5D", ticker="A", y=2):
    return {"asof_session": "2026-12-01", "ticker": ticker, "horizon": horizon, "p_down": (1 - p_up) / 2,
            "p_side": (1 - p_up) / 2, "p_up": p_up, "real_y": y, "late": late}


def test_gain_is_old_minus_new_on_matched_on_time_rows():
    E = _mod()
    old = pd.DataFrame([_rows(0.3), _rows(0.3, ticker="B"), _rows(0.3, horizon="3M", ticker="C")])
    new = pd.DataFrame([_rows(0.5), _rows(0.5, ticker="B", late=True), _rows(0.9, horizon="3M", ticker="C")])
    g = E.session_gains(old, new)
    assert len(g) == 1 and g.rows.iloc[0] == 1                      # B is late in v1.2; 3M is outside 1D-1M
    assert np.isclose(g.gain.iloc[0], -np.log(0.3) + np.log(0.5))


def test_empty_when_nothing_matches():
    E = _mod()
    old = pd.DataFrame([_rows(0.3, ticker="A")])
    new = pd.DataFrame([_rows(0.4, ticker="Z")])
    assert E.session_gains(old, new).empty

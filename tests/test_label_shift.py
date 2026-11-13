"""E29a prior-shift recalibration (D-058): exact algebra, point in time, no correction without enough matured rows."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nexus_alpha.calibration import label_shift as LS  # noqa: E402


def test_prior_shift_algebra():
    p = np.array([[0.2, 0.5, 0.3]])
    q = LS.prior_shift(p, np.array([0.3, 0.4, 0.3]), np.array([0.3, 0.4, 0.3]))
    assert np.allclose(q, p)                                            # same priors: unchanged
    q = LS.prior_shift(p, np.array([1 / 3] * 3), np.array([0.2, 0.6, 0.2]))
    assert np.isclose(q.sum(), 1) and q[0, 1] > p[0, 1] and q[0, 0] < p[0, 0]
    q = LS.prior_shift(p, np.array([0.01, 0.98, 0.01]), np.array([0.98, 0.01, 0.01]))
    w = q[0] / p[0]; assert w.max() / w.min() <= 16 + 1e-9               # ratios clipped to [0.25, 4]


def _scored(n_days=8, per_day=400, seed=0):
    rng = np.random.default_rng(seed); rows = []
    days = pd.bdate_range("2026-01-05", periods=n_days)
    for i, d in enumerate(days):
        dec = (d + pd.offsets.BDay(1)).strftime("%Y-%m-%d") + "T14:00:00+00:00"
        y = rng.choice(3, per_day, p=[0.2, 0.6, 0.2])
        rows.append(pd.DataFrame({"asof_session": d.strftime("%Y-%m-%d"), "ticker": [f"T{k}" for k in range(per_day)],
                                  "horizon": "1D", "p_down": 0.3, "p_side": 0.4, "p_up": 0.3, "real_y": y,
                                  "real_end_session": (d + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
                                  "decision_utc": dec, "late": False}))
    return pd.concat(rows, ignore_index=True)


def test_point_in_time_and_min_rows(monkeypatch):
    s = _scored()
    monkeypatch.setattr(LS, "MIN_ROWS", 1000)
    c = LS.corrected(s, s)
    first = c[c.asof_session == c.asof_session.min()]
    assert (first.n_prior_rows == 0).all()                              # nothing known before the first decision
    # a row is never corrected with its own outcome: the rows used are those known before its decision time
    for dec, g in c.groupby("decision_utc"):
        known = LS.label_known_utc(s.real_end_session) < pd.Timestamp(dec)
        n_known = int(known.sum())
        assert (g.n_prior_rows == (n_known if n_known >= 1000 else 0)).all()
    late = c[c.n_prior_rows > 0]
    assert len(late) and (late.p_side_ls > late.p_side).all()          # sideways under-predicted (0.4 vs 0.6): raised


def test_scorer_reports_progress_only_before_120_sessions(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("e29_score", REPO / "scripts" / "shadow" / "e29_score.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    monkeypatch.setattr(LS, "MIN_ROWS", 1000)
    g = M.session_gains(_scored())
    assert len(g) > 0 and (g.rows > 0).all() and g.gain.mean() > 0      # the shift helps when priors really moved

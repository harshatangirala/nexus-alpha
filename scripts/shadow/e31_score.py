"""E31 forward score (D-062): does v1.2 (v1.1's recipe refit to 2021-11-30, calibrated 2022) beat v1.1 live?

    python scripts/shadow/e31_score.py        # writes artifacts/e31/e31_status.json and prints progress

Reads the scored, on-time rows of both models (shadow/v1.1/scores.csv and shadow/v1.2/scores.csv, written by
shadow_score.py). Endpoint (pre-registered, docs/E31_PREREGISTRATION.md): per decision session, the mean over matched
stock-horizon rows (1D-1M) of log loss(v1.1) - log loss(v1.2); one-sided moving-block bootstrap over sessions (blocks of
20); decided at 120 sessions. Before that it reports progress only. Nothing is fitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.evaluation import forward_criteria as FC  # noqa: E402

MIN_SESSIONS = 120
HORIZONS = ["1D", "2D", "5D", "10D", "20D", "1M"]
OUT = REPO / "artifacts" / "e31" / "e31_status.json"
KEY = ["asof_session", "ticker", "horizon"]


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))


def session_gains(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per session: mean log loss(v1.1) - log loss(v1.2) over matched on-time, matured rows (positive = v1.2 better)."""
    def ok(d):
        return d[d.real_y.notna() & ~d.late.astype(bool) & d.horizon.isin(HORIZONS)]
    m = ok(old)[KEY + ["p_down", "p_side", "p_up", "real_y"]].merge(
        ok(new)[KEY + ["p_down", "p_side", "p_up", "real_y"]], on=KEY, suffixes=("", "_new"))
    if m.empty:
        return pd.DataFrame(columns=["asof_session", "rows", "gain"])
    assert (m.real_y == m.real_y_new).all(), "the two scorers disagree on an outcome"
    y = m.real_y.astype(int).to_numpy()
    m["gain"] = (_ll(m[["p_down", "p_side", "p_up"]].to_numpy(float), y)
                 - _ll(m[["p_down_new", "p_side_new", "p_up_new"]].to_numpy(float), y))
    return m.groupby("asof_session").agg(rows=("gain", "size"), gain=("gain", "mean")).reset_index()


def main() -> None:
    read = lambda v: pd.read_csv(SH.SHADOW_ROOT / v / "scores.csv") if (SH.SHADOW_ROOT / v / "scores.csv").exists() else pd.DataFrame()
    old, new = read("v1.1"), read("v1.2")
    g = session_gains(old, new) if len(old) and len(new) else pd.DataFrame(columns=["asof_session", "rows", "gain"])
    n = len(g)
    res = {"test": "E31 v1.2 (recipe refit to 2021, calibrated 2022) vs v1.1, forward log loss 1D-1M",
           "sessions": n, "min_sessions": MIN_SESSIONS, "rows": int(g.rows.sum()) if n else 0,
           "mean_gain_per_session": float(g.gain.mean()) if n else None}
    if n >= MIN_SESSIONS:
        p = FC.block_p_value(g.set_index("asof_session").gain)
        res.update({"p_one_sided": p, "decision": "REFIT HELPS" if p < 0.05 and g.gain.mean() > 0 else "NOT SHOWN"})
    else:
        res["decision"] = f"progress only: {n} of {MIN_SESSIONS} sessions"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + chr(10), encoding="utf-8")
    print(f"E31: {res['decision']}; matched rows {res['rows']}", flush=True)


if __name__ == "__main__":
    main()

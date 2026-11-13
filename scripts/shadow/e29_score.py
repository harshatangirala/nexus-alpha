"""E29a forward score: does prior-shift recalibration improve the frozen v1.1's probabilities? (D-058)

    python scripts/shadow/e29_score.py          # writes artifacts/e29/e29_status.json and prints progress

Reads only the forward test's scored rows (shadow/v1.1/scores.csv: on-time rows whose outcome has matured). For each
row, `calibration.label_shift.corrected` reweights v1.1's probabilities with class shares of rows whose labels were
known before that row's decision time, so the corrected forecast never sees its own or a later outcome.

Endpoint (pre-registered, docs/E29_PREREGISTRATION.md): per decision session, the mean over horizons and stocks of
log loss(raw v1.1) - log loss(E29a), on rows the correction touched; one-sided moving-block bootstrap over sessions
(blocks of 20); decided once there are at least 120 such sessions, Holm across E29's arms. Before that it reports
progress only. Nothing is fitted; nothing here changes a logged forecast.
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
from nexus_alpha.calibration import label_shift as LS  # noqa: E402
from nexus_alpha.evaluation import forward_criteria as FC  # noqa: E402

MIN_SESSIONS = 120
OUT = REPO / "artifacts" / "e29" / "e29_status.json"


def session_gains(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per decision session: mean log-loss gain of E29a over raw v1.1 on the corrected rows."""
    s = scored[scored.real_y.notna() & ~scored.late.astype(bool)].copy()
    c = LS.corrected(s, s)
    c = c[c.n_prior_rows > 0]
    if c.empty:
        return pd.DataFrame(columns=["asof_session", "rows", "gain"])
    y = c.real_y.astype(int).to_numpy()
    raw = c[["p_down", "p_side", "p_up"]].to_numpy(float)
    cor = c[["p_down_ls", "p_side_ls", "p_up_ls"]].to_numpy(float)
    r_ = np.arange(len(c))
    c["gain"] = -np.log(np.clip(raw[r_, y], 1e-7, 1)) + np.log(np.clip(cor[r_, y], 1e-7, 1))
    return c.groupby("asof_session").agg(rows=("gain", "size"), gain=("gain", "mean")).reset_index()


def main() -> None:
    f = SH.SHADOW_ROOT / "v1.1" / "scores.csv"
    scored = pd.read_csv(f) if f.exists() else pd.DataFrame()
    g = session_gains(scored) if len(scored) else pd.DataFrame(columns=["asof_session", "rows", "gain"])
    n = len(g)
    res = {"arm": "E29a prior-shift recalibration of v1.1", "sessions_with_correction": n, "min_sessions": MIN_SESSIONS,
           "rows": int(g.rows.sum()) if n else 0, "mean_gain_per_row": float(g.gain.mean()) if n else None}
    if n >= MIN_SESSIONS:
        p = FC.block_p_value(g.set_index("asof_session").gain)
        res.update({"p_one_sided": p, "decision": "pending Holm with E29b (docs/E29_PREREGISTRATION.md)"})
    else:
        res["decision"] = f"progress only: {n} of {MIN_SESSIONS} sessions"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"E29a: {res['decision']}; rows corrected {res['rows']}", flush=True)


if __name__ == "__main__":
    main()

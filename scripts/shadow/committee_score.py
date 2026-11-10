"""E27 forward scoring of the LLM committee (TradingAgents; pre-registered in docs/V3_RESEARCH.md).

    python scripts/shadow/committee_score.py

For every public committee record (shadow/committee/runs/*.json) that is on time and has a rating: realized labels at
5D and 20D with the training label rule on Yahoo bars, once each horizon has matured. Ratings map Buy/Overweight = up,
Hold = sideways, Underweight/Sell = down; REVIEW counts as a failure. One vote per stock per 5 sessions: the first.
Reports the up-vs-down hit rate on rows whose outcome is not sideways, the agreement with v1.1's lean, and the share of
Hold calls against the share of sideways outcomes. No claim before 100 scored runs. Writes shadow/committee/summary.json.
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
sys.path.insert(0, str(REPO / "scripts" / "shadow"))

from nexus_alpha import shadow as SH  # noqa: E402

HORIZONS = {"5D": 5, "20D": 20}
MIN_CLAIM = 100


def one_vote_per_window(runs: pd.DataFrame, gap: int = 5) -> pd.DataFrame:
    """Keep the first run per stock, then the next run of that stock at least `gap` sessions (weekdays) later."""
    keep = []
    for _, g in runs.sort_values("asof_session").groupby("ticker"):
        last = None
        for i, r in g.iterrows():
            d = pd.Timestamp(r.asof_session)
            if last is None or np.busday_count(last.date(), d.date()) >= gap:
                keep.append(i)
                last = d
    return runs.loc[keep]


def summarize(rows: pd.DataFrame) -> dict:
    out = {"scored": int(len(rows))}
    if not len(rows):
        return out
    mv = rows[rows.real_y != 1]
    called = mv[mv.direction_class != 1]
    out["hit_rate_up_vs_down"] = float((called.direction_class == called.real_y).mean()) if len(called) else None
    out["n_directional_calls_on_moves"] = int(len(called))
    out["hold_share"] = float((rows.direction_class == 1).mean())
    out["sideways_share"] = float((rows.real_y == 1).mean())
    if "v11_lean" in rows and rows.v11_lean.notna().any():
        r = rows.dropna(subset=["v11_lean"])
        out["agree_with_v11_lean"] = float(((r.direction_class == 2) == (r.v11_lean > 0)).mean())
    if len(called) >= 20:
        g = called.assign(hit=(called.direction_class == called.real_y).astype(float)).groupby("asof_session").hit
        s, c = g.sum().to_numpy(), g.count().to_numpy()
        rng = np.random.default_rng(11)
        draws = [s[ix].sum() / c[ix].sum() for ix in (rng.integers(0, len(s), len(s)) for _ in range(1000))]
        out["hit_rate_ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    out["claim"] = ("none yet: fewer than 100 scored runs" if len(rows) < MIN_CLAIM else
                    "eligible (runs are chosen by the user, so the sample is not random)")
    return out


def main() -> None:
    import shadow_score as SS
    probs = SH.verify_chain("committee")
    pub = sorted((SH.SHADOW_ROOT / "committee" / "runs").glob("*.json"))
    runs = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in pub])
    res = {"chain_intact": not probs, "runs_logged": int(len(runs)), "results": {}}
    if len(runs):
        res["late_runs"] = int(runs.late.sum())
        res["review_runs"] = int(runs.direction_class.isna().sum())
        ok = one_vote_per_window(runs[~runs.late.astype(bool) & runs.direction_class.notna()])
        rows = {h: [] for h in HORIZONS}
        for r in ok.itertuples():
            try:
                d = SS.yahoo_daily(r.ticker, (pd.Timestamp(r.asof_session) - pd.Timedelta(days=120)).strftime("%Y-%m-%d"))
            except Exception:  # noqa: BLE001  (scored on a later night)
                continue
            lab = SH.realized_labels(d, pd.Timestamp(r.asof_session), HORIZONS)
            lean = None
            f = SH.SHADOW_ROOT / "v1.1" / "forecasts" / f"{r.asof_session}.csv"
            if f.exists():
                v = pd.read_csv(f)
                v = v[(v.ticker == r.ticker) & (v.horizon == "5D")]
                lean = float(v.p_up.iloc[0] - v.p_down.iloc[0]) if len(v) else None
            for h in HORIZONS:
                if h in lab:
                    rows[h].append({"asof_session": r.asof_session, "ticker": r.ticker,
                                    "direction_class": int(r.direction_class), "real_y": lab[h]["y"], "v11_lean": lean})
        for h, rr in rows.items():
            res["results"][h] = summarize(pd.DataFrame(rr))
    dst = SH.SHADOW_ROOT / "committee" / "summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    scored = "; ".join(f"{h}: {v.get('scored', 0)} scored" for h, v in res["results"].items()) or "nothing to score"
    print(f"committee: {res['runs_logged']} runs logged; {scored}; chain {'intact' if not probs else probs}")


if __name__ == "__main__":
    main()

"""Statistical process control for the forward shadow test (Six Sigma "Control"; audit 2026-11-06).

    python scripts/shadow/spc_monitor.py

Per frozen version, from the hash-chained logs:
* p-chart of the share of universe tickers that failed to forecast per session (3-sigma limits on the pooled rate);
* late rows (forecast after the decision time) must be 0 (any is a defect);
* once >= 20 sessions have matured, an individuals (X-mR) chart of the daily log-loss gain vs the training prior per
  horizon, with Western Electric rule 1 (a point beyond 3 sigma) and rule 2 (9 consecutive points on one side of the
  centre line).
Writes shadow/spc_report.json and prints alerts (exit 0 always; alerts are information for the owner).
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

UNIVERSE = {"v1.1": 500, "v2": 500, "market_e19": 1, "v3_fund": 500}


def p_chart(counts: pd.DataFrame, n: int) -> dict:
    """counts: per session failed tickers. Pooled p and 3-sigma upper limit; sessions above it are out of control."""
    p = counts.failed.sum() / (n * len(counts))
    ucl = p + 3 * np.sqrt(max(p * (1 - p), 1e-12) / n)
    out = counts[counts.failed / n > ucl]
    return {"sessions": int(len(counts)), "p_bar": float(p), "ucl": float(ucl), "dpmo": float(p * 1e6),
            "out_of_control": out.asof_session.tolist()}


def xmr(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 20:
        return {"n": int(len(x)), "status": "fewer than 20 matured sessions: chart not yet defined"}
    c = float(x.mean())
    mr = float(x.diff().abs().mean())
    sig = mr / 1.128
    rule1 = x.index[(x - c).abs() > 3 * sig].astype(str).tolist()
    side = np.sign(x - c)
    run, rule2 = 0, []
    for i in range(1, len(side)):
        run = run + 1 if side.iloc[i] == side.iloc[i - 1] and side.iloc[i] != 0 else 1
        if run >= 9:
            rule2.append(str(x.index[i]))
    return {"n": int(len(x)), "centre": c, "sigma": sig, "ucl": c + 3 * sig, "lcl": c - 3 * sig,
            "rule1_beyond_3sigma": rule1, "rule2_nine_same_side": rule2}


def main() -> None:
    report, alerts = {}, []
    for ver, n in UNIVERSE.items():
        root = SH.SHADOW_ROOT / ver
        if not (root / "chain.csv").exists():
            continue
        probs = SH.verify_chain(ver)
        ch = pd.read_csv(root / "chain.csv")
        per = []
        for asof, g in ch.groupby("asof_session"):
            files = [root / f for f in g.file]
            logged = pd.concat([pd.read_csv(f, usecols=["ticker"]) for f in files]).ticker.nunique()
            per.append({"asof_session": asof, "logged": int(logged), "failed": int(max(n - logged, 0)),
                        "late_rows": int(g.late_rows.sum())})
        per = pd.DataFrame(per)
        r = {"chain_intact": not probs, "coverage": p_chart(per, n), "late_rows_total": int(per.late_rows.sum())}
        if probs:
            alerts.append(f"{ver}: hash chain problem {probs}")
        if r["late_rows_total"]:
            alerts.append(f"{ver}: {r['late_rows_total']} late rows (forecasts after decision time)")
        if r["coverage"]["out_of_control"]:
            alerts.append(f"{ver}: coverage out of control on {r['coverage']['out_of_control']}")
        sc = root / "scores.csv"
        if sc.exists():
            s = pd.read_csv(sc)
            prior = json.loads((SH.FROZEN_DIR / f"{ver}.json").read_text(encoding="utf-8"))["train_class_prior"]
            charts = {}
            for h, g in s.groupby("horizon"):
                P = g[["p_down", "p_side", "p_up"]].to_numpy()
                y = g.real_y.astype(int).to_numpy()
                gain = -np.log(np.asarray(prior[h])[y]) + np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
                daily = pd.Series(gain, index=g.asof_session.values).groupby(level=0).mean().sort_index()
                charts[h] = xmr(daily)
                for rule in ("rule1_beyond_3sigma", "rule2_nine_same_side"):
                    if charts[h].get(rule):
                        alerts.append(f"{ver} {h}: {rule} at {charts[h][rule][-3:]}")
            r["log_loss_gain_xmr"] = charts
        report[ver] = r
    report["alerts"] = alerts
    (SH.SHADOW_ROOT / "spc_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("SPC:", "no alerts" if not alerts else "; ".join(alerts))
    for ver, r in report.items():
        if ver != "alerts":
            c = r["coverage"]
            print(f"  {ver}: sessions {c['sessions']}, failure rate {c['p_bar']:.4f} (DPMO {c['dpmo']:.0f}), "
                  f"late rows {r['late_rows_total']}, chain {'intact' if r['chain_intact'] else 'BROKEN'}")


if __name__ == "__main__":
    main()

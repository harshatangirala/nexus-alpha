"""E04: economic evaluation of pre-specified policies (docs/PROJECT_SPECIFICATION.md section 7).

Policies (horizons 5D and 20D):
  P0 rule_only      weekly permission + daily setup + same-side hourly trigger; structural stop, first-obstacle target
  P1 model_atr      |P(UP) - P(DOWN)| >= theta from the selected calibrated E03 model; stop 2 ATR, target 3 ATR
  P2 nexus          P1 signal AND a same-side rule setup and trigger; structural levels (the planner with its skill gate
                    bypassed, to measure what the gate is protecting against)
  P2g nexus_gated   the live planner as is (requires validated directional skill per horizon)
theta in {0.05, 0.10, 0.15, 0.20} chosen on the VALID block by net Sharpe and frozen before the test block is simulated.
Benchmark: equal-weight buy-and-hold of the survivor universe (survivorship-biased). Costs base/stressed/severe.
Deflated Sharpe uses the number of configurations tried; PBO by CSCV over all configurations on valid+calib+test.
Run: .venv/Scripts/python -I scripts/experiments/e04_backtest.py <e03_run_id>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.data import bars as B
from nexus_alpha.data.calendar import sessions
from nexus_alpha.evaluation.stats import pbo_cscv
from nexus_alpha.experiments import Run
from nexus_alpha.indicators.market import market_context
from nexus_alpha.targets import HORIZONS
from nexus_alpha.trading.backtest import Order, performance, portfolio, simulate
from nexus_alpha.trading.costs import CostModel
from nexus_alpha.trading.plan import RiskConfig, plan_trade

THETAS = (0.05, 0.10, 0.15, 0.20)
HZ = ("5D", "20D")
BLOCK_DATES = {"valid": ("2017-01-01", "2018-12-31"), "calib": ("2019-01-01", "2019-12-31"), "test": ("2020-01-01", "2022-12-30")}


def orders_for(ev: pd.DataFrame, probs: np.ndarray, policy: str, h: str, theta: float, skill: dict) -> list[Order]:
    j = list(HORIZONS).index(h)
    n = HORIZONS[h]
    risk = RiskConfig(min_edge=theta)
    out = []
    for i, r in enumerate(ev.itertuples(index=False)):
        p = probs[i, j]
        setups = json.loads(r.setups_json)
        trig = {"LONG": json.loads(r.trigger_long_json), "SHORT": json.loads(r.trigger_short_json)}
        weekly = {"regime": r.regime, "permission": r.permission}
        if policy == "rule_only":
            for s in setups:
                if trig[s["direction"]].get("trigger") != "NO_TRIGGER":
                    tp = plan_trade(h, n, None, True, True, True, weekly, [s], trig, r.atr14, r.close, risk,
                                    side_override=s["direction"])
                    if tp.decision in ("LONG", "SHORT"):
                        out.append(Order(r.sample_id, r.ticker, r.k, 1 if tp.decision == "LONG" else -1, tp.stop,
                                         tp.target_1, n, tp.max_chase_price, policy))
                    break
        elif policy == "model_atr":
            edge = p[2] - p[0]
            if abs(edge) >= theta:
                side = 1 if edge > 0 else -1
                out.append(Order(r.sample_id, r.ticker, r.k, side, r.close - side * 2 * r.atr14, r.close + side * 3 * r.atr14,
                                 n, r.close + side * r.atr14, policy))
        else:
            gated = policy == "nexus_gated"
            tp = plan_trade(h, n, {"up": p[2], "down": p[0], "side": p[1]}, True,
                            bool(skill.get(h, {}).get("validated")) if gated else True, True, weekly, setups, trig,
                            r.atr14, r.close, risk)
            if tp.decision in ("LONG", "SHORT"):
                out.append(Order(r.sample_id, r.ticker, r.k, 1 if tp.decision == "LONG" else -1, tp.stop, tp.target_1, n,
                                 tp.max_chase_price, policy))
    return out


def run_config(orders, daily, block, cost):
    sims = [simulate(o, daily[o.ticker], cost) for o in orders]
    tr = pd.DataFrame(sims) if sims else pd.DataFrame(columns=["fill_status"])
    s = sessions()
    a, b = BLOCK_DATES[block]
    cal = s[(s >= pd.Timestamp(a)) & (s <= pd.Timestamp(b))]
    if len(tr) and (tr.fill_status == "FILLED").any():
        ret, kept = portfolio(tr, cal)
    else:
        ret, kept = pd.Series(0.0, index=cal), tr
    return ret, tr, kept


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("e03")
    ap.add_argument("--probs", default=None, help="npy (N,6,3) aligned with the E03 rows; default: E03 selected calibrated")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--min-trades", type=int, default=0, help="v1.1 (audit A-06): a threshold is eligible only with >= this many validation trades")
    ap.add_argument("--bundle", default=None, help="bundle whose direction-skill flags drive the gated policy; must be the model "
                    "that produced --probs (audit: v1.1 used E03's flags, which agree with E08's at 5D and 20D)")
    a = ap.parse_args()
    e03 = a.e03
    pdir = paths.ARTIFACTS / "predictions" / e03
    rows = pd.read_parquet(pdir / "rows.parquet")
    probs = np.load(a.probs) if a.probs else np.load(pdir / "selected_calibrated__probs.npy")
    meta = json.loads((paths.ARTIFACTS / "bundles" / (a.bundle or e03) / "bundle_meta.json").read_text(encoding="utf-8"))
    skill = meta["direction_skill"]
    ev_all = pd.read_parquet(paths.ARTIFACTS / "features" / "rule_events.parquet")
    run = Run("e04_backtest_" + a.tag, {"e03_run": e03, "probs": a.probs, "skill_bundle": a.bundle or e03, "min_trades": a.min_trades, "thetas": THETAS, "horizons": HZ, "risk": RiskConfig().__dict__,
                               "policies": ["rule_only", "model_atr", "nexus", "nexus_gated"]})
    daily = {}
    res, daily_rets = {}, {}
    try:
        for block in ("valid", "calib", "test"):
            m = (rows.block == block).to_numpy()
            r = rows[m].reset_index(drop=True)
            ev = r[["sample_id"]].merge(ev_all, on="sample_id", how="left")
            for t in ev.ticker.unique():
                if t not in daily:
                    daily[t] = B.read_daily(t)
            pb = probs[m]
            for h in HZ:
                for pol in ("rule_only", "model_atr", "nexus", "nexus_gated"):
                    for th in (THETAS if pol in ("model_atr", "nexus") else (0.10,)):
                        key = f"{pol}|{h}|{th}"
                        ret, tr, kept = run_config(orders_for(ev, pb, pol, h, th, skill), daily, block, CostModel())
                        perf = performance(ret, kept if len(kept) else None)
                        perf["orders"] = int(len(tr))
                        perf["cancelled"] = int((tr.fill_status == "CANCELLED").sum()) if len(tr) else 0
                        res.setdefault(block, {})[key] = perf
                        daily_rets.setdefault(key, []).append(ret)
                        print(f"{block} {key:<26} trades {perf.get('trades', 0):>5} Sharpe {perf['sharpe']:.2f} "
                              f"CAGR {perf['cagr'] if perf['cagr'] == perf['cagr'] else float('nan'):.3f} MDD {perf['max_drawdown']:.3f}", flush=True)
        # theta frozen on valid, per policy and horizon
        frozen = {}
        for h in HZ:
            for pol in ("model_atr", "nexus"):
                cands = {th: res["valid"][f"{pol}|{h}|{th}"]["sharpe"] for th in THETAS
                         if res["valid"][f"{pol}|{h}|{th}"].get("trades", 0) >= a.min_trades}
                if not cands:                                    # nothing eligible: the most active threshold
                    cands = {max(THETAS, key=lambda th: res["valid"][f"{pol}|{h}|{th}"].get("trades", 0)): 0.0}
                cands = {k: (v if v == v else -99) for k, v in cands.items()}
                frozen[f"{pol}|{h}"] = max(cands, key=cands.get)
        # test with frozen thresholds, cost scenarios
        final = {}
        m = (rows.block == "test").to_numpy()
        r = rows[m].reset_index(drop=True)
        ev = r[["sample_id"]].merge(ev_all, on="sample_id", how="left")
        daily_final = {}
        all_sr = [v["sharpe"] for v in res["valid"].values() if v["sharpe"] == v["sharpe"]]
        n_trials = len(res["valid"])
        sr_var = float(np.var(np.array(all_sr) / np.sqrt(252), ddof=1)) if len(all_sr) > 2 else 0.0
        for h in HZ:
            for pol in ("rule_only", "model_atr", "nexus", "nexus_gated"):
                th = frozen.get(f"{pol}|{h}", 0.10)
                for sc in ("base", "stressed", "severe"):
                    ret, tr, kept = run_config(orders_for(ev, probs[m], pol, h, th, skill), daily, "test", CostModel.from_scenario(sc))
                    perf = performance(ret, kept if len(kept) else None, n_trials=n_trials, trial_sr_var=sr_var)
                    final[f"{pol}|{h}|theta={th}|{sc}"] = perf
                    if sc == "base":
                        daily_final[f"{pol}|{h}"] = ret
        s = sessions()
        mk = market_context()
        bench = {}
        for block, (d0, d1) in BLOCK_DATES.items():          # (not 'a': that name holds the parsed arguments)
            cal = s[(s >= pd.Timestamp(d0)) & (s <= pd.Timestamp(d1))]
            bench[block] = performance(mk.mkt_ret1.reindex(cal).fillna(0.0))
        cal_t = s[(s >= pd.Timestamp(BLOCK_DATES["test"][0])) & (s <= pd.Timestamp(BLOCK_DATES["test"][1]))]
        dfr = pd.DataFrame(daily_final)
        dfr["benchmark_ew_survivors"] = mk.mkt_ret1.reindex(cal_t).fillna(0.0).values
        rep0 = paths.REPO_ROOT / "reports" / "backtest"
        rep0.mkdir(parents=True, exist_ok=True)
        dfr.to_csv(rep0 / f"e04_daily_returns_{a.tag}.csv", index_label="date")
        perf_mat = pd.concat({k: pd.concat(v) for k, v in daily_rets.items()}, axis=1).fillna(0.0)
        pbo = pbo_cscv(perf_mat.to_numpy(), n_splits=10)
        out = {"by_block": res, "frozen_thresholds": frozen, "test_final": final, "benchmark_ew_survivors": bench,
               "pbo": pbo, "n_trials_valid": n_trials, "direction_skill_used_by_gated_policy": skill}
        rep = paths.REPO_ROOT / "reports" / "backtest"
        rep.mkdir(parents=True, exist_ok=True)
        (rep / ("e04_backtest.json" if a.tag == "v1" else f"e04_backtest_{a.tag}.json")).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        summ = "; ".join(f"{k.split('|theta')[0]} Sharpe {v['sharpe']:.2f}" for k, v in final.items() if k.endswith("|base"))
        summ += f"; EW benchmark test Sharpe {bench['test']['sharpe']:.2f}; PBO {pbo.get('pbo')}"
        run.finish("COMPLETED", summ, out)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

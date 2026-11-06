"""E15 v2 backtest (pre-registered in docs/V2_ARCHITECTURE.md section 4; threshold block stated in D-025).

The E04 v1.1 engine unchanged: orders from `orders_for` (rule-only, model + ATR stops, NEXUS planner, NEXUS gated by the
direction-skill flag), fills, costs, kill-switch and gross cap from `run_config`. Probabilities: the deployed v2 bundle's
calibrated predictions (refit on 2003-2018, so 2019 and 2020-2022 are out of sample for it).
One horizon per trade type: 1D (intraday), 5D and 10D (swing), 20D and 3M (long-term).
Thresholds (min edge) are frozen on 2019 with >= 30 trades (D-025: the tabular components hold no full-row
2017-2018 predictions), then 2020-2022 is run under base, stressed and severe costs. Benchmarks: equal-weight
survivors and SPY buy-and-hold (Yahoo adjusted close, i.e. with dividends). Deflated Sharpe uses the 2019 trial count.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

import e04_backtest as E04  # noqa: E402
from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.data.calendar import sessions  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.indicators.market import market_context  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402
from nexus_alpha.trading.backtest import performance  # noqa: E402
from nexus_alpha.trading.costs import CostModel  # noqa: E402

E04.HORIZONS = HORIZONS_V2          # orders_for looks horizons up in this module global
HZ = ("1D", "5D", "10D", "20D", "3M")
MIN_TRADES = 30


def main() -> None:
    runs = sorted((REPO / "experiments").glob("*_e14_blend_freeze_v2_*"))
    e14 = [r for r in runs if json.loads((r / "manifest.json").read_text()).get("status") == "COMPLETED"][-1].name
    pdir = paths.ARTIFACTS / "predictions" / e14
    rows = pd.read_parquet(pdir / "rows.parquet")
    probs = np.load(pdir / "probs_calibrated.npy")
    meta = json.loads((paths.ARTIFACTS / "bundles" / e14 / "bundle_meta.json").read_text(encoding="utf-8"))
    skill = meta["direction_skill"]
    ev_all = pd.read_parquet(paths.ARTIFACTS / "features" / "rule_events.parquet")
    run = Run("e15_backtest_v2", {"bundle": e14, "horizons": HZ, "thetas": E04.THETAS, "freeze_block": "calib",
                                  "min_trades": MIN_TRADES})
    daily, res, final = {}, {}, {}
    for block in ("calib", "test"):
        m = (rows.block == block).to_numpy()
        r = rows[m].reset_index(drop=True)
        ev = r[["sample_id"]].merge(ev_all, on="sample_id", how="left")
        for t in ev.ticker.dropna().unique():
            if t not in daily:
                daily[t] = B.read_daily(t)
        pb = probs[m]
        if block == "calib":
            for h in HZ:
                for pol in ("rule_only", "model_atr", "nexus", "nexus_gated"):
                    for th in (E04.THETAS if pol in ("model_atr", "nexus") else (0.10,)):
                        ret, tr, kept = E04.run_config(E04.orders_for(ev, pb, pol, h, th, skill), daily, block, CostModel())
                        perf = performance(ret, kept if len(kept) else None)
                        res[f"{pol}|{h}|{th}"] = perf
            frozen = {}
            for h in HZ:
                for pol in ("model_atr", "nexus"):
                    c = {th: res[f"{pol}|{h}|{th}"]["sharpe"] for th in E04.THETAS
                         if res[f"{pol}|{h}|{th}"].get("trades", 0) >= MIN_TRADES}
                    if not c:
                        c = {max(E04.THETAS, key=lambda th: res[f"{pol}|{h}|{th}"].get("trades", 0)): 0.0}
                    frozen[f"{pol}|{h}"] = max(c, key=lambda k: c[k] if c[k] == c[k] else -99)
            srs = [v["sharpe"] for v in res.values() if v["sharpe"] == v["sharpe"]]
            n_trials, sr_var = len(res), (float(np.var(np.array(srs) / np.sqrt(252), ddof=1)) if len(srs) > 2 else 0.0)
        else:
            daily_rets = {}
            for h in HZ:
                for pol in ("rule_only", "model_atr", "nexus", "nexus_gated"):
                    th = frozen.get(f"{pol}|{h}", 0.10)
                    for sc in ("base", "stressed", "severe"):
                        ret, tr, kept = E04.run_config(E04.orders_for(ev, pb, pol, h, th, skill), daily, "test",
                                                       CostModel.from_scenario(sc))
                        final[f"{pol}|{h}|theta={th}|{sc}"] = performance(ret, kept if len(kept) else None,
                                                                          n_trials=n_trials, trial_sr_var=sr_var)
                        if sc == "base":
                            daily_rets[f"{pol}|{h}"] = ret
    s = sessions()
    cal_t = s[(s >= pd.Timestamp("2020-01-01")) & (s <= pd.Timestamp("2022-12-30"))]
    mk = market_context()
    bench = {"ew_survivors": performance(mk.mkt_ret1.reindex(cal_t).fillna(0.0))}
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").history(start="2019-12-01", end="2023-01-01", auto_adjust=True)["Close"]
        spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
        bench["spy_total_return"] = performance(spy.pct_change().reindex(cal_t).fillna(0.0))
    except Exception as e:
        bench["spy_total_return"] = {"error": repr(e)[:200]}
    out = {"bundle": e14, "calib_2019": res, "frozen_thresholds": frozen, "test_final": final, "benchmarks_test": bench,
           "n_trials_2019": n_trials}
    rep = REPO / "reports" / "backtest"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "e15_backtest_v2.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(daily_rets).to_csv(rep / "e15_daily_returns_v2.csv", index_label="date")
    summ = "; ".join(f"{k.split('|theta')[0]} {v['sharpe']:.2f}" for k, v in final.items() if k.endswith("|base"))
    summ += f"; EW {bench['ew_survivors']['sharpe']:.2f}; SPY {bench['spy_total_return'].get('sharpe', float('nan')):.2f}"
    run.finish("COMPLETED", "E15 test Sharpe (base costs): " + summ, out)
    print(summ)


if __name__ == "__main__":
    main()

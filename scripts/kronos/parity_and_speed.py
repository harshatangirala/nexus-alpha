"""Kronos integration checks (run before any use of Kronos output):

1. parity of NEXUS's reference sampler (no cache) with the vendored KronosPredictor, both under deterministic top-1
   decoding;
2. parity of the key/value-cached sampler with the reference, under top-1 decoding and under seeded sampling;
3. determinism and batch independence of the seeded sampler;
4. CPU seconds per stock for the forward-test configuration.

    python scripts/kronos/parity_and_speed.py [--models Kronos-mini Kronos-small] [--paths 32] [--lookback 400]

Writes reports/verification/kronos_parity_speed.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.data.calendar import offset_session  # noqa: E402
from nexus_alpha.kronos import engine as KE, weights as W  # noqa: E402
from nexus_alpha.kronos.vendor import KronosPredictor  # noqa: E402

TOP1 = 1e-9            # nucleus threshold that keeps only the most likely token: deterministic decoding


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Kronos-mini", "Kronos-small"])
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=20)
    a = ap.parse_args()
    d = B.read_daily("AAPL")
    k = int(np.flatnonzero(d.date.to_numpy() <= np.datetime64("2018-06-29"))[-1])
    rep = {"threads": torch.get_num_threads(), "lookback": a.lookback, "horizon": a.horizon, "paths": a.paths,
           "ticker_asof": "AAPL 2018-06-29"}
    for name in a.models:
        t_load = time.time()
        model, tok, mc = W.load(name)
        c = KE.context(d, k, a.lookback, a.horizon)
        H = a.horizon
        # 1. reference sampler vs the vendored predictor (top-1)
        pred = KronosPredictor(model, tok, device="cpu", max_context=mc)
        w = d.iloc[k + 1 - a.lookback: k + 1]
        x_df = w[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)
        x_ts = pd.Series(pd.to_datetime(w.date.to_numpy()))
        last = pd.DatetimeIndex([x_ts.iloc[-1]])
        y_ts = pd.Series(pd.DatetimeIndex([offset_session(last, i)[0] for i in range(1, H + 1)]))
        orig = pred.predict(x_df, x_ts, y_ts, pred_len=H, T=1.0, top_p=TOP1, sample_count=1, verbose=False)
        orig = orig[["open", "high", "low", "close"]].to_numpy()
        ref1 = KE.sample_paths(model, tok, [c], [1], 1, H, mc, top_p=TOP1, cache=False)[0, 0, :, :4]
        kv1 = KE.sample_paths(model, tok, [c], [1], 1, H, mc, top_p=TOP1, cache=True)[0, 0, :, :4]
        # 2. cached vs reference under seeded sampling
        t0 = time.time()
        ref8 = KE.sample_paths(model, tok, [c], [7], 8, H, mc, cache=False)
        t_ref8 = time.time() - t0
        t0 = time.time()
        kv8 = KE.sample_paths(model, tok, [c], [7], 8, H, mc, cache=True)
        t_kv8 = time.time() - t0
        same_path = np.all(np.isclose(ref8, kv8, rtol=1e-4, atol=1e-6), axis=(2, 3))[0]
        # 3. determinism and batch independence (cached)
        again = KE.sample_paths(model, tok, [c], [7], 8, H, mc)
        in_batch = KE.sample_paths(model, tok, [c, c], [99, 7], 8, H, mc)[1:2]
        # 4. speed (cached): one stock, and 8 stocks in one batch, with the forward-test path count
        t0 = time.time()
        KE.sample_paths(model, tok, [c], [3], a.paths, H, mc)
        one = time.time() - t0
        t0 = time.time()
        KE.sample_paths(model, tok, [c] * 8, list(range(8)), a.paths, H, mc)
        eight = (time.time() - t0) / 8
        r = {"max_context": mc, "total_check_s": round(time.time() - t_load, 1),
             "top1_reference_vs_vendored_predictor_max_rel": rel(orig, ref1),
             "top1_cached_vs_reference_max_rel": rel(ref1, kv1),
             "seeded_cached_vs_reference_identical_paths": f"{int(same_path.sum())}/{len(same_path)}",
             "seeded_cached_vs_reference_max_rel": rel(ref8, kv8),
             "deterministic": bool(np.array_equal(kv8, again)),
             "batch_independent": bool(np.allclose(kv8, in_batch, rtol=1e-5, atol=1e-6)),
             "speedup_cache_8_paths": round(t_ref8 / t_kv8, 1),
             "sec_per_stock_single": round(one, 2), "sec_per_stock_batch8": round(eight, 2),
             "minutes_for_500_stocks": round(500 * min(one, eight) / 60, 1)}
        rep[name] = r
        print(name, json.dumps(r), flush=True)
    out = REPO / "reports" / "verification" / "kronos_parity_speed.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

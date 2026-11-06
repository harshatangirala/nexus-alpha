"""E24: Kronos zero-shot, contaminated diagnostic on 2017-18 (pre-registered in docs/V3_RESEARCH.md; D-043).

    python scripts/experiments/e24_kronos_zero_shot.py --phase sample     # resumable; checkpoints every chunk
    python scripts/experiments/e24_kronos_zero_shot.py --phase score      # endpoints, decision, experiment record

Kronos's pre-training data includes NYSE/Nasdaq daily bars from 2000 to June 2024, so this period is inside its training
data. The result is a necessary condition only: no skill here means none to find; skill here may be memory, and only
the forward test can show otherwise.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402

MODEL = "Kronos-small"
LOOKBACK, HMAX, N_PATHS, T, TOP_P = 400, 20, 32, 1.0, 1.0
HORIZONS = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "15D": 15, "20D": 20}
SWING = ["2D", "5D", "10D", "15D", "20D"]
OUT = paths.ARTIFACTS / "kronos" / "e24"
BATCH, CHUNK = 8, 200
SEED = 11


def below_normal_priority() -> None:
    """The nightly forward test keeps precedence over this long diagnostic."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    else:
        import os
        os.nice(10)


def rows() -> pd.DataFrame:
    cols = ["ticker", "asof_date", "block"] + [f"y_{h}" for h in HORIZONS] + [f"ret_{h}" for h in HORIZONS]
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=cols)
    v = fr[fr.block == "valid"].copy()
    v["asof_date"] = pd.to_datetime(v.asof_date)
    return v[v.asof_date.dt.month % 2 == 1].sort_values(["ticker", "asof_date"]).reset_index(drop=True)


def sample() -> None:
    import torch
    from nexus_alpha.data import bars as B
    from nexus_alpha.indicators.core import ewma_sigma
    from nexus_alpha.kronos import engine as KE, weights as W
    below_normal_priority()
    OUT.mkdir(parents=True, exist_ok=True)
    R = rows()
    done = set()
    for f in sorted(OUT.glob("chunk_*.parquet")):
        d = pd.read_parquet(f, columns=["ticker", "asof_date"])
        done |= set(zip(d.ticker, pd.to_datetime(d.asof_date)))
    todo = R[[(t, a) not in done for t, a in zip(R.ticker, R.asof_date)]]
    print(f"rows {len(R)}, done {len(done)}, to do {len(todo)}; torch threads {torch.get_num_threads()}", flush=True)
    model, tok, mc = W.load(MODEL)
    t0, buf, skipped = time.time(), [], 0
    pend_ctx, pend_meta = [], []

    def flush_batch():
        if not pend_ctx:
            return
        P = KE.sample_paths(model, tok, pend_ctx, [m["seed"] for m in pend_meta], N_PATHS, HMAX, mc, T=T, top_p=TOP_P)
        pr = KE.label_probabilities(P, np.array([m["sigma"] for m in pend_meta]), HORIZONS)
        for i, m in enumerate(pend_meta):
            r = {"ticker": m["ticker"], "asof_date": m["asof_date"], "sigma": m["sigma"]}
            for h, o in pr.items():
                r.update({f"p_down_{h}": o["p"][i, 0], f"p_side_{h}": o["p"][i, 1], f"p_up_{h}": o["p"][i, 2],
                          f"ret_mean_{h}": o["ret_mean"][i], f"ret_sd_{h}": o["ret_sd"][i]})
            buf.append(r)
        pend_ctx.clear()
        pend_meta.clear()

    n_done = 0
    for t, g in todo.groupby("ticker", sort=True):
        d = B.read_daily(t)
        dates = d.date.to_numpy()
        sig = ewma_sigma(d.close).to_numpy()
        for a in g.asof_date:
            k = int(np.searchsorted(dates, np.datetime64(a), side="right") - 1)
            if k < 0 or pd.Timestamp(dates[k]) != a:
                skipped += 1
                continue
            c = KE.context(d, k, LOOKBACK, HMAX)
            if c is None:
                skipped += 1
                continue
            pend_ctx.append(c)
            pend_meta.append({"ticker": t, "asof_date": a, "sigma": float(sig[k]), "seed": KE.series_seed(t, a)})
            if len(pend_ctx) == BATCH:
                flush_batch()
            n_done += 1
            if len(buf) >= CHUNK:
                pd.DataFrame(buf).to_parquet(OUT / f"chunk_{int(time.time() * 1000)}.parquet", index=False)
                buf.clear()
                el = time.time() - t0
                print(f"{n_done}/{len(todo)} rows, {el / 60:.1f} min, eta {el / n_done * (len(todo) - n_done) / 3600:.1f} h", flush=True)
    flush_batch()
    if buf:
        pd.DataFrame(buf).to_parquet(OUT / f"chunk_{int(time.time() * 1000)}.parquet", index=False)
    print(f"sampling finished: {n_done} rows, skipped {skipped} ({(time.time() - t0) / 3600:.2f} h)", flush=True)


def b0_predictions(R: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """E22's stored B (F0 LightGBM) validation predictions, re-aligned to (ticker, as-of) exactly as E22 built them."""
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["ticker", "asof_date", "block"] +
                         [f"y_{h}" for h in HORIZONS])
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"]) & (fr.asof_date >= "2009-01-02")].reset_index(drop=True)
    out = {}
    for h in HORIZONS:
        f = paths.ARTIFACTS / "predictions" / "e22" / f"{h}_B_lgbm_valid.npy"
        if not f.exists():
            continue
        m = fr[f"y_{h}"].notna() & (fr.block == "valid")
        P = np.load(f)
        x = fr.loc[m, ["ticker", "asof_date"]].reset_index(drop=True)
        if len(x) != len(P):
            continue
        x[["b0_down", "b0_side", "b0_up"]] = P
        x["asof_date"] = pd.to_datetime(x.asof_date)
        out[h] = x
    return out


def score(smoke: bool = False) -> None:
    """smoke=True: run the whole scoring path on the chunks written so far and report only that it ran (no metric values
    are printed or stored), to test the code without looking at results before the registered run is complete."""
    import e19_market_model as E19
    import e21_ranker as E21
    from nexus_alpha.experiments import Run
    from nexus_alpha.kronos import weights as W
    R = rows()
    K = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("chunk_*.parquet"))], ignore_index=True)
    K["asof_date"] = pd.to_datetime(K.asof_date)
    K = K.drop_duplicates(["ticker", "asof_date"])
    D = R.merge(K, on=["ticker", "asof_date"], how="inner")
    tr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["block"] + [f"y_{h}" for h in HORIZONS])
    tr = tr[tr.block == "train"]
    prior = {h: np.array([(tr[f"y_{h}"] == c).mean() for c in range(3)]) / tr[f"y_{h}"].notna().mean() for h in HORIZONS}
    B0 = b0_predictions(R)
    rng = np.random.default_rng(SEED)
    month = (D.asof_date.dt.year * 12 + D.asof_date.dt.month).to_numpy()
    per_ll, per_ll_b0, ics, ics_b0, by_h = {}, {}, {}, {}, {}
    for h in HORIZONS:
        m = D[f"y_{h}"].notna().to_numpy()
        y = D.loc[m, f"y_{h}"].astype(int).to_numpy()
        P = D.loc[m, [f"p_down_{h}", f"p_side_{h}", f"p_up_{h}"]].to_numpy()
        ll_k = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
        ll_p = -np.log(prior[h][y])
        per_ll[h] = (month[m], ll_p - ll_k)
        ic = E21.daily_ic(D.asof_date[m].to_numpy(), D.loc[m, f"ret_mean_{h}"].to_numpy(), D.loc[m, f"ret_{h}"].to_numpy())
        ic.index = pd.DatetimeIndex(ic.index)
        ics[h] = ic
        row = {"n": int(m.sum()), "logloss_kronos": float(ll_k.mean()), "logloss_prior": float(ll_p.mean()),
               "mean_daily_ic": float(ic.mean()), "ic_days": int(len(ic))}
        if h in B0:
            J = D.loc[m, ["ticker", "asof_date", f"y_{h}", f"ret_{h}"]].merge(B0[h], on=["ticker", "asof_date"], how="left")
            ok = J.b0_up.notna().to_numpy()
            Pb = J.loc[ok, ["b0_down", "b0_side", "b0_up"]].to_numpy()
            yb = J.loc[ok, f"y_{h}"].astype(int).to_numpy()
            llb = -np.log(np.clip(Pb[np.arange(len(yb)), yb], 1e-12, 1))
            per_ll_b0[h] = (month[m][ok], -np.log(prior[h][yb]) - llb)
            icb = E21.daily_ic(J.asof_date[ok].to_numpy(), (J.b0_up - J.b0_down)[ok].to_numpy(), J.loc[ok, f"ret_{h}"].to_numpy())
            icb.index = pd.DatetimeIndex(icb.index)
            ics_b0[h] = icb
            row.update({"b0_rows": int(ok.sum()), "logloss_b0": float(llb.mean()), "b0_mean_daily_ic": float(icb.mean())})
        by_h[h] = row
    blen = max(1, math.ceil(max(HORIZONS[h] for h in SWING) / 21))
    res = {}
    for g, hs in (("1D", ["1D"]), ("swing", SWING)):
        L_ = max(1, math.ceil(max(HORIZONS[h] for h in hs) / 21))
        res[f"logloss_gain_vs_prior|{g}"] = dict(zip(("gain", "ci95"), E19.block_boot([per_ll[h] for h in hs], L_, rng)))
        res[f"rank_ic|{g}"] = dict(zip(("ic", "ci95"), E21.block_ci([ics[h] for h in hs], L_, rng)))
        if all(h in per_ll_b0 for h in hs):
            res[f"b0_logloss_gain_vs_prior|{g}"] = dict(zip(("gain", "ci95"), E19.block_boot([per_ll_b0[h] for h in hs], L_, rng)))
            res[f"b0_rank_ic|{g}"] = dict(zip(("ic", "ci95"), E21.block_ci([ics_b0[h] for h in hs], L_, rng)))
    if smoke:
        print(f"smoke: scored {len(D)} rows, {len(by_h)} horizons, endpoints {sorted(res)}; no values shown")
        return
    cond_ll = res["logloss_gain_vs_prior|swing"]["ci95"][0] > 0
    cond_ic = res["rank_ic|swing"]["ci95"][0] > 0
    met = bool(cond_ll or cond_ic)
    run = Run("e24_kronos_zero_shot", {"model": MODEL, "lookback": LOOKBACK, "paths": N_PATHS, "T": T, "top_p": TOP_P,
                                       "horizons": list(HORIZONS), "rows_rule": "valid block, odd calendar months",
                                       "weights": W.manifest().get(MODEL), "tokenizer": W.manifest().get("Kronos-Tokenizer-base"),
                                       "block_len_months": blen, "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    summ = (f"E24 Kronos zero-shot (contaminated 2017-18 diagnostic): necessary condition {'MET' if met else 'NOT met'}; "
            f"swing log-loss gain {res['logloss_gain_vs_prior|swing']['gain']:+.4f} {res['logloss_gain_vs_prior|swing']['ci95']}, "
            f"swing rank IC {res['rank_ic|swing']['ic']:+.4f} {res['rank_ic|swing']['ci95']}")
    run.finish("COMPLETED", summ, {"rows_sampled": int(len(K)), "rows_scored": int(len(D)), "rows_registered": int(len(R)),
                                   "by_horizon": by_h, "groups": res, "condition_logloss": bool(cond_ll),
                                   "condition_rank_ic": bool(cond_ic), "necessary_condition_met": met,
                                   "contamination": "pre-training to June 2024 includes NYSE/Nasdaq daily bars from 2000"})
    print(summ)
    for k, v in res.items():
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else [round(x, 5) for x in vv]) for kk, vv in v.items()})
    for h, r in by_h.items():
        print(h, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sample", "score", "smoke"], required=True)
    a = ap.parse_args()
    sample() if a.phase == "sample" else score(smoke=a.phase == "smoke")

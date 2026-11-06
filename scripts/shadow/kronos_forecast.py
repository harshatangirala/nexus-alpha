"""Forward test of Kronos zero-shot (`kronos_zs`; E24, D-043). Logs only if `kronos_zs` is frozen, which E24's
pre-registered necessary condition decides. No trading engine uses it.

    python scripts/shadow/kronos_forecast.py [--dry-run] [--limit N]

* Bars: only tonight's live cache (filled by the v1.1 forecaster; NEXUS_LIVE_CACHE), so no extra Yahoo requests. A
  stock without cached bars is counted as failed.
* Model: the frozen bundle (weights hash-checked file by file), Kronos-small, 400-bar context, 32 paths, T = 1.0,
  top-p = 1.0, seeds from (ticker, as-of), exactly as in E24.
* Output: shadow/kronos_zs/forecasts/<asof>.csv (first write wins; later runs add only missing stocks in an addendum),
  appended to the hash chain, scored by shadow_score.py (log loss vs the class prior, AUROC, daily rank IC of
  `rank_score` = the mean path return). Stops 10 minutes before the decision time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402

VERSION = "kronos_zs"
STOP_BEFORE_DECISION = pd.Timedelta(minutes=10)


def tonight_cache() -> Path:
    return paths.ARTIFACTS / "live_cache" / pd.Timestamp.now(tz="America/New_York").date().isoformat()


def rows_for(meta: dict, tickers: list, asofs: list, closes: list, pr: dict, bundle_id: str, run: pd.Timestamp) -> list:
    out = []
    for i, (t, a, c) in enumerate(zip(tickers, asofs, closes)):
        dec = SH.decision_utc_for(pd.Timestamp(a))
        for h, o in pr.items():
            out.append({"asof_session": str(pd.Timestamp(a).date()), "ticker": t, "horizon": h,
                        "p_down": float(o["p"][i, 0]), "p_side": float(o["p"][i, 1]), "p_up": float(o["p"][i, 2]),
                        "ret_q10": float(o["ret_q10"][i]), "ret_q50": float(o["ret_q50"][i]), "ret_q90": float(o["ret_q90"][i]),
                        "direction_skill_validated": False, "reference_close": float(c),
                        "data_quality": "kronos zero-shot (pre-trained to 2024-06)", "bundle_id": bundle_id,
                        "run_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(),
                        "late": bool(run > dec), "rank_score": float(o["ret_mean"][i])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not (SH.FROZEN_DIR / f"{VERSION}.json").exists():
        print(f"{VERSION} is not frozen (E24's necessary condition not met, or not evaluated yet): nothing logged")
        return
    bdir, rec = SH.verify_frozen(VERSION)
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    import torch
    from nexus_alpha.cv import store as ST
    from nexus_alpha.indicators.core import ewma_sigma
    from nexus_alpha.kronos import engine as KE
    from nexus_alpha.kronos import weights as W
    model, tok = W.load_files(bdir / "kronos_config.json", bdir / "kronos.safetensors", bdir / "tokenizer_config.json",
                              bdir / "tokenizer.safetensors")             # every file hash-checked by verify_frozen
    torch.set_grad_enabled(False)
    cache = tonight_cache()
    tickers = sorted(ST.sample_index().ticker.unique())
    if a.limit:
        tickers = tickers[: a.limit]
    t0, rows, failed = time.time(), [], []
    batch = []
    deadline = None

    def run_batch():
        nonlocal deadline
        if not batch:
            return
        ctxs = [b["ctx"] for b in batch]
        P = KE.sample_paths(model, tok, ctxs, [KE.series_seed(b["t"], b["asof"]) for b in batch], cfg["paths"],
                            cfg["horizon_sessions"], meta["max_context"], T=cfg["T"], top_p=cfg["top_p"])
        pr = KE.label_probabilities(P, np.array([b["sigma"] for b in batch]), meta["horizons"])
        rows.extend(rows_for(meta, [b["t"] for b in batch], [b["asof"] for b in batch], [b["close"] for b in batch], pr,
                             rec["bundle_id"], SH.now_utc()))
        batch.clear()

    for i, t in enumerate(tickers):
        f = cache / f"{t}_daily.pkl"
        if not f.exists():
            failed.append({"ticker": t, "error": "no cached bars tonight"})
            continue
        d = pd.read_pickle(f).sort_values("date").reset_index(drop=True)
        k = len(d) - 1
        c = KE.context(d, k, cfg["lookback"], cfg["horizon_sessions"])
        if c is None:
            failed.append({"ticker": t, "error": "short history"})
            continue
        asof = pd.Timestamp(d.date.iloc[k])
        dec = SH.decision_utc_for(asof)
        deadline = dec - STOP_BEFORE_DECISION if deadline is None else min(deadline, dec - STOP_BEFORE_DECISION)
        if SH.now_utc() > deadline:
            failed.extend({"ticker": x, "error": "not attempted: decision-time deadline"} for x in tickers[i:])
            break
        batch.append({"t": t, "asof": asof, "ctx": c, "sigma": float(ewma_sigma(d.close).iloc[k]),
                      "close": float(d.close.iloc[k])})
        if len(batch) == 8:
            run_batch()
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(tickers)} ({time.time() - t0:.0f}s)", flush=True)
    run_batch()
    if not rows:
        print("no Kronos forecasts produced", failed[:3])
        sys.exit(1)
    df = pd.DataFrame(rows)
    out_dir = SH.SHADOW_ROOT / VERSION / "forecasts"
    for asof, g in df.groupby("asof_session"):
        if a.dry_run:
            print(f"[dry-run] {asof}: {g.ticker.nunique()} tickers x {g.horizon.nunique()} horizons, late {int(g.late.sum())}, "
                  f"failed {len(failed)} ({time.time() - t0:.0f}s)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        prior = sorted(out_dir.glob(f"{asof}*.csv"))
        if prior:
            done = set(pd.concat([pd.read_csv(p, usecols=["ticker"]) for p in prior]).ticker)
            g = g[~g.ticker.isin(done)]
            if g.empty:
                print(f"{asof}: every ticker already logged; nothing rewritten")
                continue
            f = out_dir / f"{asof}_add{len(prior)}.csv"
        else:
            f = out_dir / f"{asof}.csv"
        g.to_csv(f, index=False, float_format="%.6g")
        SH.append_chain(VERSION, {"run_utc": SH.now_utc().isoformat(timespec="seconds"), "asof_session": asof,
                                  "file": f"forecasts/{f.name}", "file_sha256": SH.sha256_file(f), "n_rows": len(g),
                                  "n_failed": len(failed), "late_rows": int(g.late.sum())})
        print(f"logged {f.name}: {g.ticker.nunique()} tickers x {g.horizon.nunique()} horizons, late {int(g.late.sum())}, "
              f"failed {len(failed)} ({time.time() - t0:.0f}s)")
    if failed and not a.dry_run:
        (SH.SHADOW_ROOT / VERSION).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed).to_csv(SH.SHADOW_ROOT / VERSION / "last_run_failures.csv", index=False)
    if not a.dry_run:
        probs = SH.verify_chain(VERSION)
        print("chain:", "intact" if not probs else probs, flush=True)
        sys.stdout.flush()
        os._exit(0 if not probs else 1)


if __name__ == "__main__":
    main()

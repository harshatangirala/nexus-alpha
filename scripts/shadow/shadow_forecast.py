"""Forward shadow test, forecasting step (run every evening after the US close; see docs/FORWARD_SHADOW_TEST.md).

    python scripts/shadow/shadow_forecast.py --version v1.1 [--tickers AAPL MSFT] [--limit 20]

Verifies the frozen bundle's file hashes, forecasts every universe stock from live Yahoo bars as of the last completed
session, writes shadow/<version>/forecasts/<asof>.csv once (never rewritten) and appends it to the hash chain.
Rows whose run time is after their decision time (09:00 New York on t+1) are flagged late and are never scored.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.cv import store as ST  # noqa: E402
from nexus_alpha.inference.pipeline import NexusPredictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="forecast but do not write or chain (for testing)")
    a = ap.parse_args()

    bdir, rec = SH.verify_frozen(a.version)
    out_dir = SH.SHADOW_ROOT / a.version / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = a.tickers or sorted(ST.sample_index().ticker.unique())
    if a.limit:
        tickers = tickers[: a.limit]
    is_v2 = a.version.startswith("v2")
    if is_v2:
        from nexus_alpha.inference.pipeline_v2 import NexusPredictorV2
        P = NexusPredictorV2(bdir)
    else:
        P = NexusPredictor(bdir)
    rows, failed, t0 = [], [], time.time()

    def forecast(t: str) -> bool:
        try:
            r = P.predict(t, mode="live") if is_v2 else P.predict(t, mode="live", explain=False)
        except Exception as e:  # missing Yahoo data, rate limit, short history
            failed.append({"ticker": t, "error": repr(e)[:200]})
            return False
        run = SH.now_utc()
        if is_v2:
            asof_s, ref, dq = r["asof_session"], r["reference_close"], f"missing_share={r['features_missing_share']:.2f}"
            fc = {h: (f["p_down"], f["p_side"], f["p_up"], f["ret_q10"], f["ret_q50"], f["ret_q90"],
                      f["direction_skill_validated"]) for h, f in r["forecasts"].items()}
        else:
            asof_s, ref, dq = r.instrument.asof_session, r.instrument.reference_price, r.instrument.data_quality
            fc = {h: (f.down_probability, f.sideways_probability, f.up_probability, f.return_q10, f.expected_return,
                      f.return_q90, f.direction_skill_validated) for h, f in r.forecasts.items()}
        dec = SH.decision_utc_for(pd.Timestamp(asof_s))
        for h, (pd_, ps, pu, q10, q50, q90, sk) in fc.items():
            rows.append({"asof_session": asof_s, "ticker": t, "horizon": h, "p_down": pd_, "p_side": ps, "p_up": pu,
                         "ret_q10": q10, "ret_q50": q50, "ret_q90": q90, "direction_skill_validated": sk,
                         "reference_close": ref, "data_quality": dq, "bundle_id": rec["bundle_id"],
                         "run_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(),
                         "late": bool(run > dec)})
        return True

    for i, t in enumerate(tickers):
        forecast(t)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(tickers)} done, {time.time() - t0:.0f}s", flush=True)
    # one retry pass for empty Yahoo responses (rate limiting), after a pause
    retry = [f["ticker"] for f in failed if "no daily data" in f["error"]]
    if retry:
        time.sleep(60)
        failed[:] = [f for f in failed if f["ticker"] not in retry]
        for t in retry:
            forecast(t)
            time.sleep(1.0)
        print(f"retry pass: {len(retry)} tickers, {len(retry) - sum(f['ticker'] in retry for f in failed)} recovered")
    if not rows:
        print("no forecasts produced", failed[:5])
        sys.exit(1)
    df = pd.DataFrame(rows)
    # One file per as-of session. Stocks whose last Yahoo bar is older (halted, stale) go to that older session's
    # file only if it does not exist yet; otherwise they are dropped (first write wins).
    for asof, g in df.groupby("asof_session"):
        if a.dry_run:
            print(f"[dry-run] {asof}: {g.ticker.nunique()} tickers, late rows {int(g.late.sum())}")
            continue
        prior = sorted(out_dir.glob(f"{asof}*.csv"))
        if prior:   # first write wins: only tickers not yet logged for this session go into an addendum file
            done = set(pd.concat([pd.read_csv(f, usecols=["ticker"]) for f in prior]).ticker)
            g = g[~g.ticker.isin(done)]
            if g.empty:
                print(f"{asof}: every ticker already logged; nothing rewritten")
                continue
            f = out_dir / f"{asof}_add{len(prior)}.csv"
        else:
            f = out_dir / f"{asof}.csv"
        g.to_csv(f, index=False, float_format="%.6g")
        SH.append_chain(a.version, {"run_utc": SH.now_utc().isoformat(timespec="seconds"), "asof_session": asof,
                                    "file": f"forecasts/{f.name}", "file_sha256": SH.sha256_file(f),
                                    "n_rows": len(g), "n_failed": len(failed), "late_rows": int(g.late.sum())})
        print(f"logged {f.name}: {g.ticker.nunique()} tickers x {g.horizon.nunique()} horizons, "
              f"late rows {int(g.late.sum())}, failed tickers {len(failed)}")
    if failed and not a.dry_run:
        pd.DataFrame(failed).to_csv(SH.SHADOW_ROOT / a.version / "last_run_failures.csv", index=False)
    probs = SH.verify_chain(a.version)
    print("chain:", "intact" if not probs else probs)


if __name__ == "__main__":
    main()

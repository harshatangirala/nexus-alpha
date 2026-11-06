"""Forward shadow test of the E19 market-direction read (D-031): SPY 20D lean, logged every evening before decision time.

    python scripts/shadow/market_read.py [--version market_e19] [--dry-run]

Verifies the frozen bundle, builds the M2 features from Yahoo (SPY and market ETFs, market series, the 979-ticker
universe), scores the last completed session and writes shadow/<version>/forecasts/<asof>.csv (first write wins) to the
same hash chain format as the stock forecasts, so scripts/shadow/shadow_score.py scores it unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "shadow"))
warnings.filterwarnings("ignore")

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.macro import market_model as MM  # noqa: E402
from shadow_forecast import with_timeout  # noqa: E402

MIN_UNIVERSE = 900          # fewer universe tickers with data on the session = data problem: do not log a read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="market_e19")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bdir, rec = SH.verify_frozen(a.version)
    model = MM.MarketModel(bdir)
    feats, dq = with_timeout(lambda: MM.live_features(), 1800)
    asof = feats.index[-1]
    row = feats.iloc[-1]
    n_uni = dq["universe_with_data_last_session"]
    if n_uni < MIN_UNIVERSE:
        print(f"only {n_uni} universe tickers have a close on {asof.date()}: no read logged")
        _exit(1)
    preds = model.predict(row)
    run = SH.now_utc()
    dec = SH.decision_utc_for(pd.Timestamp(asof))
    spy_close = float(spy_close_on(asof))
    rows = [{"asof_session": str(asof.date()), "ticker": "SPY", "horizon": h, "p_down": p["p_down"], "p_side": p["p_side"],
             "p_up": p["p_up"], "ret_q10": None, "ret_q50": None, "ret_q90": None,
             "direction_skill_validated": bool(p["direction_skill_validated"]), "reference_close": spy_close,
             "data_quality": f"universe={n_uni}; missing_share={p['missing_share']:.2f}", "bundle_id": rec["bundle_id"],
             "run_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(), "late": bool(run > dec)}
            for h, p in preds.items()]
    df = pd.DataFrame(rows)
    print(df[["asof_session", "horizon", "p_down", "p_side", "p_up", "data_quality", "late"]].to_string(index=False))
    if a.dry_run:
        _exit(0)
    out_dir = SH.SHADOW_ROOT / a.version / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{asof.date()}.csv"
    if f.exists():
        print(f"{f.name} already logged; nothing rewritten")
    else:
        df.to_csv(f, index=False, float_format="%.6g")
        SH.append_chain(a.version, {"run_utc": SH.now_utc().isoformat(timespec="seconds"),
                                    "asof_session": str(asof.date()), "file": f"forecasts/{f.name}",
                                    "file_sha256": SH.sha256_file(f), "n_rows": len(df), "n_failed": 0,
                                    "late_rows": int(df.late.sum())})
        print(f"logged {f.name}")
    probs = SH.verify_chain(a.version)
    print("chain:", "intact" if not probs else probs, flush=True)
    _exit(0 if not probs else 1)


def _exit(code: int) -> None:
    """Exit without waiting for abandoned request threads, after flushing output (os._exit skips the flush)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def spy_close_on(asof) -> float:
    from nexus_alpha.macro.context import _yahoo_daily
    d = _yahoo_daily("SPY", (pd.Timestamp(asof) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    return float(d.loc[d.date == pd.Timestamp(asof), "close"].iloc[0])


if __name__ == "__main__":
    main()

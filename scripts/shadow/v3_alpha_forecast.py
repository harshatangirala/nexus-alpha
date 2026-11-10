"""Forward test of v3_alpha: LightGBM on F0 + Qlib Alpha158 (E25 kept on rank IC; D-047, D-049). No trading engine uses it.

    python scripts/shadow/v3_alpha_forecast.py [--dry-run]

Inputs, all from tonight's run:
* the F0 rows the frozen v2 predictor assembled for each stock (artifacts/live_features/<asof>/v2_x.parquet, written by
  shadow_forecast.py --version v2);
* tonight's cached Yahoo daily bars (artifacts/live_cache/<New York date>/<ticker>_daily.pkl), the same bars v1.1 and
  v2 forecast from, for the 157 Alpha158 features of the as-of session.

Only the horizons in the frozen record are logged: 5D-1Y. The live-parity check passed there and failed at 1D/2D
(reports/verification/alpha158_live_parity.json; D-049). Writes shadow/v3_alpha/forecasts/<asof>.csv once (first
write wins), with the v3_fund row schema plus `rank_score` (p_up - p_down), and appends it to the hash chain.
shadow_score.py scores it. Promotion needs the forward criterion in docs/V3_RESEARCH.md.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.indicators import alpha158 as A  # noqa: E402

VERSION = "v3_alpha"
MIN_STOCKS = 450


def bar_cache_for(asof: str, tickers: list[str]) -> Path | None:
    """The newest live-cache folder whose daily bars reach the as-of session for at least MIN_STOCKS tickers."""
    croot = paths.ARTIFACTS / "live_cache"
    for d in sorted((p for p in croot.glob("*") if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)),
                    reverse=True):
        ok = 0
        for t in tickers[:60]:                  # a quick probe; every ticker is checked again when it is scored
            f = d / f"{t}_daily.pkl"
            if f.exists() and str(pd.Timestamp(pd.read_pickle(f).date.max()).date()) == asof:
                ok += 1
        if ok >= 0.9 * min(60, len(tickers)):
            return d
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bdir, rec = SH.verify_frozen(VERSION)
    horizons = list(rec["horizons"])
    models = {h: lgb.Booster(model_file=str(bdir / f"lgbm_{h}.txt")) for h in horizons}
    import json
    cols = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))["cols"]
    feats = sorted(f for f in (paths.ARTIFACTS / "live_features").glob("*/v2_x.parquet")
                   if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.parent.name))
    if not feats:
        sys.exit("no v2 feature rows logged tonight (run shadow_forecast.py --version v2 first)")
    X = pd.read_parquet(feats[-1])
    asof = str(X.asof_session.max())
    X = X[X.asof_session == asof].reset_index(drop=True)
    expected = str(SH.expected_asof().date())
    if asof != expected or len(X) < MIN_STOCKS:
        sys.exit(f"v2 feature rows are for {asof} with {len(X)} stocks, expected session {expected} with at least "
                 f"{MIN_STOCKS}: not logged (a partial or stale log would be permanent)")
    cache = bar_cache_for(asof, sorted(X.ticker))
    if cache is None:
        sys.exit(f"no bar cache reaches session {asof}: not logged")
    a158, failed = [], []
    for t in X.ticker:
        try:
            s = A.features_at(pd.read_pickle(cache / f"{t}_daily.pkl"), asof)
            a158.append(s.rename(t))
        except Exception as e:  # noqa: BLE001  (counted; the stock is not logged)
            failed.append((t, f"{type(e).__name__}: {e}"[:100]))
    F = pd.DataFrame(a158)
    F.index.name = "ticker"
    X = X.merge(F.reset_index(), on="ticker", how="inner")
    if len(X) < MIN_STOCKS:
        sys.exit(f"only {len(X)} stocks have Alpha158 for {asof} (failed: {failed[:5]}): not logged")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        sys.exit(f"feature rows lack {len(missing)} model columns, e.g. {missing[:3]}")
    run = SH.now_utc()
    rows = []
    for h, bst in models.items():
        P = bst.predict(X[cols].astype(float))
        for i, r in enumerate(X.itertuples(index=False)):
            dec = pd.Timestamp(r.decision_utc)
            rows.append({"asof_session": asof, "ticker": r.ticker, "horizon": h, "p_down": P[i, 0], "p_side": P[i, 1],
                         "p_up": P[i, 2], "ret_q10": None, "ret_q50": None, "ret_q90": None,
                         "direction_skill_validated": False, "reference_close": r.reference_close,
                         "data_quality": f"bars={cache.name}", "bundle_id": rec["bundle_id"],
                         "run_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(),
                         "late": bool(run > dec), "rank_score": float(P[i, 2] - P[i, 0])})
    df = pd.DataFrame(rows)
    if a.dry_run:
        print(f"[dry-run] {asof}: {df.ticker.nunique()} tickers x {df.horizon.nunique()} horizons, late {int(df.late.sum())}, "
              f"failed {len(failed)} {failed[:3]}, bars from {cache.name}")
        return
    out_dir = SH.SHADOW_ROOT / VERSION / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{asof}.csv"
    if f.exists():
        print(f"{asof}: already logged; nothing rewritten")
        return
    df.to_csv(f, index=False, float_format="%.6g")
    SH.append_chain(VERSION, {"run_utc": run.isoformat(timespec="seconds"), "asof_session": asof,
                              "file": f"forecasts/{f.name}", "file_sha256": SH.sha256_file(f), "n_rows": len(df),
                              "n_failed": len(failed), "late_rows": int(df.late.sum())})
    probs = SH.verify_chain(VERSION)
    print(f"logged {f.name}: {df.ticker.nunique()} tickers x {df.horizon.nunique()} horizons, late rows {int(df.late.sum())}, "
          f"failed {len(failed)}; chain {'intact' if not probs else probs}", flush=True)
    os._exit(0 if not probs else 1)


if __name__ == "__main__":
    main()

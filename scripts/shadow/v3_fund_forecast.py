"""Forward test of v3_fund: LightGBM on F0 + industry-neutral fundamentals (E23 kept; D-042). No trading engine uses it.

    python scripts/shadow/v3_fund_forecast.py [--dry-run]

Inputs, all from tonight's run:
* the numeric features that the frozen v2 predictor assembled for each stock
  (artifacts/live_features/<asof>/v2_x.parquet, written by shadow_forecast.py --version v2);
* the live fundamentals cross-section (artifacts/fundamentals/live_panel.parquet, live_fundamentals.py) for the same
  session.

Writes shadow/v3_fund/forecasts/<asof>.csv once (first write wins), with the v2 row schema plus `rank_score`
(p_up - p_down), and appends it to the hash chain. shadow_score.py scores it: log loss vs the class prior, AUROC,
daily rank IC and quintile spread. Promotion needs the forward criterion in docs/V3_RESEARCH.md (D-042).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import json

import lightgbm as lgb
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402

VERSION = "v3_fund"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--features", help="a v2 feature-row file to use instead of tonight's (testing; implies --dry-run)")
    a = ap.parse_args()
    bdir, rec = SH.verify_frozen(VERSION)
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    models = {h: lgb.Booster(model_file=str(bdir / f"lgbm_{h}.txt")) for h in meta["horizons"]}
    cols, fcols = meta["cols"], meta["fund_cols"]
    if a.features:
        a.dry_run = True
    import re
    feats = [Path(a.features)] if a.features else sorted(
        f for f in (paths.ARTIFACTS / "live_features").glob("*/v2_x.parquet")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.parent.name))           # session folders only (audit 4)
    if not feats:
        sys.exit("no v2 feature rows logged tonight (run shadow_forecast.py --version v2 first)")
    X = pd.read_parquet(feats[-1])
    asof = str(X.asof_session.max())
    X = X[X.asof_session == asof]
    lp_path = paths.ARTIFACTS / "fundamentals" / "live_panel.parquet"
    if not lp_path.exists():
        sys.exit("no live fundamentals cross-section (run live_fundamentals.py first)")
    LP = pd.read_parquet(lp_path)
    lp_session = str(pd.Timestamp(LP.date.iloc[0]).date())
    if lp_session != asof:
        sys.exit(f"live fundamentals are for {lp_session}, the v2 features for {asof}: not logged")
    missing = [c for c in cols if c not in X.columns and c not in fcols]
    if missing:
        sys.exit(f"v2 feature rows lack {len(missing)} F0 columns, e.g. {missing[:3]}")
    X = X.drop(columns=[c for c in fcols if c in X.columns]).merge(LP[["ticker"] + fcols], on="ticker", how="left")
    X["fund_present"] = X.fund_present.fillna(0.0)
    run = SH.now_utc()
    rows = []
    for h, bst in models.items():
        P = bst.predict(X[cols].astype(float))
        for i, r in enumerate(X.itertuples(index=False)):
            dec = pd.Timestamp(r.decision_utc)
            rows.append({"asof_session": asof, "ticker": r.ticker, "horizon": h, "p_down": P[i, 0], "p_side": P[i, 1],
                         "p_up": P[i, 2], "ret_q10": None, "ret_q50": None, "ret_q90": None,
                         "direction_skill_validated": False, "reference_close": r.reference_close,
                         "data_quality": f"fund_present={int(getattr(r, 'fund_present'))}", "bundle_id": rec["bundle_id"],
                         "run_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(),
                         "late": bool(run > dec), "rank_score": float(P[i, 2] - P[i, 0])})
    df = pd.DataFrame(rows)
    out_dir = SH.SHADOW_ROOT / VERSION / "forecasts"
    if a.dry_run:
        print(f"[dry-run] {asof}: {df.ticker.nunique()} tickers x {df.horizon.nunique()} horizons, late {int(df.late.sum())}, "
              f"fundamentals present {X.fund_present.mean():.0%}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{asof}.csv"
    if f.exists():
        print(f"{asof}: already logged; nothing rewritten")
        return
    df.to_csv(f, index=False, float_format="%.6g")
    SH.append_chain(VERSION, {"run_utc": run.isoformat(timespec="seconds"), "asof_session": asof,
                              "file": f"forecasts/{f.name}", "file_sha256": SH.sha256_file(f), "n_rows": len(df),
                              "n_failed": 0, "late_rows": int(df.late.sum())})
    probs = SH.verify_chain(VERSION)
    print(f"logged {f.name}: {df.ticker.nunique()} tickers x {df.horizon.nunique()} horizons, late rows {int(df.late.sum())}; "
          f"chain {'intact' if not probs else probs}", flush=True)
    os._exit(0 if not probs else 1)


if __name__ == "__main__":
    main()

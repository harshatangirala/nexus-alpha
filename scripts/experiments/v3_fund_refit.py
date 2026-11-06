"""Refit v3_fund, the candidate that E22/E23's pre-registered use clauses produce (D-042).

    python scripts/experiments/v3_fund_refit.py [--e23 <run dir>]

E23 kept the industry-neutral fundamentals (LightGBM on both endpoints), and E22's universe-rank set was not kept. The
use clauses therefore give: LightGBM, F0 (v1.1 numeric features) + FUND_IND (fi_* ranks, days since filing, present
flag), with E07 parameters, refitted on as-of 2009-2018 (train, inner and valid blocks) for E23's 9 horizons.

Tree counts follow E08's refit convention: E23's selection-stage count for B_FUND_IND (early stopping on 2016, or the
inherited count at 1Y) scaled by refit rows / selection-train rows. Output: artifacts/bundles/<run id>/ (one LightGBM
text model per horizon and bundle_meta.json), then `freeze_model.py --kind fund --version v3_fund`.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.fundamentals.ratios import RATIOS  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2, LABEL_VERSION_V2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e23", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    e23 = Path(a.e23) if a.e23 else Path(sorted(glob.glob(str(REPO / "experiments" / "*_e23_industry_fundamentals_*")))[-1])
    m23 = json.loads((e23 / "metrics.json").read_text(encoding="utf-8"))
    assert m23["kept"] and m23["keep"]["lgbm|logloss"]["kept"], "E23 did not keep LightGBM: no v3_fund"
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid"]) & (fr.asof_date >= "2009-01-02")].reset_index(drop=True)
    fcols = [f"fi_{c}" for c in RATIOS + ["days_since_filing"]] + ["fund_present"]
    P = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "panel.parquet", columns=["ticker", "date"] + fcols)
    P = P.rename(columns={"date": "asof_date"})
    P["asof_date"] = pd.to_datetime(P.asof_date).astype("datetime64[ns]")
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    fr = fr.merge(P, on=["ticker", "asof_date"], how="left", validate="many_to_one")
    fr["fund_present"] = fr.fund_present.fillna(0.0)
    f0 = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    cols = f0 + fcols
    horizons = list(L.REP_HORIZONS)
    run = Run("v3_fund_refit", {"from": e23.name, "cols": len(cols), "horizons": horizons, "rows_from": "2009-01-02",
                                "blocks": ["train", "inner", "valid"], "params": L.E07_PARAMS, "decision": "D-042"},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                      "fund_panel": paths.ARTIFACTS / "fundamentals" / "panel.parquet"})
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    prior, trees = {}, {}
    for h in horizons:
        lab = fr[f"y_{h}"].notna().to_numpy()
        n_sel = int((lab & (fr.block == "train").to_numpy()).sum())
        rounds = max(1, int(round(m23["by_horizon"][h]["B_FUND_IND"]["lgbm"]["trees"] * lab.sum() / n_sel)))
        y = fr.loc[lab, f"y_{h}"].astype(int).to_numpy()
        prm = {**L.LGB_BASE, **L.E07_PARAMS, "num_threads": a.threads}
        bst = lgb.train(prm, lgb.Dataset(fr.loc[lab, cols], y), num_boost_round=rounds)
        bst.save_model(str(bdir / f"lgbm_{h}.txt"))
        prior[h] = [float((y == c).mean()) for c in range(3)]
        trees[h] = rounds
        print(f"{h}: {lab.sum()} rows, {rounds} trees ({time.time() - t0:.0f}s)", flush=True)
    (bdir / "bundle_meta.json").write_text(json.dumps({
        "bundle_id": run.run_id, "recipe": "LightGBM (E07 params), F0 + FUND_IND (fi_* industry-neutral ranks); E23 kept",
        "trained_through_block": "valid (as-of 2009-01-02 .. 2018-12-31)", "cols": cols, "fund_cols": fcols,
        "trees": trees, "horizons": {h: HORIZONS_V2[h] for h in horizons}, "label_version": LABEL_VERSION_V2,
        "refit_class_prior": prior, "selection_run": e23.name, "decision": "D-042"}, indent=1), encoding="utf-8")
    run.finish("COMPLETED", f"v3_fund refitted on 2009-2018: bundle {run.run_id}",
               {"trees": trees, "prior": prior, "elapsed_s": round(time.time() - t0, 1)})
    print("bundle", run.run_id)


if __name__ == "__main__":
    main()

"""Refit v3_alpha, the candidate that E25's pre-registered use clause produces (D-047, D-049).

    python scripts/experiments/v3_alpha_refit.py [--e25 <run dir>]

E25 kept Alpha158 on the rank-IC endpoint for both families, the LightGBM family included. The use clause therefore
gives: LightGBM, F0 (the 80 v1.1 numeric features) + A158 (157 Alpha158 features, nexus_alpha.indicators.alpha158),
E07 parameters, refitted on as-of 2003-2018 (train, inner and valid blocks) for E25's 9 horizons.

Tree counts follow E08's refit convention, as for v3_fund: E25's selection-stage count for B_A158 (early stopping on
2016, or the inherited count at 1Y) scaled by refit rows / selection-train rows. Output: artifacts/bundles/<run id>/
(one LightGBM text model per horizon and bundle_meta.json), then `freeze_model.py --kind alpha --version v3_alpha`.
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
from nexus_alpha.indicators import alpha158 as A  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2, LABEL_VERSION_V2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e25", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    e25 = Path(a.e25) if a.e25 else Path(sorted(glob.glob(str(REPO / "experiments" / "*_e25_alpha158_*")))[-1])
    m25 = json.loads((e25 / "metrics.json").read_text(encoding="utf-8"))
    assert m25["kept"] and m25["keep"]["lgbm|rank_ic"]["kept"], "E25 did not keep the LightGBM family: no v3_alpha"
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid"])].reset_index(drop=True)
    P = pd.read_parquet(paths.ARTIFACTS / "features" / "alpha158.parquet")
    P["asof_date"] = pd.to_datetime(P.asof_date).astype("datetime64[ns]")
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    n0 = len(fr)
    fr = fr.merge(P, on=["ticker", "asof_date"], how="left", validate="one_to_one")
    assert len(fr) == n0 and fr[A.COLUMNS[0]].notna().mean() > 0.999
    f0 = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    cols = f0 + list(A.COLUMNS)
    horizons = list(L.REP_HORIZONS)
    run = Run("v3_alpha_refit", {"from": e25.name, "cols": len(cols), "horizons": horizons, "rows": "2003-2018",
                                 "blocks": ["train", "inner", "valid"], "params": L.E07_PARAMS,
                                 "decision": "D-047, D-049"},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                      "alpha158": paths.ARTIFACTS / "features" / "alpha158.parquet"})
    bdir = paths.ARTIFACTS / "bundles" / run.run_id
    bdir.mkdir(parents=True, exist_ok=True)
    prior, trees = {}, {}
    for h in horizons:
        lab = fr[f"y_{h}"].notna().to_numpy()
        n_sel = int((lab & (fr.block == "train").to_numpy()).sum())
        rounds = max(1, int(round(m25["by_horizon"][h]["B_A158"]["lgbm"]["trees"] * lab.sum() / n_sel)))
        y = fr.loc[lab, f"y_{h}"].astype(int).to_numpy()
        prm = {**L.LGB_BASE, **L.E07_PARAMS, "num_threads": a.threads}
        bst = lgb.train(prm, lgb.Dataset(fr.loc[lab, cols], y), num_boost_round=rounds)
        bst.save_model(str(bdir / f"lgbm_{h}.txt"))
        prior[h] = [float((y == c).mean()) for c in range(3)]
        trees[h] = rounds
        print(f"{h}: {lab.sum()} rows, {rounds} trees ({time.time() - t0:.0f}s)", flush=True)
    (bdir / "bundle_meta.json").write_text(json.dumps({
        "bundle_id": run.run_id, "kind": "alpha",
        "recipe": "LightGBM (E07 params), F0 + A158 (Qlib Alpha158 without VWAP0); E25 kept on rank IC",
        "trained_through_block": "valid (as-of 2003 .. 2018-12-31)", "cols": cols, "a158_cols": list(A.COLUMNS),
        "trees": trees, "horizons": {h: HORIZONS_V2[h] for h in horizons}, "label_version": LABEL_VERSION_V2,
        "refit_class_prior": prior, "selection_run": e25.name, "decision": "D-047, D-049"}, indent=1), encoding="utf-8")
    run.finish("COMPLETED", f"v3_alpha refitted on 2003-2018: bundle {run.run_id}",
               {"trees": trees, "prior": prior, "elapsed_s": round(time.time() - t0, 1)})
    print("bundle", run.run_id)


if __name__ == "__main__":
    main()

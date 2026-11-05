"""E11 macro ladder (pre-registered in docs/V2_ARCHITECTURE.md section 4).

Nested feature sets on the v2 frame, every v2 horizon, two model families (multinomial logistic, LightGBM):
  F0 v1.1 numeric (80) | F1 + rel_* | F2 + mkt_* | F3 + gmd_* | F4 + fred_* (only if present in the frame)
plus the control <top>_placebo: whole-date permutation of the date-level blocks (mkt_*, gmd_*, fred_*), which must
remove any gain that comes from real market/macro information rather than from fitting date effects.
Engine and keep rule: nexus_alpha.evaluation.ladder.
"""
from __future__ import annotations

import argparse
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
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEED = 11


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--horizons", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    num = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    rel = [c for c in fr.columns if c.startswith("rel_")]
    mkt = [c for c in fr.columns if c.startswith("mkt_")]
    gmd = [c for c in fr.columns if c.startswith("gmd_") and c != "gmd_data_year"]
    fred = [c for c in fr.columns if c.startswith("fred_")]
    sets = {"F0": num, "F1": num + rel, "F2": num + rel + mkt, "F3": num + rel + mkt + gmd}
    if fred:
        sets["F4"] = sets["F3"] + fred
    order = list(sets)
    top = order[-1]
    date_level = mkt + gmd + fred
    dates = np.sort(fr.asof_date.unique())
    perm = dict(zip(dates, np.random.default_rng(SEED).permutation(dates)))
    dl = fr.groupby("asof_date")[date_level].first()
    plac = dl.loc[[perm[d] for d in fr.asof_date]].reset_index(drop=True)
    plac.columns = [f"PL_{c}" for c in date_level]
    fr = pd.concat([fr, plac], axis=1)
    sets[f"{top}_placebo"] = num + rel + [f"PL_{c}" for c in date_level]
    horizons = a.horizons or list(HORIZONS_V2)
    run = Run("e11_macro_ladder", {"sets": {k: len(v) for k, v in sets.items()}, "horizons": horizons,
                                   "lgb": L.E07_PARAMS, "logistic_l2_per_row": 1e-3, "n_boot": 1000,
                                   "placebo": "whole-date permutation of date-level blocks", "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pred_dir = paths.artifact("predictions", "e11", "x").parent
    res, rows_ll = L.fit_eval(fr, sets, horizons, a.threads, pred_dir, log=lambda s: print(s, flush=True))
    comps = [(order[i], order[i - 1]) for i in range(1, len(order))] + [(f"{top}_placebo", "F1"), (top, "F0")]
    gains = L.group_gains(rows_ll, comps, horizons)
    keep = {f"{m}|{order[i]}": L.keep_decision(gains, m, order[i], order[i - 1])
            for m in ("lgbm", "logistic") for i in range(1, len(order))}
    placebo_ok = {m: not L.keep_decision(gains, m, f"{top}_placebo", "F1") for m in ("lgbm", "logistic")}
    metrics = {"by_horizon": res, "gains": gains, "keep_rule_decisions": keep,
               "placebo_shows_no_gain": placebo_ok, "sets": sets, "elapsed_s": round(time.time() - t0, 1)}
    summ = "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", f"E11 ladder, {len(horizons)} horizons. Keep rule (validation): {summ}. "
                            f"Placebo shows no gain: {placebo_ok}", metrics)
    print(summ, "| placebo ok:", placebo_ok)


if __name__ == "__main__":
    main()

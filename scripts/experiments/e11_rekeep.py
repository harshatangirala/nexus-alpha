"""D-032: E11 keep decisions recomputed with E19's horizon-length block bootstrap and validation-window eligibility.

Per-row log losses are rebuilt from E11's saved predictions (artifacts/predictions/e11) and the v2 frame's labels; the
group bootstrap is E19's (`e19_market_model.block_boot`: circular runs of max(1, ceil(h/21)) months for the group's
longest horizon, the same months for every horizon, 1000 draws, seed 11). A horizon enters the decision only if its
validation block holds >= 4 non-overlapping label windows (distinct labelled validation sessions / h). The carry rule
(union of blocks kept by a family whose placebo passes) is E11's, unchanged. Writes an `e11_rekeep` run that
`ladder.carried_feature_set` uses in place of the as-registered decisions.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

import e19_market_model as E19  # noqa: E402
from nexus_alpha import paths  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402


def main() -> None:
    e11_dir = sorted((REPO / "experiments").glob("*_e11_macro_ladder_*"), reverse=True)
    e11_dir = next(r for r in e11_dir if json.loads((r / "manifest.json").read_text(encoding="utf-8")).get("status") == "COMPLETED")
    m = json.loads((e11_dir / "metrics.json").read_text(encoding="utf-8"))
    sets, horizons = m["sets"], list(m["by_horizon"])
    order = [k for k in sets if not k.endswith("placebo")]
    top = order[-1]
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                         columns=["asof_date", "block"] + [f"y_{h}" for h in horizons])
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    blk = fr.block.to_numpy()
    month = (fr.asof_date.dt.year * 12 + fr.asof_date.dt.month).to_numpy()
    rows, eligible, windows = {}, {}, {}
    for h in horizons:
        y = fr[f"y_{h}"].to_numpy()
        lab = ~np.isnan(y)
        va = lab & (blk == "valid")
        windows[h] = round(float(fr.asof_date[va].nunique() / HORIZONS_V2[h]), 2)
        eligible[h] = windows[h] >= E19.MIN_WINDOWS
        for split in ("valid", "test"):
            mm = lab & (blk == split)
            yy = y[mm].astype(int)
            for name in sets:
                for mdl in ("lgbm", "logistic"):
                    P = np.load(paths.ARTIFACTS / "predictions" / "e11" / f"{h}_{name}_{mdl}_{split}.npy").astype(float)
                    assert len(P) == len(yy), (h, name, mdl, split)
                    rows[(h, name, mdl, split)] = (L.ll_rows(yy, P), month[mm])
    # consistency: rebuilt mean losses equal E11's recorded ones
    for h in horizons:
        for name in sets:
            for mdl in ("lgbm", "logistic"):
                rec = m["by_horizon"][h][name][mdl]["valid_ll"]
                assert abs(rows[(h, name, mdl, "valid")][0].mean() - rec) < 1e-5, (h, name, mdl)
    rng = np.random.default_rng(11)
    comps = [(order[i], order[i - 1]) for i in range(1, len(order))] + [(f"{top}_placebo", "F1"), (top, "F0")]
    gains = {}
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in L.GROUPS.items():
                    hs = [h for h in hs if h in horizons and (split == "test" or eligible[h])]
                    if not hs:
                        continue
                    per = [(rows[(h, old, mdl, split)][1], rows[(h, old, mdl, split)][0] - rows[(h, new, mdl, split)][0])
                           for h in hs]
                    blen = max(max(1, math.ceil(HORIZONS_V2[h] / 21)) for h in hs)
                    pt, ci = E19.block_boot(per, blen, rng)
                    gains[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": pt, "ci95": ci, "horizons": hs, "block_months": blen}
    keep = {f"{mdl}|{order[i]}": L.keep_decision(gains, mdl, order[i], order[i - 1])
            for mdl in ("lgbm", "logistic") for i in range(1, len(order))}
    placebo_ok = {mdl: not L.keep_decision(gains, mdl, f"{top}_placebo", "F1") for mdl in ("lgbm", "logistic")}
    run = Run("e11_rekeep", {"e11_run": e11_dir.name, "rule": "D-032", "n_boot": 1000, "seed": 11,
                             "min_windows": E19.MIN_WINDOWS}, inputs={})
    summ = "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", f"E11 keep decisions under D-032: {summ}. Placebo shows no gain: {placebo_ok}",
               {"e11_run": e11_dir.name, "sets": sets, "keep_rule_decisions": keep, "placebo_shows_no_gain": placebo_ok,
                "gains": gains, "validation_windows": windows, "eligible": eligible,
                "as_registered": {"keep_rule_decisions": m["keep_rule_decisions"],
                                  "placebo_shows_no_gain": m["placebo_shows_no_gain"]}})
    print("windows:", windows)
    print("as registered:", m["keep_rule_decisions"])
    print("D-032:        ", keep, "| placebo ok:", placebo_ok)
    for k in ("lgbm|F3_vs_F2|valid|intraday", "lgbm|F3_vs_F2|valid|long", "lgbm|F4_vs_F3|valid|swing",
              "lgbm|F4_vs_F3|valid|long"):
        v = gains.get(k)
        print(k, None if not v else (round(v["gain"], 5), [round(x, 5) for x in v["ci95"]], v["horizons"]))


if __name__ == "__main__":
    main()

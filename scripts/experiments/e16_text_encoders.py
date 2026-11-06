"""E16 text encoder comparison (pre-registered in docs/V2_ARCHITECTURE.md section 4).

Base B = the feature set carried forward by the latest completed E11 run (F0 + every block that passed the keep rule
for a model family whose placebo also passed). Compared sets: B + v1 news features (TF-IDF/SVD + Loughran-McDonald),
B + FinBERT features (fb_*), B + both. Same engine, bootstrap and keep rule as E11 (nexus_alpha.evaluation.ladder).
Also reported on the subset of rows that have news.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402


def carried_set() -> tuple[list[str], str]:
    return L.carried_feature_set(REPO / "experiments")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--horizons", nargs="*")
    a = ap.parse_args()
    base, e11 = carried_set()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    nv1 = pd.read_parquet(paths.ARTIFACTS / "features" / "news_features.parquet")
    fb = pd.read_parquet(paths.ARTIFACTS / "features" / "finbert_features.parquet")
    fb["fb_present"] = 1.0
    fr = fr.merge(nv1, on="sample_id", how="left").merge(fb, on="sample_id", how="left")
    fr["fb_present"] = fr.fb_present.fillna(0.0)
    v1 = [c for c in nv1.columns if c != "sample_id"]
    fbc = [c for c in fb.columns if c != "sample_id"]
    sets = {"B": base, "B_newsv1": base + v1, "B_finbert": base + fbc, "B_both": base + v1 + fbc}
    horizons = a.horizons or list(L.REP_HORIZONS)
    run = Run("e16_text_encoders", {"base_from": e11, "sets": {k: len(v) for k, v in sets.items()},
                                    "horizons": horizons, "finbert": "ProsusAI/finbert (fixed weights)"},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                      "finbert": paths.ARTIFACTS / "features" / "finbert_features.parquet"})
    pred_dir = paths.artifact("predictions", "e16", "x").parent
    res, rows_ll = L.fit_eval(fr, sets, horizons, a.threads, pred_dir, log=lambda s: print(s, flush=True))
    comps = [("B_newsv1", "B"), ("B_finbert", "B"), ("B_both", "B"), ("B_both", "B_newsv1")]
    gains = L.group_gains(rows_ll, comps, horizons)
    keep = {f"{m}|{new}_vs_{old}": L.keep_decision(gains, m, new, old) for m in ("lgbm", "logistic") for new, old in comps}
    # the same comparisons restricted to rows that have news (row order matches fit_eval's masks)
    news_rows = {}
    for (h, name, mdl, split), (l, mo) in rows_ll.items():
        m = fr[f"y_{h}"].notna().to_numpy() & (fr.block == split).to_numpy()
        has = fr.loc[m, "news_present"].fillna(0).to_numpy() > 0
        news_rows[(h, name, mdl, split)] = (l[has], mo[has])
    gains_news = L.group_gains(news_rows, comps, horizons)
    news_share = {b: float(fr.loc[fr.block == b, "news_present"].fillna(0).mean()) for b in ("train", "valid", "test")}
    run.finish("COMPLETED", "E16 text encoders: " + "; ".join(f"{k}: {'keep' if v else 'drop'}" for k, v in keep.items()),
               {"by_horizon": res, "gains": gains, "gains_rows_with_news": gains_news, "keep_rule_decisions": keep,
                "sets": sets,
                "news_present_share": news_share})
    print(keep)


if __name__ == "__main__":
    main()

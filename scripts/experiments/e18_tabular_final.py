"""E18: the tabular feature set chosen by E11/E16 (same rule as E14's tabular_choice), fitted on all 23 v2 horizons with
the shared engine (LightGBM + logistic; train 'train', early stopping 'inner'). Predictions -> artifacts/predictions/e18/
for E14. No selection happens here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402


def main() -> None:
    import e14_blend_freeze_v2 as E14
    cols, _, set_name, src, _ = E14.tabular_choice(final=False)
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    text = paths.ARTIFACTS / "features"
    if any(c.startswith("fb_") for c in cols):
        fb = pd.read_parquet(text / "finbert_features.parquet")
        fb["fb_present"] = 1.0
        fr = fr.merge(fb, on="sample_id", how="left")
        fr["fb_present"] = fr.fb_present.fillna(0.0)
    if any(c.startswith("news_") for c in cols):
        fr = fr.merge(pd.read_parquet(text / "news_features.parquet"), on="sample_id", how="left")
    run = Run("e18_tabular_final", {"set": set_name, "from": src, "n_cols": len(cols), "horizons": list(HORIZONS_V2)},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pred_dir = paths.artifact("predictions", "e18", "x").parent
    res, _ = L.fit_eval(fr, {"FINAL": cols}, list(HORIZONS_V2), 8, pred_dir, log=lambda s: print(s, flush=True))
    run.finish("COMPLETED", f"E18 tabular final: set {set_name} from {src}, {len(cols)} columns, 23 horizons",
               {"by_horizon": res, "set": set_name, "source": src, "cols": cols})


if __name__ == "__main__":
    main()

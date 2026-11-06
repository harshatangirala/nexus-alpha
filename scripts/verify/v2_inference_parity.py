"""Train/serve parity of the v2 bundle: NexusPredictorV2.predict(mode="historical") must reproduce E14's own calibrated
probabilities for the same (ticker, as-of) rows.

    python scripts/verify/v2_inference_parity.py [--bundle <E14 run id>] [--n 24]

Samples rows from 2019-2022 (calib and test blocks) across tickers, re-predicts them through the inference path the
dashboard and the forward test use, and reports the largest absolute probability difference per horizon. Output:
reports/verification/v2_inference_parity.csv.
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

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.inference.pipeline_v2 import NexusPredictorV2, latest_v2_bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

H = list(HORIZONS_V2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    bdir = paths.ARTIFACTS / "bundles" / a.bundle if a.bundle else latest_v2_bundle()
    pdir = paths.ARTIFACTS / "predictions" / bdir.name
    rows = pd.read_parquet(pdir / "rows.parquet")
    Pc = np.load(pdir / "probs_calibrated.npy")
    rng = np.random.default_rng(5)
    pick = []
    for blk, n in (("calib", a.n // 3), ("test", a.n - a.n // 3)):
        idx = np.flatnonzero(rows.block.to_numpy() == blk)
        pick += list(rng.choice(idx, n, replace=False))
    P = NexusPredictorV2(bdir)
    out, t0 = [], time.time()
    for i in pick:
        r = rows.iloc[i]
        try:
            f = P.predict(r.ticker, asof=str(pd.Timestamp(r.asof_date).date()), mode="historical")
        except Exception as e:
            out.append({"ticker": r.ticker, "asof": str(r.asof_date.date()), "error": repr(e)[:160]})
            continue
        for j, h in enumerate(H):
            q = f["forecasts"][h]
            served = np.array([q["p_down"], q["p_side"], q["p_up"]])
            out.append({"ticker": r.ticker, "asof": str(r.asof_date.date()), "block": r.block, "horizon": h,
                        "max_abs_diff": float(np.abs(served - Pc[i, j]).max()),
                        "lean_served": float(served[2] - served[0]), "lean_e14": float(Pc[i, j, 2] - Pc[i, j, 0]),
                        "asof_served": f["asof_session"]})
    d = pd.DataFrame(out)
    (REPO / "reports" / "verification").mkdir(parents=True, exist_ok=True)
    d.to_csv(REPO / "reports" / "verification" / "v2_inference_parity.csv", index=False, float_format="%.6g")
    ok = d.dropna(subset=["max_abs_diff"]) if "max_abs_diff" in d else d.iloc[0:0]
    print(f"bundle {bdir.name}; {len(pick)} rows; errors {int(d.get('error', pd.Series(dtype=str)).notna().sum())}; "
          f"{time.time() - t0:.0f}s")
    if len(ok):
        print("max |diff| over all horizons:", round(ok.max_abs_diff.max(), 6), "| median:", round(ok.max_abs_diff.median(), 6))
        print(ok.groupby("horizon").max_abs_diff.max().reindex(H).round(6).to_string())
        print("lean sign agreement:", round(float((np.sign(ok.lean_served) == np.sign(ok.lean_e14)).mean()), 4))


if __name__ == "__main__":
    main()

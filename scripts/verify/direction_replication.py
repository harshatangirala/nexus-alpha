"""Did the frozen models' direction skill replicate out of sample? (audit 9, finding A-101)

Gate 3 of the risk engine trades a direction only where the model's up-vs-down AUROC on the 2017-18 validation block
had a 95% interval above 0.5 (the `direction_skill` flags frozen in each bundle). This script recomputes the same
statistic, the same way, on the 2020-2022 development test block:

* score P(UP) - P(DOWN), rows that moved (label UP or DOWN), month-block bootstrap, 500 resamples, seed 7
  (`evaluation.stats.block_bootstrap`, exactly as `scripts/experiments/e08_final_model.py` built the flags);
* v1.1: the frozen E08 bundle's saved test predictions (`e08__final_deployed_calibrated`, 6 horizons);
* v2: the frozen E14 bundle's saved test predictions (`probs_calibrated`, 23 horizons) with v2's own labels.

Inputs are the staged notebook copies (`artifacts/notebook`, `artifacts/samples/v2_frame.parquet`); both end in 2022 and
the script asserts it. Nothing is fitted or selected: the result can only close a gate (`trading/skill_evidence.py`).

    python scripts/verify/direction_replication.py      # writes reports/verification/direction_replication_2020_2022.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from nexus_alpha.evaluation.stats import block_bootstrap, month_blocks  # noqa: E402

NB = REPO / "artifacts" / "notebook"
OUT = REPO / "reports" / "verification" / "direction_replication_2020_2022.json"
H6 = ["1D", "2D", "5D", "10D", "20D", "1M"]
H23 = [f"{i}D" for i in range(1, 21)] + ["3M", "6M", "1Y"]
SEAL = pd.Timestamp("2023-01-01")


def replication(P: np.ndarray, Y: np.ndarray, dates: np.ndarray, horizons: list[str]) -> dict:
    out = {}
    for j, h in enumerate(horizons):
        y = Y[:, j]
        m = np.isfinite(y) & (y >= 0) & (y != 1)
        bs = block_bootstrap(lambda a, b: roc_auc_score(a, b), ((y[m] == 2).astype(int), P[m, j, 2] - P[m, j, 0]),
                             month_blocks(dates[m]), n_boot=500, seed=7)
        out[h] = {"test_up_vs_down_auroc": round(float(bs["estimate"]), 4),
                  "ci95": [round(float(bs["ci95"][0]), 4), round(float(bs["ci95"][1]), 4)],
                  "n_moves": int(m.sum()), "replicated": bool(bs["ci95"][0] > 0.5)}
    return out


def main() -> None:
    fr = pd.read_parquet(NB / "samples" / "s500_frame.parquet", columns=["sample_id", "asof_date"] + [f"y_{h}" for h in H6])
    assert pd.to_datetime(fr.asof_date).max() < SEAL, "input reaches the sealed period"
    rows = pd.read_parquet(NB / "predictions" / "rows_eval.parquet")
    te = rows.block.to_numpy() == "test"
    P1 = np.load(NB / "predictions" / "e08__final_deployed_calibrated.npy").astype(np.float64)[te]
    Y1 = fr.set_index("sample_id")[[f"y_{h}" for h in H6]].reindex(rows.sample_id[te]).to_numpy(np.float64)
    d1 = pd.to_datetime(rows.asof_date[te]).to_numpy()

    v2f = pd.read_parquet(REPO / "artifacts" / "samples" / "v2_frame.parquet",
                          columns=["ticker", "asof_date"] + [f"y_{h}" for h in H23])
    assert pd.to_datetime(v2f.asof_date).max() < SEAL, "input reaches the sealed period"
    v2f["sample_id"] = v2f.ticker + ":" + pd.to_datetime(v2f.asof_date).dt.strftime("%Y-%m-%d")
    r2 = pd.read_parquet(NB / "predictions" / "rows_v2.parquet")
    te2 = r2.block.to_numpy() == "test"
    P2 = np.load(NB / "predictions" / "e14__v2_probs_calibrated.npy").astype(np.float64)[te2]
    Y2 = v2f.set_index("sample_id")[[f"y_{h}" for h in H23]].reindex(r2.sample_id[te2]).to_numpy(np.float64)
    d2 = pd.to_datetime(r2.asof_date[te2]).to_numpy()

    res = {"statistic": "up-vs-down AUROC of P(UP) - P(DOWN) on rows that moved; month-block bootstrap, 500 resamples, "
                        "seed 7; 95% percentile interval; replicated = lower bound above 0.5 (gate 3's own rule)",
           "block": "test 2020-01-01 to 2022-12-01 (development test; the sealed 2023+ period is not read)",
           "v1.1": {"bundle": "E08 refit 2003-2018 (frozen v1.1)", "rows": int(te.sum()),
                    "horizons": replication(P1, Y1, d1, H6)},
           "v2": {"bundle": "E14 (frozen v2)", "rows": int(te2.sum()), "horizons": replication(P2, Y2, d2, H23)}}
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for m in ("v1.1", "v2"):
        rep = [h for h, v in res[m]["horizons"].items() if v["replicated"]]
        print(f"{m}: replicated at {rep or 'no horizon'}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

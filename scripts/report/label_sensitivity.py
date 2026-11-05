"""Class distributions under the registered threshold rule and three alternatives (sensitivity only; the registered rule
is the only one used for training and evaluation).

Rules: registered  tau = max(0.5 sigma sqrt h, 10 bps) | fixed  tau = 0.5% sqrt h | atr  tau = 0.25 (ATR14/close) sqrt h |
cost_only tau = 10 bps. Reported by horizon for the train and test blocks, and by volatility tercile (train cut points).
Output: reports/labels/class_distributions.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.targets import HORIZONS


def main() -> int:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    rows = []
    q = fr.loc[fr.block == "train", "sigma_ewma"].quantile([1 / 3, 2 / 3]).to_numpy()
    fr["vol_tercile"] = np.digitize(fr.sigma_ewma, q)
    for h, n in HORIZONS.items():
        r = fr[f"ret_{h}"]
        taus = {"registered": fr[f"tau_{h}"], "fixed_0.5pct_sqrt_h": pd.Series(0.005 * np.sqrt(n), index=fr.index),
                "atr_0.25_sqrt_h": 0.25 * fr.d_atr_pct * np.sqrt(n), "cost_only_10bps": pd.Series(0.001, index=fr.index)}
        for rule, tau in taus.items():
            y = np.select([r > tau, r < -tau], [2, 0], 1).astype(float)
            y[r.isna()] = np.nan
            for grp, mask in [("train", fr.block == "train"), ("test", fr.block == "test")] + \
                             [(f"train_vol_tercile_{k}", (fr.block == "train") & (fr.vol_tercile == k)) for k in range(3)]:
                yy = pd.Series(y[mask.to_numpy()]).dropna()
                rows.append({"horizon": h, "rule": rule, "group": grp, "n": len(yy), "down": (yy == 0).mean(),
                             "sideways": (yy == 1).mean(), "up": (yy == 2).mean()})
    out = pd.DataFrame(rows)
    p = paths.REPO_ROOT / "reports" / "labels"
    p.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(p / "class_distributions.csv", index=False)
    print(out[out.group.isin(["train", "test"])].pivot_table(index=["horizon", "rule"], columns="group",
                                                               values=["sideways", "up"]).round(3).to_string())
    print(out[(out.rule == "registered") & out.group.str.contains("tercile")].pivot_table(index="horizon", columns="group", values="sideways").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

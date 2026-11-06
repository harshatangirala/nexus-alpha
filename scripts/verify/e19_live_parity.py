"""Parity of the live E19 feature builder (Yahoo) with development values on pre-seal dates (2022-09-01..2022-12-01).

Compares every M2 feature and the deployed 20D market read built from each source. Yahoo keeps only 60 days of hourly
bars, so the two SPY hourly features cannot be rebuilt for 2022; the read is also compared with those two taken from
development, to isolate everything else. Output: reports/verification/e19_live_parity.csv and a printed summary.
"""
from __future__ import annotations

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
from nexus_alpha.macro import market_model as MM  # noqa: E402
from nexus_alpha.macro.context import live_date_features  # noqa: E402

HOURLY = ["mkt_spy_first_hour_ret", "mkt_spy_last_hour_ret"]


def main() -> None:
    D, sets = E19.date_frame()
    D = D.set_index("asof_date")
    dates = D.index[(D.index >= "2022-09-01") & (D.index <= "2022-12-01")]
    model = MM.MarketModel(MM.latest_bundle())
    cols = model.models["20D"]["cols"]
    days = (pd.Timestamp.now().normalize() - pd.Timestamp("2021-01-04")).days
    live = live_date_features(days)
    closes = MM.yahoo_closes(MM.universe_tickers(), "2022-03-01", "2022-12-02")
    u = MM.universe_context(closes.reindex(closes.index.intersection(live.index)))
    live = live.join(u[MM.UNIVERSE_COLS + ["n_tickers"]])
    dev = D.loc[dates, cols]
    lv = live.reindex(dates)
    rows = []
    for c in cols:
        a, b = dev[c].astype(float), lv[c].astype(float)
        both = a.notna() & b.notna()
        sd = D.loc[D.block == "train", c].std()
        rows.append({"feature": c, "n_both": int(both.sum()), "live_missing": int(b.isna().sum()),
                     "corr": float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 2 else np.nan,
                     "median_abs_diff_in_train_sd": float((a[both] - b[both]).abs().median() / sd) if both.any() else np.nan})
    rep = pd.DataFrame(rows).sort_values("median_abs_diff_in_train_sd", ascending=False)
    out = REPO / "reports" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "e19_live_parity.csv", index=False, float_format="%.5g")
    print(rep.head(12).to_string(index=False))
    print("universe tickers with Yahoo data on the last date:", int(u.n_tickers.reindex(dates).iloc[-1]),
          "of", len(MM.universe_tickers()))
    lv2 = lv.copy()
    lv2[HOURLY] = dev[HOURLY]
    for name, src in (("live as built", lv), ("live, hourly from dev", lv2)):
        pd_ = np.array([[model.predict(dev.loc[d])["20D"][k] for k in ("p_down", "p_up")] for d in dates])
        pl_ = np.array([[model.predict(src.loc[d])["20D"][k] for k in ("p_down", "p_up")] for d in dates])
        lean_d, lean_l = pd_[:, 1] - pd_[:, 0], pl_[:, 1] - pl_[:, 0]
        print(f"20D read, {name}: mean |diff| of (p_up - p_down) {np.abs(lean_d - lean_l).mean():.4f}; "
              f"sign agreement {np.mean(np.sign(lean_d) == np.sign(lean_l)):.3f}; corr {np.corrcoef(lean_d, lean_l)[0, 1]:.3f}"
              f"; n {len(dates)}")


if __name__ == "__main__":
    main()

"""Build the v2 sample frame: the s500 decision rows + v2 labels + market, macro and stock-relative features.

    python scripts/data/build_v2_frame.py [--rebuild-mkt]

Rows: the same 236,520 (ticker, as-of) decisions and blocks as v1 (artifacts/samples/s500_frame.parquet).
Adds: v2 labels (1D-20D, 3M, 6M, 1Y; per-horizon purge), `rel_*` (per stock), `mkt_*` (per date), `gmd_*` (per date,
publication-lagged), `fred_*` (per date, real-time vintages; only if artifacts/macro/alfred_vintages.parquet exists).
Writes artifacts/samples/v2_frame.parquet and reports/data/v2_frame_coverage.csv (non-missing share per feature/block).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.data import etf_bars as EB  # noqa: E402
from nexus_alpha.macro import context as CX, features as MF, fred as FR, gmd as G  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2, purge_per_horizon  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-mkt", action="store_true")
    a = ap.parse_args()
    fr = pq.read_table(paths.ARTIFACTS / "samples" / "s500_frame.parquet").to_pandas()
    fr = fr[[c for c in fr.columns if not c.startswith(("y_", "z_", "end_", "ret_"))]]   # v1 labels replaced by v2 labels (A-29: incl. ret_)
    fr["asof_date"] = pd.to_datetime(fr.asof_date)
    etf = EB.closes([t for t in CX.ETF_LIST if t in EB.available()])
    parts = []
    from nexus_alpha.targets_v2 import make_labels_v2
    for i, (t, g) in enumerate(fr.groupby("ticker", sort=True)):
        d = B.read_daily(t)
        lab = make_labels_v2(d).rename(columns={"date": "asof_date"})
        rel = MF.stock_relative(d[["date", "close"]], etf).reset_index().rename(columns={"date": "asof_date"})
        x = g.merge(lab, on="asof_date", how="left").merge(rel, on="asof_date", how="left")
        parts.append(x)
        if (i + 1) % 100 == 0:
            print(f"{i + 1} stocks", flush=True)
    v2 = pd.concat(parts, ignore_index=True)
    mkt = CX.dev_date_features(rebuild=a.rebuild_mkt)
    v2 = v2.merge(mkt.reset_index().rename(columns={"date": "asof_date"}), on="asof_date", how="left")
    dates = pd.DatetimeIndex(sorted(v2.asof_date.unique()))
    gm = G.asof_features(dates).reset_index().rename(columns={"index": "asof_date"})
    v2 = v2.merge(gm, on="asof_date", how="left")
    if (paths.ARTIFACTS / FR.STORE).exists():
        fx = FR.asof_features(dates).reset_index().rename(columns={"index": "asof_date"})
        v2 = v2.merge(fx, on="asof_date", how="left")
        print("FRED vintages joined:", len([c for c in fx.columns if c.startswith("fred_")]), "features")
    else:
        print("FRED vintages not available (FRED_API_KEY not set): fred_* block skipped")
    v2 = purge_per_horizon(v2)
    print("per-horizon purge (rows NaN-ed):", {k: v for k, v in v2.attrs["purged"].items() if k.split(":")[1] in ("20D", "3M", "6M", "1Y")})
    v2.to_parquet(paths.artifact("samples", "v2_frame.parquet"), index=False)
    feat = [c for c in v2.columns if c.startswith(("rel_", "mkt_", "gmd_", "fred_"))]
    cov = v2.groupby("block")[feat].apply(lambda x: x.notna().mean()).T.round(3)
    rep = REPO / "reports" / "data"
    rep.mkdir(parents=True, exist_ok=True)
    cov.to_csv(rep / "v2_frame_coverage.csv")
    lab = v2.groupby("block")[[f"y_{h}" for h in HORIZONS_V2]].count().T
    lab.to_csv(rep / "v2_label_counts.csv")
    print(v2.shape, "features added:", len(feat))
    print(lab.loc[["y_1D", "y_5D", "y_20D", "y_3M", "y_6M", "y_1Y"]])


if __name__ == "__main__":
    main()

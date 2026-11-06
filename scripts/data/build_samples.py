"""Build the NEXUS sample frame on TriFrame's s500 chart sets (so stored chart embeddings can be reused).

One row per s500 sample (ticker, as-of session t): six-horizon labels, multi-timeframe structured indicator features,
TriFrame's 33 visible-window features (through TriFrame's sealed ImageStore), market context, the decision timestamp
(09:00 New York on session t+1) and the purged primary-split block. Every bar read is sealed; rows whose labels would need
sealed bars carry NaN labels for those horizons.

Output: artifacts/samples/s500_frame.parquet (+ data_manifests/s500_frame_summary.json)
Run: .venv/Scripts/python -I scripts/data/build_samples.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import pandas as pd

from nexus_alpha import paths, seal, sources_bridge
from nexus_alpha.cv import store as ST
from nexus_alpha.data import bars as B
from nexus_alpha.data.calendar import decision_time_utc
from nexus_alpha.indicators.core import STRUCTURED_ALL, structured_features
from nexus_alpha.indicators.market import MARKET_FEATURES, market_context
from nexus_alpha.splits import assign_blocks
from nexus_alpha.targets import HORIZONS, make_labels

REPO = Path(__file__).resolve().parents[2]


def per_ticker(args) -> pd.DataFrame:
    ticker, asof = args
    from nexus_alpha.cv.render import rth_hourly
    from nexus_alpha.data.calendar import calendar
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    d = B.read_daily(ticker)
    h = rth_hourly(B.read_hourly(ticker), early)
    lab = make_labels(d).set_index("date")
    asof = pd.DatetimeIndex(sorted(asof))
    f = structured_features(d, h, asof)
    L = lab.reindex(asof)
    adv = (d.close * d.volume).rolling(20).mean()
    out = pd.concat([f, L], axis=1)
    out["adv20_usd"] = adv.set_axis(d.date).reindex(asof).values
    out["close"] = d.set_index("date").close.reindex(asof).values
    out["n_daily_bars_before"] = d.date.searchsorted(asof, side="right")
    out.index.name = "asof_date"
    out = out.reset_index()
    out.insert(0, "ticker", ticker)
    return out


def main() -> int:
    t0 = time.time()
    idx = ST.sample_index()
    groups = idx.groupby("ticker").asof_date.apply(list)
    parts = []
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        for i, p in enumerate(ex.map(per_ticker, list(groups.items()), chunksize=4)):
            parts.append(p)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(groups)} tickers, {time.time() - t0:.0f} s", flush=True)
    fr = pd.concat(parts, ignore_index=True)
    fr["sample_id"] = fr.ticker + ":" + fr.asof_date.dt.strftime("%Y-%m-%d")
    fr = fr.merge(idx[["sample_id", "row"]], on="sample_id", how="left", validate="one_to_one")
    assert fr.row.notna().all()

    # TriFrame's 33 deployable visible-window features through its sealed loader
    st = sources_bridge.triframe_image_store(who="nexus_build_samples")
    sealed_idx = st.index()[["sample_id", "ticker"]]
    X = st.features(sorted(fr.ticker.unique()))
    fv = sources_bridge.triframe_feat_visible()
    xcols = ["x_" + c for c in fv.DEPLOY]
    fr = fr.merge(X[["sample_id"] + xcols], on="sample_id", how="left", validate="one_to_one")
    fr["in_triframe_sealed_index"] = fr.sample_id.isin(set(sealed_idx.sample_id))

    mk = market_context()
    fr = fr.merge(mk[MARKET_FEATURES].reset_index().rename(columns={"date": "asof_date"}), on="asof_date", how="left")
    fr["decision_utc"] = decision_time_utc(fr.asof_date).values
    fr["block"] = assign_blocks(fr.asof_date).values
    seal.assert_dev(fr.asof_date, what="sample frame as-of dates")
    for k in HORIZONS:
        seal.assert_dev(fr.loc[fr[f"y_{k}"].notna(), "asof_date"], fr.loc[fr[f"y_{k}"].notna(), f"end_{k}"],
                        what=f"labelled rows {k}")
    out = paths.ARTIFACTS / "samples"
    out.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(out / "s500_frame.parquet", index=False)

    summ = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": len(fr),
            "tickers": int(fr.ticker.nunique()), "asof_min": str(fr.asof_date.min().date()),
            "asof_max": str(fr.asof_date.max().date()), "rows_by_block": fr.block.fillna("purged_or_pre2003").value_counts().to_dict(),
            "x_features_missing_share": float(fr[xcols].isna().any(axis=1).mean()),
            "structured_missing_share_by_group": {g: float(fr[[c for c in STRUCTURED_ALL if c.startswith(g)]].isna().any(axis=1).mean())
                                                  for g in ("d_", "w_", "h_")},
            "in_triframe_sealed_index_share": float(fr.in_triframe_sealed_index.mean()), "class_distribution": {}}
    for k in HORIZONS:
        for b in ("train", "valid", "calib", "test"):
            y = fr.loc[fr.block == b, f"y_{k}"].dropna()
            summ["class_distribution"].setdefault(k, {})[b] = {"n": int(len(y)),
                                                                "down": round(float((y == 0).mean()), 4),
                                                                "sideways": round(float((y == 1).mean()), 4),
                                                                "up": round(float((y == 2).mean()), 4)}
    summ["elapsed_s"] = round(time.time() - t0, 1)
    (REPO / "data_manifests" / "s500_frame_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "class_distribution"}, indent=1))
    for k in HORIZONS:
        print(k, summ["class_distribution"][k]["train"], summ["class_distribution"][k]["test"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

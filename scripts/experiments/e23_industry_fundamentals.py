"""E23 industry-neutral fundamentals (pre-registered in docs/V3_RESEARCH.md before any E22 result).

As E22 (rows 2009-2022, blocks, models, 9 horizons, two endpoints, keep rule), with the FUND ranks taken within the
SIC major group (>= 8 stocks with the ratio that day), else the SIC division, else the universe (fi_* columns of the
fundamentals panel). Sets: B = F0; B + FUND_IND; placebo = FUND_IND permuted across tickers within each as-of date.
B_FUND_IND vs E22's B_FUND is reported descriptively from E22's saved predictions when they exist.
"""
from __future__ import annotations

import math
import sys
import time
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
import e21_ranker as E21  # noqa: E402
from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.fundamentals.ratios import RATIOS  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEED = 11


def main() -> None:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"]) & (fr.asof_date >= "2009-01-02")].reset_index(drop=True)
    P = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "panel.parquet")
    fcols = [f"fi_{c}" for c in RATIOS + ["days_since_filing"]] + ["fund_present"]
    P = P[["ticker", "date"] + fcols].rename(columns={"date": "asof_date"})
    P["asof_date"] = pd.to_datetime(P.asof_date).astype("datetime64[ns]")     # parquet may round-trip as [us]
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    n0 = len(fr)
    fr = fr.merge(P, on=["ticker", "asof_date"], how="left", validate="many_to_one")
    assert len(fr) == n0
    fr["fund_present"] = fr.fund_present.fillna(0.0)
    rng = np.random.default_rng(SEED)
    pl = fr[["asof_date"] + fcols].copy()
    for _, idx in fr.groupby("asof_date").indices.items():
        pl.loc[idx, fcols] = fr.loc[rng.permutation(idx), fcols].to_numpy()
    for c in fcols:
        fr[f"PL_{c}"] = pl[c].to_numpy()
    f0 = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    sets = {"B": f0, "B_FUND_IND": f0 + fcols, "B_FUND_IND_placebo": f0 + [f"PL_{c}" for c in fcols]}
    horizons = list(L.REP_HORIZONS)
    cov = {b: float(fr.loc[fr.block == b, "fund_present"].mean()) for b in ("train", "inner", "valid", "test")}
    print("rows", len(fr), "| fundamentals present share by block", {k: round(v, 3) for k, v in cov.items()}, flush=True)
    run = Run("e23_industry_fundamentals", {"sets": {k: len(v) for k, v in sets.items()}, "horizons": horizons, "rows_from": "2009-01-02",
                                   "fund_cols": fcols, "lgb": L.E07_PARAMS, "seed": SEED},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                      "fund_panel": paths.ARTIFACTS / "fundamentals" / "panel.parquet"})
    pred_dir = paths.artifact("predictions", "e23", "x").parent
    res, rows_ll = L.fit_eval(fr, sets, horizons, 8, pred_dir, log=lambda s: print(s, flush=True))
    blk = fr.block.to_numpy()
    eligible = {h: fr.asof_date[(blk == "valid") & fr[f"y_{h}"].notna()].nunique() / HORIZONS_V2[h] >= E19.MIN_WINDOWS
                for h in horizons}
    blen = lambda hs: max(max(1, math.ceil(HORIZONS_V2[h] / 21)) for h in hs)
    rng = np.random.default_rng(SEED)
    comps = [("B_FUND_IND", "B"), ("B_FUND_IND_placebo", "B")]
    ll_gain, ic_gain, ics = {}, {}, {}
    for h in horizons:
        lab = fr[f"y_{h}"].notna().to_numpy()
        for split in ("valid", "test"):
            m = lab & (blk == split)
            for name in sets:
                for mdl in ("lgbm", "logistic"):
                    Pp = np.load(pred_dir / f"{h}_{name}_{mdl}_{split}.npy")
                    ic = E21.daily_ic(fr.asof_date[m].to_numpy(), Pp[:, 2] - Pp[:, 0], fr[f"ret_{h}"][m].to_numpy())
                    ic.index = pd.DatetimeIndex(ic.index)
                    ics[(h, name, mdl, split)] = ic
    for mdl in ("lgbm", "logistic"):
        for new, old in comps:
            for split in ("valid", "test"):
                for g, hs in L.GROUPS.items():
                    hs = [h for h in hs if h in horizons and (split == "test" or eligible[h])]
                    if not hs:
                        continue
                    per = [(rows_ll[(h, old, mdl, split)][1], rows_ll[(h, old, mdl, split)][0] - rows_ll[(h, new, mdl, split)][0])
                           for h in hs]
                    pt, ci = E19.block_boot(per, blen(hs), rng)
                    ll_gain[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": pt, "ci95": ci, "horizons": hs}
                    diffs = [(ics[(h, new, mdl, split)] - ics[(h, old, mdl, split)]).dropna() for h in hs]
                    pt, ci = E21.block_ci(diffs, blen(hs), rng)
                    ic_gain[f"{mdl}|{new}_vs_{old}|{split}|{g}"] = {"gain": pt, "ci95": ci, "horizons": hs}
    keep = {}
    for mdl in ("lgbm", "logistic"):
        for ep, gains in (("logloss", ll_gain), ("rank_ic", ic_gain)):
            k_new = L.keep_decision(gains, mdl, "B_FUND_IND", "B")
            k_pl = L.keep_decision(gains, mdl, "B_FUND_IND_placebo", "B")
            keep[f"{mdl}|{ep}"] = {"B_FUND_IND_passes": k_new, "placebo_passes": k_pl, "kept": bool(k_new and not k_pl)}
    kept = any(v["kept"] for v in keep.values())
    abs_ic = {f"{h}|{name}|{mdl}|{split}": float(ic.mean()) for (h, name, mdl, split), ic in ics.items()}
    # descriptive: B_FUND_IND vs E22's B_FUND on the same rows (validation mean log loss and mean daily IC)
    vs_e22 = {}
    e22 = paths.ARTIFACTS / "predictions" / "e22"
    for h in horizons:
        lab = fr[f"y_{h}"].notna().to_numpy()
        m = lab & (blk == "valid")
        for mdl in ("lgbm", "logistic"):
            f = e22 / f"{h}_B_FUND_{mdl}_valid.npy"
            if not f.exists():
                continue
            pa, pb = np.load(f), np.load(pred_dir / f"{h}_B_FUND_IND_{mdl}_valid.npy")
            y = fr[f"y_{h}"][m].astype(int).to_numpy()
            ll = lambda P_: float(-np.mean(np.log(np.clip(P_[np.arange(len(y)), y], 1e-12, 1))))
            ic_a = E21.daily_ic(fr.asof_date[m].to_numpy(), pa[:, 2] - pa[:, 0], fr[f"ret_{h}"][m].to_numpy()).mean()
            vs_e22[f"{h}|{mdl}"] = {"logloss_gain_vs_B_FUND": ll(pa) - ll(pb),
                                    "ic_B_FUND_IND": float(ics[(h, "B_FUND_IND", mdl, "valid")].mean()), "ic_B_FUND": float(ic_a)}
    summ = f"E23 industry-neutral fundamentals: kept={kept}; " + "; ".join(f"{k}: {'keep' if v['kept'] else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", summ, {"by_horizon": res, "logloss_gain": ll_gain, "rank_ic_gain": ic_gain, "keep": keep,
                                   "kept": kept, "mean_daily_ic": abs_ic, "fund_present_share": cov, "sets": sets, "vs_e22_descriptive": vs_e22,
                                   "elapsed_s": round(time.time() - t0, 1)})
    print(summ)
    for name, gains in (("LOGLOSS", ll_gain), ("RANK IC", ic_gain)):
        for k, v in gains.items():
            if "placebo" not in k:
                print(name, k, round(v["gain"], 5), [round(x, 5) for x in v["ci95"]])


if __name__ == "__main__":
    main()

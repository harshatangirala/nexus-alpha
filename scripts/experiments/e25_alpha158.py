"""E25 Qlib Alpha158 factor block (pre-registered in docs/V3_RESEARCH.md before the features were built).

Rows: the v2 frame's train (2003-2015), inner (2016), valid (2017-18) and test (2020-22) blocks. Sets: B = F0 (80 v1.1
numeric features); B + A158 (157 Alpha158 features, nexus_alpha.indicators.alpha158); placebo = A158 permuted across
tickers within each as-of date; A158 alone (descriptive). Models, 9 horizons, two endpoints and the keep rule as E22/E23.
Also reported: B + A158 against its own placebo (the sharper check, not part of the decision) and Qlib-style IC, ICIR,
rank IC and rank ICIR for every set, family, horizon and split.
"""
from __future__ import annotations

import hashlib
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
from nexus_alpha.indicators import alpha158 as A  # noqa: E402
from nexus_alpha.inference.pipeline import Bundle  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEED = 11
THREADS = 6


def prereg_sha256() -> str:
    """SHA-256 of the E25 pre-registration text as it stands when the run starts (recorded in the manifest)."""
    s = (REPO / "docs" / "V3_RESEARCH.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    start = s.index("## E25 Qlib Alpha158 factor block (pre-registered")
    nxt = s.find("\n## ", start + 10)
    return hashlib.sha256(s[start: nxt if nxt > 0 else len(s)].encode("utf-8")).hexdigest()


def daily_pearson_ic(dates, score, ret) -> pd.Series:
    d = pd.DataFrame({"d": dates, "s": score, "r": ret}).dropna()
    g = d.groupby("d")
    return g.apply(lambda x: x.s.corr(x.r) if len(x) >= E21.MIN_N else np.nan).dropna()


def ir(s: pd.Series) -> float:
    sd = float(s.std())
    return float(s.mean()) / sd if sd > 0 else float("nan")


def main() -> None:
    try:   # the nightly forward test keeps precedence on this machine
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    t0 = time.time()
    pre = prereg_sha256()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "test"])].reset_index(drop=True)
    P = pd.read_parquet(paths.ARTIFACTS / "features" / "alpha158.parquet")
    P["asof_date"] = pd.to_datetime(P.asof_date).astype("datetime64[ns]")
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    n0 = len(fr)
    fr = fr.merge(P, on=["ticker", "asof_date"], how="left", validate="one_to_one")
    assert len(fr) == n0 and fr[A.COLUMNS[0]].notna().mean() > 0.999
    acols = list(A.COLUMNS)
    rng = np.random.default_rng(SEED)
    pl = fr[acols].to_numpy().copy()
    src = fr[acols].to_numpy()
    for _, idx in fr.groupby("asof_date").indices.items():
        pl[idx] = src[rng.permutation(idx)]
    plc = [f"PL_{c}" for c in acols]
    fr = pd.concat([fr, pd.DataFrame(pl, columns=plc, index=fr.index)], axis=1)
    f0 = list(Bundle(SH.verify_frozen("v1.1")[0]).pre.num_cols)
    sets = {"B": f0, "B_A158": f0 + acols, "B_A158_placebo": f0 + plc, "A158": acols}
    horizons = list(L.REP_HORIZONS)
    print("rows", len(fr), "| sets", {k: len(v) for k, v in sets.items()}, "| prereg", pre[:12], flush=True)
    run = Run("e25_alpha158", {"sets": {k: len(v) for k, v in sets.items()}, "horizons": horizons,
                               "a158_cols": acols, "lgb": L.E07_PARAMS, "seed": SEED, "threads": THREADS,
                               "prereg_sha256": pre},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                      "alpha158": paths.ARTIFACTS / "features" / "alpha158.parquet"})
    pred_dir = paths.artifact("predictions", "e25", "x").parent
    res, rows_ll = L.fit_eval(fr, sets, horizons, THREADS, pred_dir, log=lambda s: print(s, flush=True))
    blk = fr.block.to_numpy()
    eligible = {h: fr.asof_date[(blk == "valid") & fr[f"y_{h}"].notna()].nunique() / HORIZONS_V2[h] >= E19.MIN_WINDOWS
                for h in horizons}
    blen = lambda hs: max(max(1, math.ceil(HORIZONS_V2[h] / 21)) for h in hs)
    rng = np.random.default_rng(SEED)
    comps = [("B_A158", "B"), ("B_A158_placebo", "B"), ("B_A158", "B_A158_placebo"), ("A158", "B")]
    ll_gain, ic_gain, ics, qlib_style = {}, {}, {}, {}
    for h in horizons:
        lab = fr[f"y_{h}"].notna().to_numpy()
        for split in ("valid", "test"):
            m = lab & (blk == split)
            dates, ret = fr.asof_date[m].to_numpy(), fr[f"ret_{h}"][m].to_numpy()
            for name in sets:
                for mdl in ("lgbm", "logistic"):
                    Pp = np.load(pred_dir / f"{h}_{name}_{mdl}_{split}.npy")
                    score = Pp[:, 2] - Pp[:, 0]
                    ric = E21.daily_ic(dates, score, ret)
                    ric.index = pd.DatetimeIndex(ric.index)
                    ics[(h, name, mdl, split)] = ric
                    pic = daily_pearson_ic(dates, score, ret)
                    qlib_style[f"{h}|{name}|{mdl}|{split}"] = {"IC": float(pic.mean()), "ICIR": ir(pic),
                                                               "RankIC": float(ric.mean()), "RankICIR": ir(ric),
                                                               "days": int(len(ric))}
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
            k_new = L.keep_decision(gains, mdl, "B_A158", "B")
            k_pl = L.keep_decision(gains, mdl, "B_A158_placebo", "B")
            passing = [g for g in L.GROUPS if (x := gains.get(f"{mdl}|B_A158_vs_B|valid|{g}")) and x["ci95"][0] > 0]
            sharp = {g: gains.get(f"{mdl}|B_A158_vs_B_A158_placebo|valid|{g}") for g in passing}
            beats = any(v is not None and v["ci95"][0] > 0 for v in sharp.values()) if passing else None
            kept_here = bool(k_new and not k_pl)
            keep[f"{mdl}|{ep}"] = {"B_A158_passes": k_new, "placebo_passes": k_pl, "kept": kept_here,
                                   "passing_groups": passing, "beats_own_placebo_in_passing_group": beats,
                                   "weak_keep": bool(kept_here and not beats)}
    kept = any(v["kept"] for v in keep.values())
    abs_ic = {f"{h}|{name}|{mdl}|{split}": float(ic.mean()) for (h, name, mdl, split), ic in ics.items()}
    summ = f"E25 Alpha158: kept={kept}; " + "; ".join(
        f"{k}: {'keep' + (' (weak)' if v['weak_keep'] else '') if v['kept'] else 'drop'}" for k, v in keep.items())
    run.finish("COMPLETED", summ, {"by_horizon": res, "logloss_gain": ll_gain, "rank_ic_gain": ic_gain, "keep": keep,
                                   "kept": kept, "mean_daily_rank_ic": abs_ic, "qlib_style": qlib_style, "sets": sets,
                                   "prereg_sha256": pre, "elapsed_s": round(time.time() - t0, 1)})
    print(summ)
    for name, gains in (("LOGLOSS", ll_gain), ("RANK IC", ic_gain)):
        for k, v in gains.items():
            if "|valid|" in k or "|test|" in k:
                print(name, k, round(v["gain"], 5), [round(x, 5) for x in v["ci95"]])


if __name__ == "__main__":
    main()

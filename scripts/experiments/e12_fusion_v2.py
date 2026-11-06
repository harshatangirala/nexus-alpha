"""E12 v2 fusion (pre-registered in docs/V2_ARCHITECTURE.md section 4).

The gated fusion network of v1.1 (E06 settings: lr 3e-4, dropout 0.3, cosine schedule with warm-up, patience 6, 5 seeds),
with one classification head and three quantile heads per v2 horizon (23), and one extra modality, `ctx`: the columns
E11 carried forward beyond F0 (stock-relative, market, macro), standardised on training rows. Image, numeric, news, SEC and
rule modalities are built exactly as for v1.1 (multimodal.dataset.build). No new tuning.
Training rows: 'train'; early stopping: 'inner'. Saves per-seed and ensemble probabilities for valid, calib and test,
and the quantiles, for E14.
"""
from __future__ import annotations

import argparse
import json
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

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.multimodal.dataset import build  # noqa: E402
from nexus_alpha.multimodal.fusion import FusionConfig  # noqa: E402
from nexus_alpha.multimodal.trainer import ModalityData, predict, train  # noqa: E402
from nexus_alpha.targets import HORIZONS as V1H  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEEDS = (7, 11, 23, 42, 101)
KW = {"lr": 3e-4, "dropout": 0.3, "d_model": 64}          # E06 selection (run 20261106T005012_e06_optimization_fa42d8)


def load_frame() -> pd.DataFrame:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet")
    fr = fr[fr.block.isin(["train", "inner", "valid", "calib", "test"])].reset_index(drop=True)
    for name in ("news_features", "sec_features", "rule_features"):
        p = paths.ARTIFACTS / "features" / f"{name}.parquet"
        if p.exists():
            t = pd.read_parquet(p)
            if "sample_id" in t.columns:
                t = t.set_index("sample_id")
            fr = fr.merge(t, left_on="sample_id", right_index=True, how="left")
    for h in V1H:                      # v1 label columns are only placeholders for build(); v2 labels replace them
        for k in ("y", "z"):
            if f"{k}_{h}" not in fr:
                fr[f"{k}_{h}"] = np.nan
    return fr


def ctx_block(fr: pd.DataFrame, cols: list[str], fit_rows: np.ndarray, return_stats: bool = False):
    num = [c for c in cols if c != "rel_sector_id"]
    X = fr[num].to_numpy(np.float64)
    mu = np.nan_to_num(np.nanmean(X[fit_rows], 0))
    sd = np.nanstd(X[fit_rows], 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Z = np.clip(np.nan_to_num((X - mu) / sd, nan=0.0), -8, 8)
    if "rel_sector_id" in cols:
        sid = fr.rel_sector_id.to_numpy()
        Z = np.concatenate([Z, np.stack([(sid == k).astype(float) for k in range(9)], 1)], 1)
    return (Z.astype(np.float32), mu, sd) if return_stats else Z.astype(np.float32)


def auroc_ud(y, P):
    return L.auroc_ud(y, P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args()
    t0 = time.time()
    carried, e11 = L.carried_feature_set(REPO / "experiments")
    bundle_meta = json.loads((SH.verify_frozen("v1.1")[0] / "bundle_meta.json").read_text(encoding="utf-8"))
    mods = tuple(bundle_meta.get("modalities", ("img_W", "img_D", "img_H", "num", "news", "sec", "rules")))
    fr = load_frame()
    blk = fr.block.to_numpy()
    tr_mask = blk == "train"
    d, pre = build(fr, tr_mask, modalities=mods)
    ctx_cols = [c for c in carried if c not in pre.num_cols]
    names = list(d.names)
    x = dict(d.x)
    mask = d.mask
    if ctx_cols:
        x["ctx"] = ctx_block(fr, ctx_cols, np.flatnonzero(tr_mask))
        names.append("ctx")
        mask = np.concatenate([mask, np.ones((len(fr), 1), np.float32)], 1)
    H = list(HORIZONS_V2)
    y = np.stack([fr[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in H], 1)
    z = np.stack([fr[f"z_{h}"].to_numpy(np.float32) for h in H], 1)
    data = ModalityData(x, mask, y, z, names)
    run = Run("e12_fusion_v2", {"modalities": names, "ctx_cols": len(ctx_cols), "carried_from": e11, "horizons": H,
                                "kw": KW, "seeds": a.seeds, "epochs_max": a.epochs, "patience": 6, "schedule": "cosine"},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pdir = paths.artifact("predictions", run.run_id, "x").parent
    idx = {b: np.flatnonzero(blk == b) for b in ("train", "inner", "valid", "calib", "test")}
    Ps, Qs, per_seed = {b: [] for b in ("valid", "calib", "test")}, {b: [] for b in ("valid", "calib", "test")}, {}
    for s in a.seeds:
        cfg = FusionConfig(modality_dims={k: v.shape[1] for k, v in x.items()}, n_horizons=len(H), variant="gated",
                           d_model=KW["d_model"], hidden=2 * KW["d_model"], dropout=KW["dropout"], modality_dropout=0.15)
        model, hist = train(cfg, data.subset(idx["train"]), data.subset(idx["inner"]),
                            paths.ARTIFACTS / "models" / run.run_id / f"seed{s}", epochs=a.epochs, batch=2048,
                            lr=KW["lr"], weight_decay=1e-4, patience=6, seed=s, log=lambda m: None, schedule="cosine",
                            restore_best=True)
        best = int(hist[int(np.argmin([h["inner_ce"] for h in hist]))]["epoch"]) if hist else None
        per_seed[s] = {"best_epoch": best, "history": hist}
        for b in ("valid", "calib", "test"):
            P, Q, _ = predict(model, data.subset(idx[b]))
            Ps[b].append(P)
            Qs[b].append(Q)
            np.save(pdir / f"seed{s}_{b}.npy", P.astype(np.float32))
        print(f"seed {s}: best epoch {best} ({time.time() - t0:.0f}s)", flush=True)
    res = {}
    for b in ("valid", "calib", "test"):
        P = np.mean(Ps[b], 0)
        Q = np.sort(np.mean(Qs[b], 0), axis=-1)
        np.save(pdir / f"ens_{b}.npy", P.astype(np.float32))
        np.save(pdir / f"ens_{b}_quantiles.npy", Q.astype(np.float32))
        Y = y[idx[b]]
        for j, h in enumerate(H):
            m = Y[:, j] >= 0
            if m.sum() == 0:
                continue
            l = L.ll_rows(Y[m, j], P[m, j])
            res.setdefault(h, {})[f"{b}_ll"] = float(l.mean())
            res[h][f"{b}_auroc_ud"] = L.auroc_ud(Y[m, j], P[m, j])
            res[h][f"{b}_n"] = int(m.sum())
    np.save(pdir / "row_index.npy", np.concatenate([idx["valid"], idx["calib"], idx["test"]]))
    summ = (f"E12 fusion v2, {len(a.seeds)} seeds, {len(H)} horizons, ctx {len(ctx_cols)} cols; mean valid LL "
            f"{np.mean([r['valid_ll'] for r in res.values()]):.5f}")
    run.finish("COMPLETED", summ, {"by_horizon": res, "per_seed": per_seed, "pred_dir": str(pdir),
                                   "elapsed_s": round(time.time() - t0, 1)})
    print(summ)


if __name__ == "__main__":
    main()

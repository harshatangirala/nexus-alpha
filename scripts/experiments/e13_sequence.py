"""E13 sequence branch (pre-registered in docs/V2_ARCHITECTURE.md section 4): causal TCN and GRU, 63 daily bars of the
stock and SPY, every v2 horizon. E06-style settings (lr 3e-4, warm-up then cosine, weight decay 1e-4, patience 6),
3 seeds per architecture (stated before running). Train 'train', early-stop 'inner'; predictions saved for E14.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, seqmodels as SM  # noqa: E402
from nexus_alpha.data import bars as B, etf_bars as EB  # noqa: E402
from nexus_alpha.evaluation import ladder as L  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.targets_v2 import HORIZONS_V2  # noqa: E402

SEEDS = (7, 11, 23)


def build_X(fr: pd.DataFrame) -> np.ndarray:
    spy = EB.read_daily("SPY")
    X = np.full((len(fr), SM.WINDOW, len(SM.CHANNELS)), np.nan, np.float32)
    for t, g in fr.groupby("ticker"):
        ch = SM.channels(B.read_daily(t), spy)
        X[g.index.to_numpy()] = SM.windows(ch, pd.DatetimeIndex(g.asof_date))
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet",
                         columns=["ticker", "asof_date", "block"] + [f"y_{h}" for h in HORIZONS_V2])
    fr = fr[fr.block.isin(["train", "inner", "valid", "calib", "test"])].reset_index(drop=True)
    X = build_X(fr)
    blk = fr.block.to_numpy()
    tr = blk == "train"
    mu = np.nanmean(X[tr].reshape(-1, X.shape[2]), 0)
    sd = np.nanstd(X[tr].reshape(-1, X.shape[2]), 0)
    X = np.clip(np.nan_to_num((X - mu) / sd, nan=0.0), -8, 8).astype(np.float32)
    H = list(HORIZONS_V2)
    Y = np.stack([fr[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in H], 1)
    print(f"sequences built {X.shape} ({time.time() - t0:.0f}s)", flush=True)
    run = Run("e13_sequence", {"window": SM.WINDOW, "channels": SM.CHANNELS, "archs": ["tcn", "gru"], "seeds": SEEDS,
                               "lr": 3e-4, "patience": 6, "epochs_max": a.epochs, "horizons": H,
                               "channel_mean": mu.tolist(), "channel_sd": sd.tolist()},
              inputs={"v2_frame": paths.ARTIFACTS / "samples" / "v2_frame.parquet"})
    pdir = paths.artifact("predictions", run.run_id, "x").parent
    idx = {b: np.flatnonzero(blk == b) for b in ("train", "inner", "valid", "calib", "test")}
    res, per = {}, {}
    for arch in ("tcn", "gru"):
        preds = {b: [] for b in ("valid", "calib", "test")}
        for s in SEEDS:
            torch.manual_seed(s)
            model = SM.TCN(X.shape[2], len(H)) if arch == "tcn" else SM.GRUNet(X.shape[2], len(H))
            model, hist = SM.fit(model, X[idx["train"]], Y[idx["train"]], X[idx["inner"]], Y[idx["inner"]],
                                 epochs=a.epochs, seed=s, log=lambda m: None)
            per[f"{arch}_seed{s}"] = {"best_epoch": int(np.argmin([h["inner_ce"] for h in hist])), "history": hist}
            torch.save(model.state_dict(), paths.artifact("models", run.run_id, f"{arch}_seed{s}.pt"))
            for b in preds:
                preds[b].append(SM.predict(model, X[idx[b]]))
            print(f"{arch} seed {s}: best epoch {per[f'{arch}_seed{s}']['best_epoch']} ({time.time() - t0:.0f}s)", flush=True)
        for b in preds:
            P = np.mean(preds[b], 0)
            np.save(pdir / f"{arch}_ens_{b}.npy", P.astype(np.float32))
            Yb = Y[idx[b]]
            for j, h in enumerate(H):
                m = Yb[:, j] >= 0
                if m.sum():
                    res.setdefault(arch, {}).setdefault(h, {})[f"{b}_ll"] = float(L.ll_rows(Yb[m, j], P[m, j]).mean())
                    res[arch][h][f"{b}_auroc_ud"] = L.auroc_ud(Yb[m, j], P[m, j])
    summ = "; ".join(f"{a_}: mean valid LL {np.mean([r['valid_ll'] for r in res[a_].values()]):.5f}" for a_ in res)
    run.finish("COMPLETED", f"E13 sequence models ({summ})", {"by_arch": res, "per_seed": per,
                                                              "pred_dir": str(pdir), "elapsed_s": round(time.time() - t0, 1)})
    print(summ)


if __name__ == "__main__":
    main()

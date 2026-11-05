"""E06: optimisation study of the fusion model (post-registration, exploratory; selection on validation only).

1. learning-rate sweep (gated fusion, all six modalities, cosine schedule with warm-up, patience 6, <= 40 epochs)
2. capacity / regularisation variants at the best learning rate
3. five seeds of the best configuration -> seed spread and a seed-averaged ensemble
4. five seeds of the numbers-only head with the same recipe (fair comparison)
5. loss landscape: 1-D linear path between two seeds' solutions; 2-D filter-normalised random-direction slice around
   the best solution (Li et al., 2018), both scored on held-out validation rows
Every run records per-epoch cross-entropy on a training subset (eval mode), the inner block (early stopping) and the
validation block, so the loss and validation curves can be drawn. Neural networks have no certificate of a global
minimum; the evidence offered is convergence, seed stability and the local landscape.
Run: .venv/Scripts/python -I scripts/experiments/e06_optimization.py
"""
from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np
import torch

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import log_loss
from nexus_alpha.experiments import Run
from nexus_alpha.multimodal.dataset import build, load_tables
from nexus_alpha.multimodal.fusion import FusionConfig, FusionModel
from nexus_alpha.multimodal.trainer import evaluate_loss, predict, train

FULL = ("img_W", "img_D", "img_H", "num", "news", "sec")
SEEDS = (7, 11, 23, 42, 101)


def mean_ll(Y, P):
    v = [log_loss(Y[Y[:, j] >= 0, j], P[Y[:, j] >= 0, j]) for j in range(Y.shape[1]) if (Y[:, j] >= 0).any()]
    return float(np.mean(v))


def main() -> int:
    torch.set_num_threads(4)
    fr = load_tables()
    b = fr.block.to_numpy()
    tr, es, va, ca, te = (np.flatnonzero(b == k) for k in ("train", "inner", "valid", "calib", "test"))
    full, pre = build(fr, b == "train", FULL + ("rules",), "convnext_tiny")

    def sub(mods):
        names = [m for m in full.names if m in mods]
        return type(full)({k: full.x[k] for k in names}, full.mask[:, [full.names.index(k) for k in names]], full.y, full.z, names)

    run = Run("e06_optimization", {"variant": "gated", "modalities": FULL, "seeds": SEEDS, "schedule": "cosine",
                                   "epochs_max": 40, "patience": 6, "batch": 2048})
    out_dir = paths.ARTIFACTS / "predictions" / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()

    def fit(name, mods, lr=1e-3, wd=1e-4, d_model=64, dropout=0.1, seed=7, variant="gated"):
        d = sub(mods)
        cfg = FusionConfig(modality_dims={k: v.shape[1] for k, v in d.x.items()}, variant=variant, d_model=d_model,
                           hidden=2 * d_model, dropout=dropout, modality_dropout=0.15 if len(d.names) > 1 else 0.0)
        model, hist = train(cfg, d.subset(tr), d.subset(es), paths.ARTIFACTS / "models" / run.run_id / name, epochs=40,
                            batch=2048, lr=lr, weight_decay=wd, patience=6, seed=seed, log=lambda s: None,
                            monitor={"valid": d.subset(va)}, schedule="cosine")
        P, _, _ = predict(model, d)
        np.save(out_dir / f"{name}__probs.npy", P.astype(np.float32))
        r = {"lr": lr, "weight_decay": wd, "d_model": d_model, "dropout": dropout, "seed": seed, "modalities": list(d.names),
             "epochs": len(hist), "best_epoch": int(np.argmin([h["inner_ce"] for h in hist])), "history": hist,
             "valid": mean_ll(d.y[va], P[va]), "calib": mean_ll(d.y[ca], P[ca]), "test": mean_ll(d.y[te], P[te])}
        res[name] = r
        print(f"{name:<26} valid {r['valid']:.5f} test {r['test']:.5f} epochs {r['epochs']} best {r['best_epoch']} "
              f"({time.time() - t0:.0f} s)", flush=True)
        return model, d, cfg

    try:
        for lr in (3e-4, 1e-3, 3e-3):
            fit(f"lr_{lr:g}", FULL, lr=lr)
        best_lr = min((3e-4, 1e-3, 3e-3), key=lambda x: res[f"lr_{x:g}"]["valid"])
        fit("cap_d128", FULL, lr=best_lr, d_model=128)
        fit("reg_wd1e-3", FULL, lr=best_lr, wd=1e-3)
        fit("reg_dropout0.3", FULL, lr=best_lr, dropout=0.3)
        cands = {f"lr_{best_lr:g}": {}, "cap_d128": {"d_model": 128}, "reg_wd1e-3": {"wd": 1e-3},
                 "reg_dropout0.3": {"dropout": 0.3}}
        best_name = min(cands, key=lambda n: res[n]["valid"])
        kw = {"lr": best_lr, **cands[best_name]}
        models = {}
        for s in SEEDS:
            name = f"best_seed{s}"
            models[s] = fit(name, FULL, seed=s, **kw)
        for s in SEEDS:
            fit(f"num_seed{s}", ("num",), seed=s, **kw)
        Y = full.y
        for grp in ("best", "num"):
            P = np.mean([np.load(out_dir / f"{grp}_seed{s}__probs.npy") for s in SEEDS], axis=0)
            np.save(out_dir / f"{grp}_seed_ensemble__probs.npy", P.astype(np.float32))
            res[f"{grp}_seed_ensemble"] = {"valid": mean_ll(Y[va], P[va]), "calib": mean_ll(Y[ca], P[ca]), "test": mean_ll(Y[te], P[te])}
        # ---- landscape (validation rows, CE only)
        m_a, d, cfg = models[SEEDS[0]]
        m_b = models[SEEDS[1]][0]
        dv = d.subset(va[:30000])
        sa, sb = copy.deepcopy(m_a.state_dict()), m_b.state_dict()
        probe = FusionModel(cfg)
        path = []
        for a in np.linspace(-0.25, 1.25, 31):
            probe.load_state_dict({k: (1 - a) * sa[k] + a * sb[k] if sa[k].is_floating_point() else sa[k] for k in sa})
            path.append({"alpha": float(a), "valid_ce": evaluate_loss(probe, dv, cfg, torch.device("cpu"))})
        rng = torch.Generator().manual_seed(0)
        dirs = []
        for _ in range(2):                          # filter-normalised directions
            dd = {}
            for k, w in sa.items():
                r = torch.randn(w.shape, generator=rng) if w.is_floating_point() else torch.zeros_like(w)
                if w.is_floating_point() and w.dim() > 1:
                    r = r / (r.norm(dim=tuple(range(1, w.dim())), keepdim=True) + 1e-10) * w.norm(dim=tuple(range(1, w.dim())), keepdim=True)
                elif w.is_floating_point():
                    r = torch.zeros_like(w)         # biases and norms kept fixed
                dd[k] = r
            dirs.append(dd)
        grid = np.linspace(-1, 1, 21)
        surf = np.zeros((len(grid), len(grid)))
        for i, a in enumerate(grid):
            for j, c in enumerate(grid):
                probe.load_state_dict({k: sa[k] + a * dirs[0][k] + c * dirs[1][k] if sa[k].is_floating_point() else sa[k] for k in sa})
                surf[i, j] = evaluate_loss(probe, dv, cfg, torch.device("cpu"))
        np.save(out_dir / "landscape_2d.npy", surf)
        res["landscape"] = {"path_1d": path, "grid": grid.tolist(), "center_valid_ce": float(surf[10, 10]),
                            "min_on_grid": float(surf.min()), "argmin_on_grid": [float(grid[k]) for k in np.unravel_index(surf.argmin(), surf.shape)],
                            "rows": int(len(dv))}
        res["selection"] = {"best_lr": best_lr, "best_config": best_name, "kw": kw, "n_configs_searched": 6}
        seeds_v = [res[f"best_seed{s}"]["valid"] for s in SEEDS]
        seeds_t = [res[f"best_seed{s}"]["test"] for s in SEEDS]
        res["seed_spread"] = {"valid_mean": float(np.mean(seeds_v)), "valid_sd": float(np.std(seeds_v, ddof=1)),
                              "test_mean": float(np.mean(seeds_t)), "test_sd": float(np.std(seeds_t, ddof=1))}
        summ = (f"best {best_name} lr {best_lr:g}; 5-seed valid {np.mean(seeds_v):.5f}±{np.std(seeds_v, ddof=1):.5f}, "
                f"seed-ensemble valid {res['best_seed_ensemble']['valid']:.5f} test {res['best_seed_ensemble']['test']:.5f}; "
                f"numbers-only ensemble valid {res['num_seed_ensemble']['valid']:.5f} test {res['num_seed_ensemble']['test']:.5f}; "
                f"landscape centre {surf[10, 10]:.5f} grid min {surf.min():.5f}")
        run.finish("COMPLETED", summ, res)
        print(summ)
    except Exception as e:
        run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fusion trainer: masked multi-horizon loss, modality dropout, early stopping on the `inner` block only, resumable
checkpoints (state, optimiser, epoch, history) after every epoch, seeded and deterministic."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..training.hardware import detect, precision
from .fusion import FusionConfig, FusionModel, apply_modality_dropout, fusion_loss


@dataclass
class ModalityData:
    x: dict                 # name -> float32 (N, dim)
    mask: np.ndarray        # (N, M) float32, column order = names
    y: np.ndarray           # (N, H) int64, -1 = missing
    z: np.ndarray           # (N, H) float32, NaN = missing
    names: list

    def __len__(self):
        return len(self.mask)

    def subset(self, ix: np.ndarray) -> "ModalityData":
        return ModalityData({k: v[ix] for k, v in self.x.items()}, self.mask[ix], self.y[ix], self.z[ix], self.names)


def _tensors(d: ModalityData, ix, device):
    x = {k: torch.from_numpy(v[ix]).to(device) for k, v in d.x.items()}
    return x, torch.from_numpy(d.mask[ix]).to(device), torch.from_numpy(d.y[ix]).to(device), torch.from_numpy(d.z[ix]).to(device)


def evaluate_loss(model, d: ModalityData, cfg: FusionConfig, device, batch=4096) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(d), batch):
            ix = np.arange(s, min(len(d), s + batch))
            x, m, y, z = _tensors(d, ix, device)
            lg, q, _ = model(x, m)
            c = FusionConfig(**{**asdict(cfg), "quantile_weight": 0.0})
            k = int((y >= 0).sum())
            tot += float(fusion_loss(lg, q, y, z, c)) * k
            n += k
    return tot / max(n, 1)


def train(cfg: FusionConfig, tr: ModalityData, es: ModalityData, ckpt_dir: Path, epochs: int = 25, batch: int = 1024,
          lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 4, seed: int = 7, log=print,
          monitor: dict | None = None, schedule: str = "constant", warmup_epochs: int = 1, train_eval_rows: int = 20000,
          restore_best: bool = True):
    """monitor: extra blocks (e.g. {"valid": d_valid}) whose cross-entropy is RECORDED each epoch for validation curves;
    early stopping always uses `es` only. schedule: "constant" or "cosine" (linear warm-up, then cosine decay to 5% of
    lr over `epochs`). A fixed subset of training rows is scored each epoch in eval mode (no dropout), so train and
    held-out losses on the curves are comparable."""
    hw = detect()
    device = torch.device(hw["device"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(hw["device"] == "cpu", warn_only=True)
    model = FusionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_p = ckpt_dir / "last.pt"
    start, best, best_epoch, bad, hist = 0, float("inf"), -1, 0, []
    if state_p.exists():                                     # resume after a disconnect
        st = torch.load(state_p, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best, best_epoch, bad, hist = st["epoch"] + 1, st["best"], st["best_epoch"], st["bad"], st["hist"]
        log(f"resumed from epoch {st['epoch']}")
    amp = precision(hw)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr_eval = tr.subset(np.random.default_rng(seed).choice(len(tr), min(train_eval_rows, len(tr)), replace=False))
    steps_per_epoch = max(1, int(np.ceil(len(tr) / batch)))

    def lr_at(step):
        if schedule != "cosine":
            return lr
        w = warmup_epochs * steps_per_epoch
        if step < w:
            return lr * (step + 1) / w
        frac = (step - w) / max(1, epochs * steps_per_epoch - w)
        return lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, frac))))
    for ep in range(start, epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(tr), generator=gen).numpy()
        run_loss, nb = 0.0, 0
        for s in range(0, len(tr), batch):
            ix = perm[s: s + batch]
            x, m, y, z = _tensors(tr, ix, device)
            m = apply_modality_dropout(m, cfg.modality_dropout, generator=None)
            with torch.autocast(device_type=hw["device"], dtype=torch.bfloat16, enabled=(amp == "bf16")):
                lg, q, _ = model(x, m)
            loss = fusion_loss(lg.float(), q.float(), y, z, cfg)
            for g in opt.param_groups:
                g["lr"] = lr_at(ep * steps_per_epoch + s // batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item(); nb += 1
        val = evaluate_loss(model, es, cfg, device)
        rec = {"epoch": ep, "train_loss": run_loss / max(nb, 1), "train_ce_eval": evaluate_loss(model, tr_eval, cfg, device),
               "inner_ce": val, "lr_end": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        for name, dm in (monitor or {}).items():
            rec[f"{name}_ce"] = evaluate_loss(model, dm, cfg, device)
        hist.append(rec)
        if val < best - 1e-5:
            best, best_epoch, bad = val, ep, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                    "best_epoch": best_epoch, "bad": bad, "hist": hist}, state_p)
        log(f"epoch {ep}: train {hist[-1]['train_loss']:.4f} inner CE {val:.4f} ({hist[-1]['seconds']} s)")
        if bad >= patience:
            break
    if restore_best:                      # fixed-length refits keep the last epoch (no selection on the monitored block)
        st = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
    (ckpt_dir / "history.json").write_text(json.dumps({"hist": hist, "best_epoch": best_epoch, "best_inner_ce": best,
                                                        "hardware": hw, "precision": amp}, indent=1), encoding="utf-8")
    return model, hist


def predict(model, d: ModalityData, batch: int = 4096) -> tuple[np.ndarray, np.ndarray, dict]:
    device = next(model.parameters()).device
    model.eval()
    P, Q, G = [], [], []
    with torch.no_grad():
        for s in range(0, len(d), batch):
            ix = np.arange(s, min(len(d), s + batch))
            x, m, _, _ = _tensors(d, ix, device)
            lg, q, diag = model(x, m)
            P.append(torch.softmax(lg.float(), -1).cpu().numpy())
            Q.append(q.float().cpu().numpy())
            g = diag.get("gates", diag.get("modality_weights"))
            if g is not None:
                G.append(g.float().cpu().numpy())
    out = {"gates": np.concatenate(G)} if G else {}
    return np.concatenate(P), np.concatenate(Q), out

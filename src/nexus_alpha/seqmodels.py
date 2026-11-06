"""Sequence branch for v2 (E13): causal TCN and GRU over the last 63 daily bars of the stock and of SPY.

Inputs per session (scale-free, known at the close of t): stock log return, log high/low range, overnight gap, log volume
relative to its trailing 20-session mean; SPY log return and log range. Channels are standardised with training-row
statistics. Outputs: 3-class logits for every v2 horizon (masked cross-entropy; labels -1 are ignored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

WINDOW = 63
CHANNELS = ("ret", "range", "gap", "vol_rel", "spy_ret", "spy_range")


def channels(daily: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").set_index("date")
    c = d.close
    f = pd.DataFrame(index=d.index)
    f["ret"] = np.log(c / c.shift(1))
    f["range"] = np.log(d.high / d.low)
    f["gap"] = np.log(d.open / c.shift(1))
    v = d.volume.replace(0, np.nan)
    f["vol_rel"] = np.log(v / v.rolling(20, min_periods=5).mean().shift(1))
    s = spy.set_index("date").sort_index().reindex(f.index)
    f["spy_ret"] = np.log(s.close / s.close.shift(1))
    f["spy_range"] = np.log(s.high / s.low)
    return f.replace([np.inf, -np.inf], np.nan)


def windows(ch: pd.DataFrame, asof: pd.DatetimeIndex) -> np.ndarray:
    """(n, WINDOW, C) float32 windows ending at each as-of session (NaN -> 0 later, after standardisation)."""
    a = ch.to_numpy(np.float32)
    pos = ch.index.get_indexer(asof)
    out = np.full((len(asof), WINDOW, a.shape[1]), np.nan, np.float32)
    for i, k in enumerate(pos):
        if k < 0:
            continue
        lo = max(0, k - WINDOW + 1)
        out[i, WINDOW - (k - lo + 1):] = a[lo:k + 1]
    return out


class TCN(nn.Module):
    def __init__(self, c_in: int, n_h: int, ch: int = 32, dilations=(1, 2, 4, 8, 16), k: int = 3, p: float = 0.2):
        super().__init__()
        layers, c = [], c_in
        for d in dilations:
            layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(c, ch, k, dilation=d), nn.GELU(), nn.Dropout(p)]
            c = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(ch * 2, n_h * 3)
        self.n_h = n_h

    def forward(self, x):                     # x: (B, T, C)
        h = self.net(x.transpose(1, 2))       # causal: output at T-1 sees only inputs <= T-1
        z = torch.cat([h[:, :, -1], h.mean(-1)], 1)
        return self.head(z).view(-1, self.n_h, 3)


class GRUNet(nn.Module):
    def __init__(self, c_in: int, n_h: int, hidden: int = 64, p: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(c_in, hidden, batch_first=True)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(hidden, n_h * 3)
        self.n_h = n_h

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.drop(h[-1])).view(-1, self.n_h, 3)


def masked_ce(logits, y):
    m = y >= 0
    if m.sum() == 0:
        return logits.sum() * 0
    return nn.functional.cross_entropy(logits[m], y[m])


def fit(model, Xtr, Ytr, Xes=None, Yes=None, epochs=30, batch=512, lr=3e-4, wd=1e-4, patience=6, seed=7, log=print,
        restore_best=True):
    """restore_best=False (refits): train exactly `epochs` epochs and keep the final weights; Xes is not looked at."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(Xtr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 2, pct_start=max(1 / epochs, 2.5 / (steps + 2)),
                                                anneal_strategy="cos")       # linear-ish warm-up 1 epoch, cosine decay
    Xt, Yt = torch.from_numpy(np.ascontiguousarray(Xtr, np.float32)), torch.from_numpy(np.asarray(Ytr, np.int64))
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), batch):
            ix = perm[i:i + batch]
            loss = masked_ce(model(Xt[ix]), Yt[ix])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
        if not restore_best:
            hist.append({"epoch": ep, "train_loss": tot / len(Xt)})
            continue
        val = evaluate(model, Xes, Yes)
        hist.append({"epoch": ep, "train_loss": tot / len(Xt), "inner_ce": val})
        log(f"epoch {ep}: train {tot / len(Xt):.4f} inner {val:.4f}")
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if restore_best:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def evaluate(model, X, Y, batch=4096) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), batch):
        lg = model(torch.from_numpy(X[i:i + batch]))
        y = torch.from_numpy(np.asarray(Y[i:i + batch], np.int64))
        m = y >= 0
        if m.sum():
            tot += float(nn.functional.cross_entropy(lg[m], y[m], reduction="sum"))
            n += int(m.sum())
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, X, batch=4096) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.softmax(model(torch.from_numpy(X[i:i + batch])), -1).numpy()
                           for i in range(0, len(X), batch)])

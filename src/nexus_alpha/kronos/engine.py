"""Zero-shot Kronos path sampling and label-rule probabilities.

`sample_paths` is the vendored `auto_regressive_inference` loop with two differences, both documented in docs/KRONOS.md:
1. It returns every sampled path (the original returns their mean), because probabilities need the whole distribution.
2. Token sampling uses inverse-CDF draws from per-series random streams seeded by (ticker, as-of date, path), so a
   stock's forecast does not depend on which other stocks share its batch and every logged forecast can be reproduced
   exactly. Distributionally this is the same categorical draw as `torch.multinomial`; nucleus filtering is the vendored
   `top_k_top_p_filtering`.

Inputs follow `KronosPredictor.predict`: open, high, low, close, volume, amount (= volume x mean OHLC when missing), each
z-scored over its own context window and clipped to +-5; time features minute/hour/weekday/day/month; future time stamps
from the market calendar. Only bars up to and including the as-of session enter the context (point in time).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.calendar import offset_session
from ..targets import COST_FLOOR, K_SIGMA
from .vendor.kronos import calc_time_stamps, top_k_top_p_filtering

COLS = ["open", "high", "low", "close", "volume", "amount"]
CLIP = 5.0


def series_seed(ticker: str, asof, salt: str = "nexus-kronos-v1") -> int:
    h = hashlib.sha256(f"{salt}|{ticker}|{pd.Timestamp(asof).date()}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def context(daily: pd.DataFrame, k: int, lookback: int, horizon: int):
    """Context window ending at row k (the as-of session) and the next `horizon` session dates.
    Returns (x [L, 6] float32, x_stamp [L, 5], y_stamp [H, 5], mean [6], std [6]) or None when history is short."""
    if k + 1 < lookback:
        return None
    w = daily.iloc[k + 1 - lookback: k + 1]
    x = pd.DataFrame({c: w[c].astype(float).to_numpy() for c in ["open", "high", "low", "close"]})
    x["volume"] = w["volume"].astype(float).fillna(0.0).to_numpy() if "volume" in w else 0.0
    x["amount"] = x["volume"] * x[["open", "high", "low", "close"]].mean(axis=1)
    x = x[COLS].to_numpy(np.float32)
    if not np.isfinite(x).all():
        return None
    ts = pd.Series(pd.to_datetime(w["date"].to_numpy()))
    last = pd.DatetimeIndex([ts.iloc[-1]])
    fut = pd.Series(pd.DatetimeIndex([offset_session(last, i)[0] for i in range(1, horizon + 1)]))
    if fut.isna().any():
        return None                                                  # beyond the market calendar
    mean, std = x.mean(axis=0), x.std(axis=0)
    xn = np.clip((x - mean) / (std + 1e-5), -CLIP, CLIP).astype(np.float32)
    return (xn, calc_time_stamps(ts).to_numpy(np.float32), calc_time_stamps(fut).to_numpy(np.float32), mean, std)


def _draw(logits: torch.Tensor, u: torch.Tensor, T: float, top_p: float) -> torch.Tensor:
    """One categorical draw per row by inverse CDF with the given uniforms u (rows,)."""
    logits = logits / T
    if top_p < 1.0:
        logits = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=top_p)
    p = torch.softmax(logits, dim=-1)
    c = torch.cumsum(p, dim=-1)
    c[:, -1] = 1.0                                                   # guard against rounding below 1
    return torch.searchsorted(c, u.unsqueeze(-1).to(c.dtype)).clamp_max(p.size(-1) - 1)


class _KV:
    """Key/value cache of the Kronos decoder: per layer K and V, plus the final normalised hidden states (the s2 step
    cross-attends over all of them)."""

    def __init__(self, n_layers: int):
        self.k, self.v, self.ctx = [None] * n_layers, [None] * n_layers, None
        self.ck = self.cv = None          # cross-attention keys/values of the dependency layer over ctx


def _rope_at(attn, q, k, pos0: int):
    """The vendored rotary embedding, evaluated at absolute positions pos0 .. pos0+S-1 (the original always starts at 0,
    which is the same thing while the context window does not slide)."""
    inv = attn.rotary.inv_freq
    t = torch.arange(pos0, pos0 + q.size(-2), device=q.device).type_as(inv)
    emb = torch.cat([torch.einsum("i,j->ij", t, inv)] * 2, dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    rh = attn.rotary._rotate_half
    return q * cos + rh(q) * sin, k * cos + rh(k) * sin


def _block_cached(block, x, li: int, kv: _KV, pos0: int):
    a = block.self_attn
    h = block.norm1(x)
    B, S, _ = h.shape
    sh = lambda t: t.view(B, S, a.n_heads, a.head_dim).transpose(1, 2)
    q, k, v = sh(a.q_proj(h)), sh(a.k_proj(h)), sh(a.v_proj(h))
    q, k = _rope_at(a, q, k, pos0)
    if kv.k[li] is not None:
        k, v = torch.cat([kv.k[li], k], dim=2), torch.cat([kv.v[li], v], dim=2)
    kv.k[li], kv.v[li] = k, v
    # prefill: causal over the context; one new token: it may attend to every cached position
    o = F.scaled_dot_product_attention(q, k, v, is_causal=(kv.ctx is None and S > 1))
    x = x + a.out_proj(o.transpose(1, 2).contiguous().view(B, S, a.d_model))
    return x + block.ffn(block.norm2(x))


def _s1_cached(model, s1, s2, stamp, kv: _KV, pos0: int):
    x = model.embedding([s1, s2]) + model.time_emb(stamp)
    for li, layer in enumerate(model.transformer):
        x = _block_cached(layer, x, li, kv, pos0)
    x = model.norm(x)
    kv.ctx = x if kv.ctx is None else torch.cat([kv.ctx, x], dim=1)
    return model.head(x[:, -1:, :]), kv.ctx


def _s2_last(model, ctx, s1):
    """decode_s2 at the last position only (the dependency layer's cross-attention still reads the whole context)."""
    q = model.embedding.emb_s1(s1)
    att = model.dep_layer.cross_attn(query=q, key=ctx, value=ctx)
    return model.head.cond_forward(model.dep_layer.norm(ctx[:, -1:, :] + att))


def _s2_cached(model, kv: _KV, s1, n_new: int):
    """_s2_last with the cross-attention key/value projections of the context cached (only the n_new newest positions
    are projected). Exact: the vendored cross-attention applies its rotary embedding for q_len = 1, i.e. at position 0,
    where cos = 1 and sin = 0, so keys pass through unchanged and cached projections equal fresh ones."""
    ca = model.dep_layer.cross_attn
    R, S, _ = kv.ctx.shape
    new = kv.ctx[:, S - n_new:, :]
    sh = lambda t, n: t.view(R, n, ca.n_heads, ca.head_dim).transpose(1, 2)
    k_new, v_new = sh(ca.k_proj(new), n_new), sh(ca.v_proj(new), n_new)
    kv.ck = k_new if kv.ck is None else torch.cat([kv.ck, k_new], dim=2)
    kv.cv = v_new if kv.cv is None else torch.cat([kv.cv, v_new], dim=2)
    q = sh(ca.q_proj(model.embedding.emb_s1(s1)), 1)
    o = F.scaled_dot_product_attention(q, kv.ck, kv.cv)
    att = ca.out_proj(o.transpose(1, 2).contiguous().view(R, 1, ca.d_model))
    return model.head.cond_forward(model.dep_layer.norm(kv.ctx[:, -1:, :] + att))


@torch.no_grad()
def sample_paths(model, tokenizer, ctxs: list, seeds: list[int], n_paths: int, horizon: int, max_context: int,
                 T: float = 1.0, top_p: float = 1.0, device: str = "cpu", cache: bool = True) -> np.ndarray:
    """ctxs: outputs of `context` (same lookback); seeds: one per series. Returns paths [B, N, H, 6] in price units.
    cache=True uses the key/value cache (about 20x faster for a 400-bar context; valid while context + horizon fits the
    model's window, otherwise the reference loop is used). cache=False is the reference: a full forward pass per step,
    exactly as the vendored `auto_regressive_inference`."""
    B = len(ctxs)
    x = torch.from_numpy(np.stack([c[0] for c in ctxs])).to(device)
    xs = torch.from_numpy(np.stack([c[1] for c in ctxs])).to(device)
    ys = torch.from_numpy(np.stack([c[2] for c in ctxs])).to(device)
    rep = lambda t: t.repeat_interleave(n_paths, dim=0)               # series-major: row = series * n_paths + path
    # uniforms: per series and path, 2 per step (s1 and s2 tokens), from that series' own stream
    U = np.stack([np.random.default_rng(s).random((n_paths, horizon, 2)) for s in seeds]).reshape(B * n_paths, horizon, 2)
    U = torch.from_numpy(U.astype(np.float64)).to(device)
    L = x.size(1)
    use_cache = cache and L + horizon <= max_context
    # the context is identical for every path of a series: with the cache it is encoded and prefilled once per series
    tok = tokenizer.encode(torch.clip(x if use_cache else rep(x), -CLIP, CLIP), half=True)
    if use_cache:
        tok = [rep(t) for t in tok]
    full_stamp = torch.cat([rep(xs), rep(ys)], dim=1)
    pre = torch.cat([tok[0], tok[0].new_zeros(B * n_paths, horizon)], dim=1)
    post = torch.cat([tok[1], tok[1].new_zeros(B * n_paths, horizon)], dim=1)
    kv = _KV(len(model.transformer)) if use_cache else None
    for i in range(horizon):
        cur = L + i
        if use_cache:
            if i == 0:                                                # prefill the B distinct contexts, then copy
                rows = torch.arange(0, B * n_paths, n_paths, device=pre.device)
                s1_logits, _ = _s1_cached(model, pre[rows, :L], post[rows, :L], full_stamp[rows, :L, :].contiguous(), kv, 0)
                kv.k = [rep(t) for t in kv.k]
                kv.v = [rep(t) for t in kv.v]
                kv.ctx = rep(kv.ctx)
                s1_logits = rep(s1_logits)
            else:                                                     # one new token per step
                s1_logits, _ = _s1_cached(model, pre[:, cur - 1:cur], post[:, cur - 1:cur],
                                          full_stamp[:, cur - 1:cur, :].contiguous(), kv, cur - 1)
            s1 = _draw(s1_logits[:, -1, :], U[:, i, 0], T, top_p)
            s2_logits = _s2_cached(model, kv, s1, L if i == 0 else 1)
        else:
            s = max(0, cur - max_context)
            s1_logits, ctx = model.decode_s1(pre[:, s:cur], post[:, s:cur], full_stamp[:, s:cur, :].contiguous())
            s1 = _draw(s1_logits[:, -1, :], U[:, i, 0], T, top_p)
            s2_logits = model.decode_s2(ctx, s1)
        s2 = _draw(s2_logits[:, -1, :], U[:, i, 1], T, top_p)
        pre[:, cur] = s1.squeeze(-1)
        post[:, cur] = s2.squeeze(-1)
    s = max(0, L + horizon - max_context)
    z = tokenizer.decode([pre[:, s:L + horizon].contiguous(), post[:, s:L + horizon].contiguous()], half=True)
    z = z[:, -horizon:, :].reshape(B, n_paths, horizon, -1).cpu().numpy()
    mean = np.stack([c[3] for c in ctxs])[:, None, None, :]
    std = np.stack([c[4] for c in ctxs])[:, None, None, :]
    return z * (std + 1e-5) + mean


def label_probabilities(paths: np.ndarray, sigma: np.ndarray, horizons: dict[str, int], alpha: float = 0.5) -> dict:
    """paths [B, N, H, 6]; sigma [B]: EWMA daily volatility at the as-of session (the label rule's sigma_t).
    The v2 label rule on every path: R_h = close[t+h] / open[t+1] - 1, class by tau = max(0.5 sigma sqrt(h), 10 bps).
    Probabilities = (count + alpha) / (N + 3 alpha) (no 0/1 probabilities from a finite sample). Also the mean and the
    standard deviation of R_h across paths."""
    N = paths.shape[1]
    o1 = paths[:, :, 0, 0]
    out = {}
    for name, h in horizons.items():
        r = paths[:, :, h - 1, 3] / o1 - 1
        tau = np.maximum(K_SIGMA * sigma * np.sqrt(h), COST_FLOOR)[:, None]
        cnt = np.stack([(r < -tau).sum(1), ((r >= -tau) & (r <= tau)).sum(1), (r > tau).sum(1)], axis=1).astype(float)
        p = (cnt + alpha) / (N + 3 * alpha)
        out[name] = {"p": p, "ret_mean": np.nanmean(r, axis=1), "ret_sd": np.nanstd(r, axis=1),
                     "ret_q10": np.nanquantile(r, 0.1, axis=1), "ret_q50": np.nanquantile(r, 0.5, axis=1),
                     "ret_q90": np.nanquantile(r, 0.9, axis=1)}
    return out

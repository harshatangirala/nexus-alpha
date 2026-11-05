"""Trainable multimodal fusion (master spec 6.4-6.5, docs/FUSION_DESIGN.md).

Inputs are modality blocks with presence masks: img_W, img_D, img_H (chart embeddings per timeframe), num (indicator
features), news (aggregated article representation), sec (filing features and text). A missing block is zeroed AND
flagged in the mask; attention ignores it and gates are forced to zero, so absence is never read as neutral evidence.

Variants (same heads and loss):
  late      - one small head per modality; the fused logits are a learned, masked weighted average of modality logits
  concat    - projected modality vectors (+ mask bits) concatenated into an MLP
  attention - modality tokens with learned modality/timeframe embeddings and a CLS token through a Transformer encoder
              (cross-modal self-attention, padding mask for missing modalities)
  gated     - softmax gates over present modalities weight their projected vectors (gates are reported as diagnostics)
  hybrid    - attention-contextualised tokens pooled by the gating network instead of the CLS token
Heads: six horizon classifiers (DOWN, SIDEWAYS, UP) and quantile heads (0.1, 0.5, 0.9) of the volatility-scaled return z_h.
Gates and attention weights are model diagnostics, not causal explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class FusionConfig:
    modality_dims: dict = field(default_factory=dict)       # name -> input dim
    n_horizons: int = 6
    d_model: int = 64
    hidden: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    variant: str = "attention"
    modality_dropout: float = 0.15
    quantile_weight: float = 0.2


class Projector(nn.Module):
    def __init__(self, d_in: int, d: int, p: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):
        return self.net(x)


class Heads(nn.Module):
    def __init__(self, d: int, n_h: int):
        super().__init__()
        self.cls = nn.Linear(d, n_h * 3)
        self.q = nn.Linear(d, n_h * len(QUANTILES))
        self.n_h = n_h

    def forward(self, z):
        return self.cls(z).view(-1, self.n_h, 3), self.q(z).view(-1, self.n_h, len(QUANTILES))


class FusionModel(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.names = list(cfg.modality_dims)
        d = cfg.d_model
        self.proj = nn.ModuleDict({m: Projector(k, d, cfg.dropout) for m, k in cfg.modality_dims.items()})
        self.type_emb = nn.Parameter(torch.randn(len(self.names), d) * 0.02)       # modality + timeframe identity
        v = cfg.variant
        if v in ("attention", "hybrid"):
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.hidden, cfg.dropout, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
            self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        if v in ("gated", "hybrid"):
            self.gate = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        if v == "concat":
            self.mlp = nn.Sequential(nn.Linear(len(self.names) * (d + 1), cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                     nn.Linear(cfg.hidden, d), nn.LayerNorm(d))
        if v == "late":
            self.mod_heads = nn.ModuleDict({m: Heads(d, cfg.n_horizons) for m in self.names})
            self.mod_w = nn.Parameter(torch.zeros(len(self.names)))
        else:
            self.heads = Heads(d, cfg.n_horizons)

    def forward(self, x: dict, mask: torch.Tensor):
        """x: name -> (B, dim); mask: (B, M) float, 1 = present. Returns (logits (B,H,3), quantiles (B,H,Q), diag)."""
        B = mask.shape[0]
        H = torch.stack([self.proj[m](x[m]) for m in self.names], 1) + self.type_emb[None]      # (B, M, d)
        H = H * mask[..., None]
        v = self.cfg.variant
        diag = {}
        if v == "late":
            outs = [self.mod_heads[m](H[:, i]) for i, m in enumerate(self.names)]
            w = self.mod_w[None].expand(B, -1).masked_fill(mask == 0, -1e9).softmax(-1)          # (B, M)
            logits = sum(w[:, i, None, None] * o[0] for i, o in enumerate(outs))
            quant = sum(w[:, i, None, None] * o[1] for i, o in enumerate(outs))
            diag["modality_weights"] = w
            return logits, quant, diag
        if v == "concat":
            z = self.mlp(torch.cat([H.flatten(1), mask], 1))
        elif v in ("attention", "hybrid"):
            tok = torch.cat([self.cls_tok.expand(B, -1, -1), H], 1)
            pad = torch.cat([torch.zeros(B, 1, device=mask.device, dtype=torch.bool), mask == 0], 1)
            out = self.encoder(tok, src_key_padding_mask=pad)
            if v == "attention":
                z = out[:, 0]
            else:
                g = self.gate(out[:, 1:]).squeeze(-1).masked_fill(mask == 0, -1e9).softmax(-1)
                z = (g[..., None] * out[:, 1:]).sum(1)
                diag["gates"] = g
        elif v == "gated":
            g = self.gate(H).squeeze(-1).masked_fill(mask == 0, -1e9).softmax(-1)
            z = (g[..., None] * H).sum(1)
            diag["gates"] = g
        else:
            raise ValueError(v)
        logits, quant = self.heads(z)
        return logits, quant, diag


def fusion_loss(logits, quant, y, z, cfg: FusionConfig):
    """Masked multi-horizon cross-entropy (+ pinball loss on vol-scaled returns). y: (B,H) long with -1 = missing;
    z: (B,H) float with NaN = missing."""
    ce = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), ignore_index=-1, reduction="sum")
    n = (y >= 0).sum().clamp(min=1)
    loss = ce / n
    if cfg.quantile_weight > 0:
        zm = torch.isfinite(z)
        zz = torch.nan_to_num(z)
        qs = torch.tensor(QUANTILES, device=z.device)
        d = zz[..., None] - quant
        pin = torch.maximum(qs * d, (qs - 1) * d) * zm[..., None]
        loss = loss + cfg.quantile_weight * pin.sum() / (zm.sum().clamp(min=1) * len(QUANTILES))
    return loss


def apply_modality_dropout(mask: torch.Tensor, p: float, generator=None) -> torch.Tensor:
    """Randomly hide present modalities during training (never all of them)."""
    if p <= 0:
        return mask
    drop = (torch.rand(mask.shape, generator=generator, device=mask.device) < p).float()
    new = mask * (1 - drop)
    empty = new.sum(1) == 0
    new[empty] = mask[empty]
    return new

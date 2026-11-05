"""CV backbone registry and embedder (contract identical to TriFrame tfcv_core.to_input / embed_views).

Specs come from model_registry/triframe_backbones_s500.json (a byte copy of triframe-cv/results/s500_run/backbones.json).
Frozen backbones load timm ImageNet weights; fine-tuned ones (ft_*) build the timm architecture without weights and load
the s500 safetensors file from the private bundle, downloaded into artifacts/checkpoints (never committed).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import paths

SPEC_FILE = paths.REPO_ROOT / "model_registry" / "triframe_backbones_s500.json"
FT_REPO_FILES = {"ft_convnext_tiny": "ft_convnext_tiny.safetensors"}


@lru_cache(maxsize=1)
def specs() -> dict:
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def to_input(img01: np.ndarray, mean, std):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(img01.astype(np.float32))
    if t.ndim == 2:
        t = t[None]
    t = t[:, None].repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return (t - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def ft_weights_path(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    fn = FT_REPO_FILES[name]
    dest = paths.ARTIFACTS / "checkpoints" / "triframe_s500"
    dest.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(paths.HF_TRIFRAME_S500_BUNDLE, fn, local_dir=str(dest)))


_MODELS: dict = {}


def load_backbone(name: str):
    import timm
    import torch
    if name in _MODELS:
        return _MODELS[name]
    s = specs()[name]
    torch.manual_seed(int(s.get("seed", 0)))
    if name.startswith("ft_"):
        from safetensors.torch import load_file
        m = timm.create_model(s["timm"], pretrained=False, num_classes=0, **s.get("kwargs", {}))
        m.load_state_dict(load_file(str(ft_weights_path(name)), device="cpu"))
    else:
        m = timm.create_model(s["timm_name"], pretrained=bool(s.get("pretrained", True)), num_classes=0,
                              **s.get("kwargs", {}))
    _MODELS[name] = m.eval()
    return _MODELS[name]


def embed(name: str, images: list[np.ndarray], batch: int = 32) -> np.ndarray:
    """Pooled embeddings for a list of binary chart images (any widths)."""
    import torch
    s = specs()[name]
    m = load_backbone(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.cat([to_input(im, s["mean"], s["std"]) for im in images[i: i + batch]])
            out.append(m(x).float().numpy())
    return np.concatenate(out) if out else np.zeros((0, s["emb_dim"]), np.float32)

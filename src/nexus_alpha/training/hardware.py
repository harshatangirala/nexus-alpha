"""Runtime hardware detection and precision choice (master spec section 10): works on a laptop CPU or any Colab GPU."""
from __future__ import annotations

import os
import platform
import shutil


def detect() -> dict:
    import torch
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "platform": platform.platform(),
            "cpu_threads": os.cpu_count()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"device": "cuda", "gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
                     "cuda": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported(),
                     "capability": f"{p.major}.{p.minor}"})
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info.update({"device": "mps", "bf16": False})
    else:
        info.update({"device": "cpu", "bf16": False})
    try:
        total, used, free = shutil.disk_usage(os.getcwd())
        info["disk_free_gb"] = round(free / 1e9, 1)
    except OSError:
        pass
    return info


def precision(info: dict) -> str:
    if info["device"] == "cuda":
        return "bf16" if info.get("bf16") else "fp16"
    return "fp32"


def batch_size(info: dict, base: int = 1024) -> int:
    """Fusion heads on cached embeddings are small: scale the batch with VRAM, keep CPU batches modest."""
    if info["device"] == "cuda":
        v = info.get("vram_gb", 16)
        return int(base * (4 if v >= 40 else 2 if v >= 20 else 1))
    return base

"""Read-only bridge to code in the source projects, used only for verification (parity checks).

Importing a module from a source tree would normally write __pycache__ files into it. These helpers turn bytecode writing
off before importing, so the source projects stay byte-for-byte unchanged.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from . import paths


def _import_from(path: Path, name: str):
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = prev


def triframe_image_store(who: str):
    """TriFrame's sealed ImageStore (allow_holdout is never set here)."""
    mod = _import_from(paths.OHLCV_ROOT / "scripts" / "triframe_loader.py", "triframe_loader")
    return mod.ImageStore(who=who)


def triframe_feat_visible():
    return _import_from(paths.TRIFRAME_REPO / "feat_visible.py", "feat_visible")

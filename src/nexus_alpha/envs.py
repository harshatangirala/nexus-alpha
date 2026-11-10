"""Separate environments for heavy third-party tools (D-046): they never share the frozen models' environment.

* artifacts/envs/qlib           pyqlib 0.9.7 (Alpha158 parity check only)
* artifacts/envs/openbb         OpenBB 5.0.0 (CBOE option chains: dashboard and the E26 snapshot)
* artifacts/envs/tradingagents  TradingAgents v0.7.1 (the on-demand LLM committee, E27)

Each is called through a subprocess and exchanges files, so the project environment imports none of them.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import paths


def env_python(name: str) -> Path | None:
    """The interpreter of a side environment, or None if it is not installed."""
    root = paths.ARTIFACTS / "envs" / name
    for p in (root / "Scripts" / "python.exe", root / "bin" / "python"):
        if p.exists():
            return p
    return None


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for a side-environment subprocess: the current one without the sealed-period variables."""
    env = dict(os.environ)
    for k in ("NEXUS_HOLDOUT_CONFIG", "NEXUS_HOLDOUT_UNLOCK", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(k, None)
    env.update(extra or {})
    return env

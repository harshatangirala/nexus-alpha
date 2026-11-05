"""Run IDs, machine-readable run manifests and the human experiment log (master spec 18).

Every run writes experiments/<run_id>/manifest.json (config, config hash, git commit and dirty flag, input file hashes,
environment, timings, status) and metrics.json, and appends one line to docs/EXPERIMENT_LOG.md. Failed and abandoned
runs are logged too (status FAILED / ABANDONED), so the search is fully counted for multiple-testing corrections.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, paths

EXP_DIR = paths.REPO_ROOT / "experiments"
LOG = paths.REPO_ROOT / "docs" / "EXPERIMENT_LOG.md"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_digest(p: Path, max_bytes: int = 64 << 20) -> str:
    """sha256 of a file (first max_bytes plus size for very large files, flagged)."""
    p = Path(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(max_bytes))
    size = p.stat().st_size
    tag = "" if size <= max_bytes else f"+partial({size})"
    return h.hexdigest()[:24] + tag


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "nexus_alpha": __version__,
           "numpy": np.__version__}
    for mod in ("pandas", "torch", "lightgbm", "sklearn", "timm", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        env["cuda"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


class Run:
    def __init__(self, name: str, config: dict, inputs: dict[str, Path] | None = None, profile: str = "local-cpu"):
        self.name, self.config, self.profile = name, config, profile
        self.cfg_hash = config_hash(config)
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + f"_{name}_{self.cfg_hash[:6]}"
        self.dir = EXP_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.manifest = {"run_id": self.run_id, "name": name, "profile": profile, "config": config,
                         "config_hash": self.cfg_hash, "git_commit": _git("rev-parse", "HEAD"),
                         "git_dirty": bool(_git("status", "--porcelain")),
                         "git_dirty_files": _git("status", "--porcelain").splitlines()[:200],
                         "git_diff_sha256": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest(),
                         "environment": environment(),
                         "inputs": {k: {"path": str(v), "digest": file_digest(v)} for k, v in (inputs or {}).items()},
                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "RUNNING"}
        self._write()

    def _write(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1, default=str), encoding="utf-8")

    def save(self, name: str, obj) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, indent=1, default=_np_default), encoding="utf-8")
        return p

    def finish(self, status: str, summary: str, metrics: dict | None = None) -> None:
        self.manifest.update({"status": status, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "elapsed_s": round(time.time() - self.t0, 1), "summary": summary})
        if metrics is not None:
            self.save("metrics.json", metrics)
        self._write()
        LOG.parent.mkdir(exist_ok=True)
        new = not LOG.exists()
        with open(LOG, "a", encoding="utf-8") as f:
            if new:
                f.write("# Experiment log\n\nOne line per run, appended automatically by `nexus_alpha.experiments.Run`. "
                        "Failed and abandoned runs stay in the log so the full search is counted.\n\n"
                        "| run_id | name | profile | status | elapsed | summary |\n|---|---|---|---|---|---|\n")
            f.write(f"| `{self.run_id}` | {self.name} | {self.profile} | {status} | {self.manifest['elapsed_s']} s | "
                    f"{summary.replace('|', '/')} |\n")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

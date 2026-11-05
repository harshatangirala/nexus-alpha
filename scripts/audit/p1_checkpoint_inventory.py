"""Phase 1: inventory of reusable model checkpoints (metadata only; nothing is downloaded here).

* Private TriFrame bundles on Hugging Face (owner KrishnaHarshaTangirala): file lists and sizes, if the cached login
  has read access.
* Public pretrained models already in the local Hugging Face cache (CV backbones, financial language models).

Output: model_registry/checkpoint_inventory.json
Run: .venv/Scripts/python -I scripts/audit/p1_checkpoint_inventory.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
PRIVATE = [
    ("KrishnaHarshaTangirala/triframe-cv-model", "model"),
    ("KrishnaHarshaTangirala/triframe-cv-model-s120", "model"),
    ("KrishnaHarshaTangirala/triframe-cv-model-s500", "model"),
    ("KrishnaHarshaTangirala/triframe-cv-runs", "dataset"),
]
CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def main() -> int:
    api = HfApi()
    out = {"queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "private_repos": {}, "local_cache": {}}
    for repo_id, kind in PRIVATE:
        try:
            info = (api.model_info if kind == "model" else api.dataset_info)(repo_id, files_metadata=True)
            files = [{"path": s.rfilename, "bytes": getattr(s, "size", None)} for s in (info.siblings or [])]
            out["private_repos"][repo_id] = {"kind": kind, "sha": info.sha, "private": info.private,
                                             "last_modified": str(info.last_modified), "n_files": len(files),
                                             "total_bytes": sum(f["bytes"] or 0 for f in files), "files": files}
            print(f"{repo_id}: {len(files)} files, {sum(f['bytes'] or 0 for f in files) / 1e6:.1f} MB")
        except Exception as e:
            out["private_repos"][repo_id] = {"kind": kind, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{repo_id}: ERROR {type(e).__name__}")
    for d in sorted(CACHE.glob("models--*")):
        snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
        files = []
        for s in snaps:
            for p in s.rglob("*"):
                if p.is_file():
                    files.append({"snapshot": s.name, "path": p.relative_to(s).as_posix(), "bytes": p.stat().st_size})
        out["local_cache"][d.name.replace("models--", "").replace("--", "/")] = {
            "snapshots": [s.name for s in snaps], "files": files,
            "total_bytes": sum(f["bytes"] for f in files)}
    (REPO / "model_registry").mkdir(exist_ok=True)
    (REPO / "model_registry" / "checkpoint_inventory.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["local_cache"].items():
        print(f"local {k:<55} {v['total_bytes'] / 1e6:>9.1f} MB  snapshots={len(v['snapshots'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

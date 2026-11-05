"""Phase 1: layout of TriFrame's stored chart-embedding chunks, read through HTTP range requests.

An .npz file is a zip archive; this script reads only the zip central directory and each member's .npy header
(name, dtype, shape) from the private run repository, so no embedding values are transferred.

Output: model_registry/triframe_embedding_layout.json
Run: .venv/Scripts/python -I scripts/audit/p1_embedding_chunk_layout.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem

REPO = Path(__file__).resolve().parents[2]
RUNS = "datasets/KrishnaHarshaTangirala/triframe-cv-runs"
TARGETS = ["emb_s500/chunk_0000.npz", "emb_ft_s500_8ed8458840/chunk_0000.npz", "emb_s500/chunk_0012.npz"]


def npy_header(zf: zipfile.ZipFile, name: str) -> dict:
    with zf.open(name) as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    return {"shape": list(shape), "dtype": str(dtype)}


def main() -> int:
    fs = HfFileSystem()
    out = {}
    for t in TARGETS:
        rec = {"members": {}}
        with fs.open(f"{RUNS}/{t}", "rb", block_size=1 << 20) as fh:
            zf = zipfile.ZipFile(fh)
            for zi in zf.infolist():
                h = npy_header(zf, zi.filename) if zi.filename.endswith(".npy") else {}
                rec["members"][zi.filename] = {"compressed": zi.compress_size, "size": zi.file_size, **h}
        out[t] = rec
        print(f"== {t}: {len(rec['members'])} members")
        for k, v in list(rec["members"].items())[:80]:
            print(f"   {k:<60} {v.get('dtype', ''):<8} {v.get('shape', '')}  {v['size'] / 1e6:.1f} MB")
    (REPO / "model_registry" / "triframe_embedding_layout.json").write_text(json.dumps(out, indent=1),
                                                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

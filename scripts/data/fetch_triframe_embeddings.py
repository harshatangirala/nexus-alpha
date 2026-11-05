"""Fetch selected TriFrame s500 chart embeddings (reuse instead of re-embedding 236k chart sets on a laptop CPU).

Source: private HF dataset KrishnaHarshaTangirala/triframe-cv-runs, folders emb_s500/ (frozen ImageNet backbones and
random-weight controls) and emb_ft_s500_8ed8458840/ (chart-fine-tuned backbones), 13 .npz chunks each. Only the
selected members are transferred (zip members are read through HTTP range requests); the revision is pinned.

Contract (verified separately by tests/test_cv_parity.py): renderer r1 (tf_render.py), to_input = bilinear antialiased
resize to 224x224 of the binary image repeated to 3 channels, backbone normalisation, timm num_classes=0 pooled features,
stored as float16.

Governance: fine-tuned weights were trained on TriFrame targets with data to 2014 and early-stopped on 2015-2016, so
ft_* embeddings may only feed evaluation periods from 2017 onward (enforced in nexus_alpha.cv.registry).

Outputs (never committed): artifacts/cv_embeddings/s500/{sample_index.parquet, <backbone>__<view>.npy, manifest.json}
Run: .venv/Scripts/python -I scripts/data/fetch_triframe_embeddings.py [--backbones convnext_tiny,...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, HfFileSystem

from nexus_alpha import paths, seal

FROZEN_DIR = "emb_s500"
FT_DIR = "emb_ft_s500_8ed8458840"
DEFAULT_FROZEN = ["convnext_tiny", "resnet18", "vit_small_patch14_dinov2", "resnet18_random_s0"]
DEFAULT_FT = ["ft_convnext_tiny"]
VIEWS = ("W", "D", "H")
N_CHUNKS = 13


def _read_member(fs, url: str, member: str) -> np.ndarray:
    with fs.open(url, "rb", block_size=8 << 20) as fh:
        zf = zipfile.ZipFile(fh)
        with zf.open(member) as m:
            return np.lib.format.read_array(io.BufferedReader(m), allow_pickle=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=",".join(DEFAULT_FROZEN))
    ap.add_argument("--ft", default=",".join(DEFAULT_FT))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    frozen = [b for b in args.frozen.split(",") if b]
    ft = [b for b in args.ft.split(",") if b]

    api, fs = HfApi(), HfFileSystem()
    rev = api.dataset_info(paths.HF_TRIFRAME_RUNS).sha
    base = f"datasets/{paths.HF_TRIFRAME_RUNS}@{rev}"
    out_dir = paths.ARTIFACTS / "cv_embeddings" / "s500"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) sample ids of every chunk, frozen and fine-tuned folders must agree row for row
    ids = {}
    for c in range(N_CHUNKS):
        a = _read_member(fs, f"{base}/{FROZEN_DIR}/chunk_{c:04d}.npz", "sample_id.npy")
        b = _read_member(fs, f"{base}/{FT_DIR}/chunk_{c:04d}.npz", "sample_id.npy")
        if not np.array_equal(a, b):
            raise RuntimeError(f"chunk {c}: frozen and fine-tuned sample_id order differ")
        ids[c] = a
    idx = pd.DataFrame({"sample_id": np.concatenate([ids[c] for c in range(N_CHUNKS)]).astype(str),
                        "chunk": np.concatenate([np.full(len(ids[c]), c) for c in range(N_CHUNKS)]),
                        "row_in_chunk": np.concatenate([np.arange(len(ids[c])) for c in range(N_CHUNKS)])})
    idx["ticker"] = idx.sample_id.str.split(":").str[0]
    idx["asof_date"] = pd.to_datetime(idx.sample_id.str.split(":").str[1])
    keep = seal.dev_mask(idx["asof_date"]).values
    n_sealed_dropped = int((~keep).sum())
    idx = idx[keep].reset_index(drop=True)
    idx["row"] = np.arange(len(idx))
    if idx.sample_id.duplicated().any():
        raise RuntimeError("duplicate sample_id in TriFrame embeddings")
    idx.to_parquet(out_dir / "sample_index.parquet", index=False)
    print(f"{len(idx):,} sample ids ({n_sealed_dropped} sealed-period rows dropped), {time.time() - t0:.0f} s", flush=True)

    # positions of each kept chunk row in the output
    pos = {c: np.full(len(ids[c]), -1) for c in range(N_CHUNKS)}
    for c, r, o in zip(idx.chunk.values, idx.row_in_chunk.values, idx.row.values):
        pos[c][r] = o

    jobs = [(FROZEN_DIR, b) for b in frozen] + [(FT_DIR, b) for b in ft]
    outputs, manifest = {}, {"source_repo": paths.HF_TRIFRAME_RUNS, "revision": rev, "n_rows": len(idx),
                             "sealed_rows_dropped": n_sealed_dropped, "arrays": {}}

    def work(folder, backbone, view, c):
        a = _read_member(fs, f"{base}/{folder}/chunk_{c:04d}.npz", f"{backbone}|{view}.npy")
        p = pos[c]
        sel = p >= 0
        outputs[(backbone, view)][p[sel]] = a[sel]
        return a.shape[1], a.dtype

    for folder, b in jobs:
        for v in VIEWS:
            first = _read_member(fs, f"{base}/{folder}/chunk_0000.npz", f"{b}|{v}.npy")
            outputs[(b, v)] = np.lib.format.open_memmap(out_dir / f"{b}__{v}.npy", mode="w+",
                                                         dtype=np.float16, shape=(len(idx), first.shape[1]))
            p = pos[0]
            outputs[(b, v)][p[p >= 0]] = first[p >= 0]
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, folder, b, v, c) for v in VIEWS for c in range(1, N_CHUNKS)]
            for f in cf.as_completed(futs):
                f.result()
        for v in VIEWS:
            mm = outputs[(b, v)]
            mm.flush()
            h = hashlib.sha256(np.ascontiguousarray(mm).tobytes()).hexdigest()
            manifest["arrays"][f"{b}__{v}"] = {"folder": folder, "member": f"{b}|{v}.npy", "shape": list(mm.shape),
                                               "dtype": "float16", "sha256": h,
                                               "finite_share": float(np.isfinite(mm[: min(len(mm), 20000)]).mean())}
        print(f"{b}: done, {time.time() - t0:.0f} s", flush=True)
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("manifest written", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

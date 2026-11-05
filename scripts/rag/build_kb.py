"""Ingest the books and rulebooks and build the dense index (BGE-small on CPU).

Outputs (git-ignored, copyrighted text): artifacts/rag/chunks.parquet, pages.parquet, dense_bge_small.npy, manifest.json
Run: .venv/Scripts/python -I scripts/rag/build_kb.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.rag.ingest import ingest_books, ingest_rulebooks
from nexus_alpha.rag.retrieval import DENSE_MODEL, DenseIndex


def main() -> int:
    t0 = time.time()
    out = paths.ARTIFACTS / "rag"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "chunks.parquet").exists() and "--reingest" not in sys.argv:
        chunks = pd.read_parquet(out / "chunks.parquet")
        pages = pd.read_parquet(out / "pages.parquet")
        books = chunks[chunks.kind.isin(["book", "lecture_notes"])]
        rules = chunks[~chunks.kind.isin(["book", "lecture_notes"])]
    else:
        books, pages = ingest_books()
        rules = ingest_rulebooks()
        chunks = pd.concat([books, rules], ignore_index=True)
        chunks.to_parquet(out / "chunks.parquet", index=False)
        pages.to_parquet(out / "pages.parquet", index=False)
    print(f"{len(books)} book chunks from {pages.doc_id.nunique()} documents, {len(rules)} rule chunks; "
          f"{(pages.status != 'OK').sum()} pages without usable text; {time.time() - t0:.0f} s", flush=True)
    vecs = DenseIndex().encode(chunks.text.tolist(), batch=64)
    np.save(out / "dense_bge_small.npy", vecs)
    man = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dense_model": DENSE_MODEL,
           "n_chunks": int(len(chunks)), "by_doc": chunks.doc_id.value_counts().to_dict(),
           "pages": pages.groupby("doc_id").status.value_counts().unstack(fill_value=0).to_dict(orient="index"),
           "section_source": books.groupby("doc_id").section_source.agg(lambda s: s.value_counts().to_dict()).to_dict(),
           "elapsed_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "pages"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

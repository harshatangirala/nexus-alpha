"""Phase 1: integrity and text-extractability audit of the knowledge-base PDFs (no text is written to the repo).

Per document: sha256, page count, PDF metadata, embedded outline (table of contents) depth, and per-page extracted
character counts. Pages with fewer than 200 extracted characters are flagged LOW_TEXT (likely scanned, image-only or
blank); the RAG ingester marks them and never fabricates their content.

Output: data_manifests/books_audit.json
Run: .venv/Scripts/python -I scripts/audit/p1_books_audit.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pymupdf

from nexus_alpha.rag.sources import BOOKS

REPO = Path(__file__).resolve().parents[2]
LOW_TEXT = 200


def main() -> int:
    out = {"audited_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pymupdf": pymupdf.__doc__, "documents": {}}
    for doc_id, meta in BOOKS.items():
        p = Path(meta["path"])
        rec = {"path": str(p), "exists": p.exists(), **{k: v for k, v in meta.items() if k != "path"}}
        if not p.exists():
            out["documents"][doc_id] = rec
            continue
        rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        rec["bytes"] = p.stat().st_size
        try:
            doc = pymupdf.open(p)
        except Exception as e:
            rec["open_error"] = repr(e)
            out["documents"][doc_id] = rec
            continue
        rec["pages"] = doc.page_count
        rec["encrypted"] = doc.is_encrypted
        rec["pdf_metadata"] = {k: v for k, v in (doc.metadata or {}).items() if v}
        toc = doc.get_toc(simple=True)
        rec["toc_entries"] = len(toc)
        rec["toc_max_level"] = max((t[0] for t in toc), default=0)
        rec["toc_head"] = [[lvl, title[:80], page] for lvl, title, page in toc[:12]]
        chars = []
        for page in doc:
            chars.append(len(page.get_text("text").strip()))
        low = [i + 1 for i, c in enumerate(chars) if c < LOW_TEXT]
        rec["chars_total"] = int(sum(chars))
        rec["chars_per_page_median"] = int(sorted(chars)[len(chars) // 2]) if chars else 0
        rec["low_text_pages"] = len(low)
        rec["low_text_share"] = round(len(low) / max(1, len(chars)), 3)
        rec["low_text_page_numbers_head"] = low[:40]
        out["documents"][doc_id] = rec
        print(f"{doc_id:<34} pages={rec['pages']:>4} toc={rec['toc_entries']:>4} median_chars={rec['chars_per_page_median']:>5} "
              f"low_text={rec['low_text_pages']:>4} ({rec['low_text_share']:.0%})", flush=True)
    (REPO / "data_manifests" / "books_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

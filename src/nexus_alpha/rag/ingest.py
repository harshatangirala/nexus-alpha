"""Ingestion with page-level provenance (master spec section 2).

Books: one record per chunk with doc_id, file sha256, 1-based page number, section heading, chunk id, text, extraction
status and timestamp. Chunks never cross a page, so every citation is an exact page. Section headings come from the
PDF's embedded outline where it exists (`section_source = "pdf_outline"`); otherwise from a conservative heuristic
(a short title-like first line, `"heuristic_first_line"`) or are left empty. Pages with < 200 extracted characters are
recorded with status LOW_TEXT and no chunk: their content is never guessed.

Rulebooks: one chunk per rule (E001..X120) with file, line and rule id, plus one chunk per family profile.
Copyrighted book text is stored only under artifacts/ (git-ignored) and is never published.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pandas as pd

from .sources import BOOKS, RULEBOOKS

LOW_TEXT = 200
CHUNK_WORDS = 220
OVERLAP_WORDS = 40
_WS = re.compile(r"[ \t]+")


def _clean(t: str) -> str:
    t = t.replace("­", "").replace("\r", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)                     # de-hyphenate line breaks
    t = _WS.sub(" ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _outline_map(doc) -> dict[int, str]:
    toc = doc.get_toc(simple=True)
    if not toc:
        return {}
    by_page: dict[int, str] = {}
    stack: list[str] = []
    entries = sorted(((p, lvl, title.strip()) for lvl, title, p in toc if p and p > 0), key=lambda x: (x[0], x[1]))
    cur = ""
    j = 0
    for page in range(1, doc.page_count + 1):
        while j < len(entries) and entries[j][0] <= page:
            _, lvl, title = entries[j]
            stack = stack[: lvl - 1] + [title]
            cur = " > ".join(stack[-2:])
            j += 1
        by_page[page] = cur
    return by_page


def _heuristic_heading(text: str) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if 3 <= len(first) <= 80 and not first.endswith(".") and sum(ch.isdigit() for ch in first) < len(first) / 2:
        if first.isupper() or first.istitle():
            return first
    return ""


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out, step = [], size - overlap
    for i in range(0, len(words), step):
        out.append(" ".join(words[i: i + size]))
        if i + size >= len(words):
            break
    return out


def ingest_books(books: dict = BOOKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pymupdf
    chunks, pages = [], []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for doc_id, meta in books.items():
        p = Path(meta["path"])
        if not p.exists():
            pages.append({"doc_id": doc_id, "page": None, "status": "MISSING_FILE"})
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        doc = pymupdf.open(p)
        outline = _outline_map(doc)
        for i, page in enumerate(doc):
            pno = i + 1
            raw = page.get_text("text")
            text = _clean(raw)
            status = "OK" if len(text) >= LOW_TEXT else "LOW_TEXT"
            if outline:
                section, src = outline.get(pno, ""), "pdf_outline"
            else:
                h = _heuristic_heading(text)
                section, src = h, ("heuristic_first_line" if h else "none")
            pages.append({"doc_id": doc_id, "page": pno, "status": status, "chars": len(text), "section": section,
                          "section_source": src})
            if status != "OK":
                continue
            for j, ch in enumerate(chunk_words(text)):
                chunks.append({"chunk_id": f"{doc_id}:p{pno:04d}:c{j}", "doc_id": doc_id, "kind": meta["kind"],
                               "page": pno, "section": section, "section_source": src, "text": ch,
                               "n_words": len(ch.split()), "doc_sha256": sha, "source_file": p.name,
                               "extraction": "pymupdf get_text('text')", "ingested_utc": ts})
    return pd.DataFrame(chunks), pd.DataFrame(pages)


def ingest_rulebooks() -> pd.DataFrame:
    from ..rules.catalog import build_catalog
    cat = build_catalog(RULEBOOKS["entry_rulebook_150"], RULEBOOKS["exit_rulebook_120"])
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = []
    for r in cat["rules"]:
        text = f"{r['rule_id']} {r['title']}. {r['definition']}"
        if not r["fields_are_boilerplate"]:
            text += f" Evidence: {r['required_evidence']} Decision use: {r['decision_use']} Validation: {r['validation']}"
        rows.append({"chunk_id": f"rule:{r['rule_id']}", "doc_id": f"rulebook_{r['book']}", "kind": "rule",
                     "page": None, "line": r["source_line"], "rule_id": r["rule_id"],
                     "section": f"{r['family_code']} {r['family_name']}", "section_source": "markdown_heading",
                     "text": text, "n_words": len(text.split()), "source_file": r["source_file"], "ingested_utc": ts})
    for f in cat["family_profiles"]:
        text = " ".join(f"{k}: {v}" for k, v in f.items() if k not in ("book", "source_file", "source_line"))
        rows.append({"chunk_id": f"profile:{f['first']}-{f['last']}", "doc_id": f"rulebook_{f['book']}",
                     "kind": "rule_profile", "page": None, "line": f["source_line"], "rule_id": None,
                     "section": f"{f['first']}-{f['last']} {f['family']}", "section_source": "markdown_heading",
                     "text": text, "n_words": len(text.split()), "source_file": f["source_file"], "ingested_utc": ts})
    return pd.DataFrame(rows)

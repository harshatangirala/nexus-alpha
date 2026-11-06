"""Citations for the explanation layer. RAG maps observed evidence (rule events, setups, triggers) to sourced rule text
and book passages; it never changes a probability. Each citation carries document, page (books) or line (rulebooks),
retrieval method and score, plus a snippet of at most 200 characters (copyrighted text is not reproduced at length).
If nothing relevant is found the list says so explicitly.
"""
from __future__ import annotations

import pandas as pd

SNIPPET = 200
# Concept index: (document, term regex). The cited page is the page of that document where the term occurs most often
# (ties -> earliest page), like looking a term up in a book's index. Deterministic and auditable.
FAMILY_TOPICS = {
    "TREND_PULLBACK_LONG": ("tsam", r"pullback|retracement"),
    "TREND_PULLBACK_SHORT": ("tsam", r"pullback|retracement"),
    "BREAKOUT_LONG": ("ta_workbook", r"breakout"),
    "BREAKDOWN_SHORT": ("ta_workbook", r"breakdown|breaks? (below|through) support"),
    "RANGE_SUPPORT_LONG": ("complete_guide_to_trading", r"\bsupport\b"),
    "RANGE_RESISTANCE_SHORT": ("complete_guide_to_trading", r"\bresistance\b"),
}
TRIGGER_TOPICS = {("ENGULFING_CONFIRMATION", "LONG"): ("nison_candlesticks", r"bullish engulfing"),
                  ("ENGULFING_CONFIRMATION", "SHORT"): ("nison_candlesticks", r"bearish engulfing"),
                  ("REJECTION_CANDLE", "LONG"): ("nison_candlesticks", r"\bhammer"),
                  ("REJECTION_CANDLE", "SHORT"): ("nison_candlesticks", r"shooting star"),
                  ("STRUCTURE_BREAK", "LONG"): ("tsam", r"swing high|breakout"),
                  ("STRUCTURE_BREAK", "SHORT"): ("tsam", r"swing low|breakdown")}


def concept_page(kb, doc: str, regex: str):
    import re
    c = kb.chunks[kb.chunks.doc_id == doc]
    if c.empty:
        return None
    pat = re.compile(regex, re.I)
    hits = c.text.map(lambda t: len(pat.findall(t)))
    if hits.max() == 0:
        return None
    page_hits = hits.groupby(c.page).sum()
    best_page = page_hits[page_hits == page_hits.max()].index.min()
    row = c[(c.page == best_page)].loc[hits[c.page == best_page].idxmax()].copy()
    row["method"], row["score"] = "concept_index", float(page_hits.max())
    return row


def _cite(r: pd.Series, why: str) -> dict:
    return {"why": why, "doc_id": r.doc_id, "page": None if pd.isna(r.get("page")) else int(r.page),
            "line": None if pd.isna(r.get("line", None)) else int(r.line), "rule_id": r.get("rule_id"),
            "section": r.section, "method": r.method, "score": float(r.score),
            "snippet": (r.text[:SNIPPET] + "...") if len(r.text) > SNIPPET else r.text}


def citations_for(kb, ev, setups: list[dict], triggers: dict) -> list[dict]:
    out = []
    rule_mask = kb.chunks.kind.eq("rule").to_numpy()
    rule_ids = sorted({rid for s in setups for rid in s.get("rule_ids", [])}
                      | {rid for t in triggers.values() for rid in t.get("rule_ids", [])})
    for rid in rule_ids:
        r = kb.search("rule_lookup", rid, k=1, mask=rule_mask)
        if len(r) and r.rule_id.iloc[0] == rid:
            out.append(_cite(r.iloc[0], f"rule {rid} fired in the detectors"))
    topics = [FAMILY_TOPICS[s["family"]] + (s["family"],) for s in setups if s["family"] in FAMILY_TOPICS]
    topics += [TRIGGER_TOPICS[(t["trigger"], side)] + (f"{t['trigger']} ({side.lower()})",)
               for side, t in triggers.items() if (t.get("trigger"), side) in TRIGGER_TOPICS]
    for doc, regex, label in topics:
        row = concept_page(kb, doc, regex)
        if row is not None:
            out.append(_cite(row, f"background for {label} (page where the term occurs most)"))
    if not out:
        out.append({"why": "no setup, trigger or pattern fired, so no rule or book passage is relevant", "doc_id": None})
    return out


def load_kb():
    import numpy as np
    from .. import paths
    from .retrieval import KnowledgeBase
    rag = paths.ARTIFACTS / "rag"
    if not (rag / "chunks.parquet").exists():
        return None
    return KnowledgeBase(pd.read_parquet(rag / "chunks.parquet"), np.load(rag / "dense_bge_small.npy"))

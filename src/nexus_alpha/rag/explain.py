"""Citations for the explanation layer. RAG maps observed evidence (rule events, setups, triggers) to sourced rule text
and book passages; it never changes a probability. Each citation carries document, page (books) or line (rulebooks),
retrieval method and score, plus a snippet of at most 200 characters (copyrighted text is not reproduced at length).
If nothing relevant is found the list says so explicitly.
"""
from __future__ import annotations

import pandas as pd

SNIPPET = 200
FAMILY_TOPICS = {
    "TREND_PULLBACK_LONG": ("pullback to a rising moving average in an uptrend", "tsam"),
    "TREND_PULLBACK_SHORT": ("rally to a falling moving average in a downtrend", "tsam"),
    "BREAKOUT_LONG": ("breakout above resistance with volume confirmation", "ta_workbook"),
    "BREAKDOWN_SHORT": ("breakdown below support", "ta_workbook"),
    "RANGE_SUPPORT_LONG": ("buying near support at the bottom of a trading range", "complete_guide_to_trading"),
    "RANGE_RESISTANCE_SHORT": ("selling near resistance at the top of a trading range", "complete_guide_to_trading"),
}
TRIGGER_TOPICS = {"ENGULFING_CONFIRMATION": ("engulfing pattern reversal", "nison_candlesticks"),
                  "REJECTION_CANDLE": ("hammer long lower shadow reversal", "nison_candlesticks"),
                  "STRUCTURE_BREAK": ("price breaking above a prior swing high", "tsam")}


def _cite(r: pd.Series, why: str) -> dict:
    return {"why": why, "doc_id": r.doc_id, "page": None if pd.isna(r.get("page")) else int(r.page),
            "line": None if pd.isna(r.get("line", None)) else int(r.line), "rule_id": r.get("rule_id"),
            "section": r.section, "method": r.method, "score": float(r.score),
            "snippet": (r.text[:SNIPPET] + "...") if len(r.text) > SNIPPET else r.text}


def citations_for(kb, ev, setups: list[dict], triggers: dict, method: str = "bm25") -> list[dict]:
    out = []
    rule_mask = kb.chunks.kind.eq("rule").to_numpy()
    rule_ids = sorted({rid for s in setups for rid in s.get("rule_ids", [])}
                      | {rid for t in triggers.values() for rid in t.get("rule_ids", [])})
    for rid in rule_ids:
        r = kb.search("rule_lookup", rid, k=1, mask=rule_mask)
        if len(r) and r.rule_id.iloc[0] == rid:
            out.append(_cite(r.iloc[0], f"rule {rid} fired in the detectors"))
    topics = [FAMILY_TOPICS[s["family"]] + (s["family"],) for s in setups if s["family"] in FAMILY_TOPICS]
    topics += [TRIGGER_TOPICS[t["trigger"]] + (t["trigger"],) for t in triggers.values() if t.get("trigger") in TRIGGER_TOPICS]
    for query, doc, label in topics:
        mask = kb.chunks.doc_id.eq(doc).to_numpy()
        r = kb.search(method, query, k=1, mask=mask)
        if len(r) and r.score.iloc[0] > 0:
            out.append(_cite(r.iloc[0], f"background for {label}"))
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

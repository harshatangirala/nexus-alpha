"""Evaluate retrieval architectures on the curated query set (docs/RAG_DESIGN_AND_EVALUATION.md).

Book queries: retrieval restricted to the target document (as the explanation layer does when it knows the topic) and
also over the whole corpus. Metrics: Recall@1/5/10 (any gold page in the top k), MRR, nDCG@10 (binary gains),
page-citation accuracy@1 and median latency. Rule queries: hit@1/@5 and MRR for the gold rule chunk.
Output: reports/rag/rag_eval.json, reports/rag/rag_eval_gold_pages.json
Run: .venv/Scripts/python -I scripts/rag/eval_rag.py
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.experiments import Run
from nexus_alpha.rag.retrieval import KnowledgeBase

METHODS = ["bm25", "dense", "hybrid_rrf", "parent_child", "hybrid_rerank"]


def ndcg(hits: list[bool], k: int = 10) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits[:k]) if h)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, max(1, sum(hits)))))
    return dcg / ideal if ideal else 0.0


def main() -> int:
    q = json.loads((paths.REPO_ROOT / "data_manifests" / "rag_eval_queries.json").read_text(encoding="utf-8"))
    rag = paths.ARTIFACTS / "rag"
    chunks = pd.read_parquet(rag / "chunks.parquet")
    vecs = np.load(rag / "dense_bge_small.npy")
    kb = KnowledgeBase(chunks, vecs)
    run = Run("rag_eval", {"queries": q["version"], "methods": METHODS, "n_book": len(q["book_queries"]),
                           "n_rule": len(q["rule_queries"])}, inputs={"chunks": rag / "chunks.parquet"})
    gold = {}
    for b in q["book_queries"]:
        c = chunks[chunks.doc_id == b["doc"]]
        pat = re.compile(b["gold_regex"], re.I)
        by_page = c.groupby("page").text.apply(lambda s: sum(len(pat.findall(t)) for t in s))
        gold[b["id"]] = sorted(int(p) for p, n in by_page.items() if n >= b["min_hits"])
    results = {"gold_page_counts": {k: len(v) for k, v in gold.items()}, "book": {}, "rule": {}}
    usable = [b for b in q["book_queries"] if gold[b["id"]]]
    results["book_queries_with_gold"] = len(usable)
    for scope in ("within_document", "whole_corpus"):
        for m in METHODS:
            rows = []
            for b in usable:
                mask = (chunks.doc_id == b["doc"]).to_numpy() if scope == "within_document" else (~chunks.kind.isin(["rule", "rule_profile"])).to_numpy()
                try:
                    r = kb.search(m, b["query"], k=10, mask=mask)
                except Exception as e:                     # e.g. reranker download blocked: recorded, not hidden
                    results.setdefault("errors", {})[f"{scope}|{m}"] = f"{type(e).__name__}: {str(e)[:200]}"
                    rows = []
                    break
                hits = [(d == b["doc"]) and (int(p) in gold[b["id"]]) for d, p in zip(r.doc_id, r.page.fillna(-1))]
                rr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
                rows.append({"id": b["id"], "r1": hits[0], "r5": any(hits[:5]), "r10": any(hits[:10]), "mrr": rr,
                             "ndcg10": ndcg(hits), "p5": sum(hits[:5]) / 5, "latency_ms": float(r.latency_ms.iloc[0])})
            if rows:
                df = pd.DataFrame(rows)
                results["book"][f"{scope}|{m}"] = {"recall@1": df.r1.mean(), "recall@5": df.r5.mean(), "recall@10": df.r10.mean(),
                                                   "precision@5": df.p5.mean(), "mrr": df.mrr.mean(), "ndcg@10": df.ndcg10.mean(),
                                                   "page_citation_accuracy@1": df.r1.mean(),
                                                   "latency_ms_median": df.latency_ms.median(), "n": len(df)}
                print(f"{scope:<16} {m:<14} R@1 {df.r1.mean():.3f} R@5 {df.r5.mean():.3f} MRR {df.mrr.mean():.3f} "
                      f"nDCG@10 {df.ndcg10.mean():.3f} {df.latency_ms.median():.0f} ms", flush=True)
    rule_mask = chunks.kind.isin(["rule"]).to_numpy()
    for m in ["bm25", "dense", "hybrid_rrf", "rule_lookup"]:
        rows = []
        for rq in q["rule_queries"]:
            r = kb.search(m, rq["query"], k=10, mask=rule_mask)
            ids = list(r.rule_id)
            pos = ids.index(rq["rule_id"]) + 1 if rq["rule_id"] in ids else None
            rows.append({"hit1": pos == 1, "hit5": pos is not None and pos <= 5, "mrr": 1 / pos if pos else 0.0})
        df = pd.DataFrame(rows)
        results["rule"][m] = {"hit@1": df.hit1.mean(), "hit@5": df.hit5.mean(), "mrr": df.mrr.mean(), "n": len(df)}
        print(f"rules {m:<14} hit@1 {df.hit1.mean():.3f} hit@5 {df.hit5.mean():.3f} MRR {df.mrr.mean():.3f}", flush=True)
    out = paths.REPO_ROOT / "reports" / "rag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag_eval.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    (out / "rag_eval_gold_pages.json").write_text(json.dumps(gold, indent=1), encoding="utf-8")
    best = max((k for k in results["book"] if k.startswith("within")), key=lambda k: results["book"][k]["mrr"])
    run.finish("COMPLETED", f"best within-document method by MRR: {best} ({results['book'][best]['mrr']:.3f}); "
               f"rule lookup MRR {results['rule']['rule_lookup']['mrr']:.3f}", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

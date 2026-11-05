"""Retrieval architectures compared in docs/RAG_DESIGN_AND_EVALUATION.md:

1. dense            - BGE-small embeddings, cosine similarity (deterministic baseline)
2. bm25             - Okapi BM25 (k1 1.5, b 0.75), implemented here so it is transparent and testable
3. hybrid_rrf       - reciprocal rank fusion of bm25 and dense (k = 60)
4. hybrid_rerank    - hybrid top-30 re-scored by a cross-encoder (ms-marco-MiniLM-L-6-v2)
5. parent_child     - hybrid chunk scores aggregated to their parent section/page (RAPTOR-lite: no LLM summaries)
6. rule_lookup      - exact rule-id match, then BM25 over rule definitions (structured rule retrieval)

All retrieval is extractive: results are source chunks with document, page/line and score. Nothing is generated, so a
citation can never point at text the source does not contain.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an the of and or to in on for with by is are was were be been it its this that these those as at from
into than then so such can may not no but if when which who what how why where there their they them he she his her we our
you your i do does did has have had will would should could also more most other some any each per via about over under""".split())
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(t: str) -> list[str]:
    return [w for w in TOKEN.findall((t or "").lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.len = np.array([len(t) for t in self.toks], float)
        self.avg = self.len.mean() if len(self.len) else 0.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.postings = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for w in tf:
                self.postings[w].append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.toks))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class DenseIndex:
    def __init__(self, model_name: str = DENSE_MODEL):
        self.model_name = model_name
        self._tok = self._mdl = None

    def _load(self):
        if self._mdl is None:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._mdl = AutoModel.from_pretrained(self.model_name).eval()

    def encode(self, texts: list[str], batch: int = 32, max_len: int = 384) -> np.ndarray:
        import torch
        self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self._tok(texts[i: i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                h = self._mdl(**enc).last_hidden_state[:, 0]          # BGE uses the CLS vector
                out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._tok = self._mdl = None

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._mdl is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        with torch.no_grad():
            enc = self._tok([query] * len(passages), passages, padding=True, truncation=True, max_length=384,
                            return_tensors="pt")
            return self._mdl(**enc).logits.squeeze(-1).numpy()


def rrf(rank_lists: list[np.ndarray], k: int = 60, n: int | None = None) -> np.ndarray:
    n = n or max(len(r) for r in rank_lists)
    s = np.zeros(n)
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            s[idx] += 1.0 / (k + pos + 1)
    return s


class KnowledgeBase:
    """Holds chunks plus the indexes; `search(method, query, k)` returns a ranked frame with provenance and timing."""

    def __init__(self, chunks: pd.DataFrame, dense_vectors: np.ndarray | None = None, dense: DenseIndex | None = None):
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = BM25(self.chunks.text.tolist())
        self.dense = dense or DenseIndex()
        self.vecs = dense_vectors
        self.reranker = None
        self.parent = (self.chunks.doc_id + "|" + self.chunks.page.astype("Int64").astype(str)).to_numpy()
        self.rule_rows = {r: i for i, r in enumerate(self.chunks.get("rule_id", pd.Series(dtype=object))) if isinstance(r, str)}

    def _dense_scores(self, q: str) -> np.ndarray:
        if self.vecs is None:
            self.vecs = self.dense.encode(self.chunks.text.tolist())
        qv = self.dense.encode([QUERY_PREFIX + q])[0]
        return self.vecs @ qv

    def search(self, method: str, query: str, k: int = 10, mask: np.ndarray | None = None) -> pd.DataFrame:
        t0 = time.perf_counter()
        n = len(self.chunks)
        valid = np.ones(n, bool) if mask is None else mask
        if method == "rule_lookup":
            ids = re.findall(r"\b[EX]\d{3}\b", query.upper())
            s = self.bm25.scores(query)
            for rid in ids:
                if rid in self.rule_rows:
                    s[self.rule_rows[rid]] += 1e6
        elif method == "bm25":
            s = self.bm25.scores(query)
        elif method == "dense":
            s = self._dense_scores(query)
        elif method in ("hybrid_rrf", "hybrid_rerank", "parent_child"):
            b = np.where(valid, self.bm25.scores(query), -np.inf)
            d = np.where(valid, self._dense_scores(query), -np.inf)
            s = rrf([np.argsort(-b)[:200], np.argsort(-d)[:200]], n=n)
            if method == "hybrid_rerank":
                top = np.argsort(-np.where(valid, s, -np.inf))[:30]
                if self.reranker is None:
                    self.reranker = Reranker()
                rs = self.reranker.score(query, self.chunks.text.iloc[top].tolist())
                s = np.full(n, -np.inf)
                s[top] = rs
            elif method == "parent_child":
                agg = pd.Series(s).groupby(self.parent).max()
                s = s + 0.5 * agg.reindex(self.parent).to_numpy()   # child score plus its parent's best child score
        else:
            raise ValueError(method)
        s = np.where(valid, s, -np.inf)
        top = np.argsort(-s)[:k]
        res = self.chunks.iloc[top].copy()
        res["score"] = s[top]
        res["rank"] = np.arange(1, len(top) + 1)
        res["latency_ms"] = (time.perf_counter() - t0) * 1000
        res["method"] = method
        return res

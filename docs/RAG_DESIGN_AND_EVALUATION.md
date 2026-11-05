# RAG design and evaluation

## Role (master spec 9.1)

The trained model produces probabilities. RAG retrieves sourced technical-analysis concepts and rule text for the explanation layer. It cannot change a probability: the inference pipeline computes and calibrates the forecast before any retrieval runs, and retrieval output is written only to `evidence.rag_sources`. Whether structured rule features add predictive value is a separate model ablation (E03 `a17_full_rules`), never a property of RAG.

## Knowledge base (`scripts/rag/build_kb.py`, `nexus_alpha/rag/ingest.py`)

| Source | Ingestion |
|---|---|
| 8 PDFs (2,814 pages) | PyMuPDF text per page; de-hyphenation; chunks of 220 words with 40-word overlap that never cross a page, so each citation is an exact page; section from the PDF outline (TSaM, Dobelman-Williams) or a conservative first-line heuristic (marked) or empty |
| Pages with < 200 extracted characters | status LOW_TEXT, no chunk, content never guessed (120 pages; 35 of them are image slides in the Kanu Jain lecture) |
| 270 rules + 27 family profiles | one chunk per rule with file, line number and rule id |

Provenance per chunk: document id, source file, sha256 of the file, page, section and its source, chunk id, extraction method, ingestion time; embedding model BAAI/bge-small-en-v1.5 (CLS, L2-normalised; queries prefixed per BGE instructions). Book text stays in `artifacts/rag/` (git-ignored); the app shows at most 200-character snippets.

## Architectures compared

Chosen subset (not every named RAG variant): dense, BM25, hybrid (reciprocal rank fusion, k = 60), hybrid + cross-encoder rerank (ms-marco-MiniLM-L-6-v2 on the top 30), parent-child (chunk score plus its page's best child score; a RAPTOR-style tree without LLM summaries), and structured rule lookup (exact rule-id match, then BM25 over rule text). Agentic, self-reflective and corrective RAG were not built: there is no generator in the loop (retrieval is extractive), so they have nothing to correct, and the spec asks for them only with a measured benefit. A concept graph was not built; rule families and required inputs already give the grouping.

## Evaluation (`scripts/rag/eval_rag.py`, query set `data_manifests/rag_eval_queries.json`)

40 book queries written as paraphrases that avoid the concept's key term; gold pages = pages of the target document where the key term regex occurs at least `min_hits` times. 35 queries have gold pages (5 have none in the extracted text: walk-forward in TSaM, triangles and flags in the TA workbook, efficient frontier in Dobelman-Williams, Ito's lemma; their concepts may be phrased differently or sit on low-text pages). 20 rule queries paraphrase one rule each.

**Book queries, retrieval inside the target document (35 queries)**

| Method | Recall@1 | Recall@5 | MRR | nDCG@10 | Median latency |
|---|---|---|---|---|---|
| BM25 | 0.457 | 0.857 | 0.632 | 0.671 | 9 ms |
| Dense (BGE-small) | 0.429 | 0.743 | 0.558 | 0.598 | 35 ms |
| Hybrid RRF | 0.371 | 0.886 | 0.592 | 0.665 | 52 ms |
| Parent-child | 0.371 | 0.886 | 0.591 | 0.664 | 58 ms |
| Hybrid + rerank | **0.514** | 0.857 | **0.657** | **0.691** | 2,044 ms |

Over the whole corpus (document unknown) every method is weaker: MRR 0.405 (BM25), 0.355 (dense), 0.400 (hybrid), 0.433 (rerank).

**Rule queries (20)**: BM25 and rule lookup hit@1 0.85, hit@5 0.95, MRR 0.896; dense hit@1 0.70; hybrid 0.75.

Page-citation accuracy@1 equals Recall@1 here (a hit is a correct document and page).

## Reading the results honestly

* Gold pages are defined lexically, which favours BM25; paraphrased queries limit but do not remove that bias. Dense retrieval may be under-credited.
* Some gold sets are large (the Nison book mentions "hammer" on 51 pages), which makes recall easy for those queries.
* Parent-child aggregation added nothing measurable; it is kept as an option, not the default.
* Choice for the app: BM25 within the topic's document for speed (the explanation layer knows which book covers a setup or pattern), rule lookup for rule ids; hybrid + rerank is available on demand when the 2-second latency is acceptable.
* Answer faithfulness is guaranteed by construction (extractive snippets with their page), not measured: nothing is generated.

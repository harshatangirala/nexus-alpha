# Implementation status

Status words: **passed** (ran and passed), **done** (built and exercised), **partial**, **not run**, **blocked**, **not applicable**. Updated 7 November 2026.

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 0 | Paths, git state, hashes, target folder | passed | `data_manifests/source_inventory.json`, `triframe_copy_comparison.json` |
| 1 | Source audit | done | `docs/PROJECT_AUDIT.md` |
| 1 | OHLCV coverage by ticker/timeframe (sealed) | passed | `docs/DATA_COVERAGE_AUDIT.md` |
| 1 | News schema and true date coverage | passed | max date 2025-09-06; `data_manifests/news_date_coverage.json` |
| 1 | SEC, books, rulebooks, checkpoints | done | `data_manifests/sec_audit.json`, `books_audit.json`, `model_registry/` |
| 2 | Specification and pre-registration (before fusion) | done | `docs/PROJECT_SPECIFICATION.md`, commit `1b17168` |
| 2 | Architecture, point-in-time, labels, fusion, RAG, risk docs | done | `docs/` |
| 3 | Sealed readers, calendar, indicators, labels, splits | passed | tests; label parity 99.32% with TriFrame; indicator parity < 1e-9 |
| 3 | News selection, dedup, availability; SEC availability | passed | tests; 648,468 articles fetched for 91,936 samples |
| 3 | Leakage tests | passed | see `docs/VALIDATION_AND_BACKTESTING.md` |
| 4 | CV branch: renderer and embedding parity, 5 backbones | passed | `reports/verification/cv_parity.json` |
| 4 | Structured branch (E01), chart branch (E02) | done | `docs/EXPERIMENT_LOG.md`, `reports/RESULTS.md` |
| 4 | Text branch (classical, CPU) | done | `data_manifests/text_features_summary.json` |
| 4 | Transformer text encoders (FinBERT, E16), chart fine-tuning on NEXUS labels | partial | FinBERT run in E16 (`docs/V2_RESULTS.md`); other encoders and chart fine-tuning not run, GPU profile in `configs/` |
| 5 | RAG ingestion with page provenance; 6 retrieval methods; evaluation | done | `docs/RAG_DESIGN_AND_EVALUATION.md` |
| 6 | Five fusion variants, ablations, missing-modality tests | done | E03 |
| 7 | Calibration on 2019, registered hypotheses, walk-forward, backtest | done | E03, E05, E04 |
| 7 | Sealed final evaluation | blocked | needs the owner's decision and key (by design) |
| 8 | Rule catalogue (270), executable detectors, decision engine | done | `model_registry/rule_catalog.csv`, `tests/test_rules.py`, `tests/test_trading.py` |
| 8 | Attribution (leave-one-modality-out, chart occlusion) | done | inference pipeline |
| 8 | Grad-CAM | not run | occlusion used instead (model-agnostic, CPU-feasible) |
| 9 | Streamlit app on the real bundle | done | `app/streamlit_app.py`, `tests/test_app.py` |
| 9 | Colab orchestrator, pinned requirements, run profiles | done | `notebooks/`, `scripts/run_pipeline.py`, `requirements-colab.txt` |
| 9 | Colab execution | not run | no GPU session in this work |
| A | Chief audit: 17 findings, 4 High/Medium bugs fixed with regression tests | done | `docs/CHIEF_AUDIT_REPORT.md` |
| A | Fusion optimisation (LR sweep, cosine, 5 seeds, landscape) | done | E06 |
| A | LightGBM TPE tuning with per-iteration and learning curves | done | E07 |
| A | Convexity certificate for the logistic head | passed | E09, `reports/e09_convexity_certificate.json` |
| A | Final model: blend, calibration comparison, refit, bundle v1.1 | done | E08, `docs/MODEL_CARD.md` |
| A | Placebo and planted-leak controls | passed | E10 |
| A | Backtest v1.1 on the fixed engine | done | `reports/backtest/e04_backtest_v1.1.json` |
| A | Quantile coverage of the deployed bundle; return ranges in the app | passed | `scripts/verify/quantile_coverage.py` |
| A | Visual audit report (loss and validation curves, ROC/PR, reliability, confusion, by-year, landscape, equity) | done | `reports/NEXUS_Audit_Report.html`, app tab "L Audit" |
| 10 | Automated tests | see `reports/test_run.txt` | |
| 10 | Model card, limitations, results report | done | `docs/MODEL_CARD.md`, `docs/LIMITATIONS.md`, `reports/RESULTS.md` |
| v3 | Qlib Alpha158 re-implementation, parity with pyqlib 0.9.7 | done | `src/nexus_alpha/indicators/alpha158.py`, `reports/verification/alpha158_qlib_parity.json` |
| v3 | E25 Alpha158 benchmark (pre-registered) | done | `docs/V3_RESEARCH.md` (E25) |
| v3 | v3_alpha refit, live-parity check (Yahoo vs vendor), frozen, in the forward test at 5D-1Y | done | `registry/frozen_models/v3_alpha.json`, `reports/verification/alpha158_live_parity.json`, D-049 |
| App | Mission Control tab, mission-control theme, Plotly charts, Trade Desk plan chart | done | `app/mission_control.py`, `tests/test_dashboard.py` |
| App | TradingAgents LLM committee (live, on demand, logged; E27 forward scoring) | done | `src/nexus_alpha/llm/committee.py`, app tab M, `tests/test_committee.py` |
| App + forward test | OpenBB option chains (tab N) and the nightly implied-volatility snapshot for E26 | done | `scripts/openbb/options_chains.py`, `src/nexus_alpha/options.py`, `scripts/shadow/iv_snapshot.py`, `tests/test_options.py`, `tests/test_iv_benchmark.py` |
| — | GitHub repository | done, **public** | `harshatangirala/nexus-alpha` (made public by the owner; data, caches, weights and secrets are never committed) |

## Next steps, in order

1. Owner review of results and of what to freeze; then (optionally) the single sealed evaluation, reported as semi-confirmatory.
2. GPU profile: transformer text encoders (FinBERT/BGE/ChronoBERT) as a pre-registered comparison against the classical features; native-resolution chart embeddings.
3. Forward shadow test: running since 2026-11-05 for v1.1, v2, market_e19 and v3_fund, and since 2026-11-09 for v3_alpha (`docs/FORWARD_SHADOW_TEST.md`); it is the only clean test of direction and needs about 120 sessions.
4. Pre-register for the next study: an economic direction gate, rolling recalibration (or none), and the v1.1 blend as the frozen model.

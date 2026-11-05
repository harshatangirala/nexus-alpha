# NEXUS Alpha

**Neural Exchange for Unified Signal Intelligence.** A point-in-time multimodal research system for US stocks that combines weekly, daily and hourly chart images, technical indicators, financial news and SEC filings, forecasts calibrated UP / SIDEWAYS / DOWN probabilities for six horizons (1, 2, 5, 10, 20 and 21 sessions), explains them with sourced evidence, and passes them to a separate, deterministic risk engine that outputs LONG, SHORT, NO TRADE or NO SIGNAL with entry, stop, targets and size.

Research and paper-trading decision support only. Nothing here connects to a broker.

NEXUS builds on two finished projects and leaves both untouched: **TriFrame CV** (chart images and 33 numeric features, 979 stocks) and the **SEC filings price-impact study** (10-K, 10-Q and 8-K text). It reuses their verified data, chart embeddings, fine-tuned weights, filing timestamps and conventions after checking each contract (renderer byte-identical, embeddings cosine ≥ 0.99998, labels 99.3% identical, indicators within 1e-9).

## Results in brief

Development data: 236,520 decisions on 500 stocks, 2003-2022 (train 2003-2015, selection 2017-18, calibration 2019, test 2020-22). The test years were already read by TriFrame, so they are a historical diagnostic. Full tables: [`reports/RESULTS.md`](reports/RESULTS.md).

1. **"Will it move" is predictable; "which way" barely is.** Every model beats the class prior in log loss, mostly by separating SIDEWAYS from a move (AUROC about 0.58-0.61), a volatility effect. Up-versus-down AUROC on 2020-22 is 0.47-0.52 across models and horizons (0.50-0.51 for the selected model). Over 15 walk-forward years (2008-2022) a structured LightGBM improved log loss in 15 of 15 years at 1 and 5 days, with a small but persistent direction AUROC of 0.517 (p < 0.01 across years); at 20 days it swings from 0.43 to 0.57.
2. **Multimodal fusion adds nothing measurable.** None of the five pre-registered hypotheses (tri-frame, news + SEC, indicators, trainable fusion vs late fusion, calibration) is supported after Holm correction. A numbers-only model matches or beats the full model. Charts alone carry some information (better than random-weight controls, hourly view best), news and filings alone almost none.
3. **No trading edge.** No policy beat equal-weight buy-and-hold (Sharpe 0.53) on 2020-22; all intervals include zero, deflated Sharpe ≈ 0; the rule-only technical policy lost money. The app attaches this evidence to every trade plan it shows.
4. **Retrieval works for explanation.** Knowledge-base retrieval over 8 books and 270 rules finds the right rule first 85% of the time and the right book page in the top 5 for 86% of queries; RAG never changes a probability (tested).

These replicate the two parent projects (TriFrame: images add ~0 to volatility R²; SEC study: models know whether a stock moves, not which way) on a new, six-horizon, multimodal design.

## What is in the repository

| Path | Contents |
|---|---|
| `src/nexus_alpha/` | package: seal, data, indicators, targets, splits, CV, NLP, rules, fusion, calibration, evaluation, trading, RAG, explainability, inference |
| `app/streamlit_app.py` | dashboard (forecast, charts, news, SEC, indicators, explanation, trade planner, diagnostics, backtest, knowledge base, configuration) |
| `scripts/` | audits, data builds, experiments E01-E05, RAG build and evaluation, verification, report compiler, pipeline orchestrator |
| `notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb` | Colab front end (Drive persistence, Colab Secrets, run profiles) |
| `configs/` | YAML settings (tested against the code) |
| `tests/` | 67 tests: seal, leakage, labels, splits, news timing, rules, fusion masking, calibration, trading, schema, inference, app |
| `docs/` | audit, coverage, specification and pre-registration, point-in-time policy, labels, architecture, fusion, RAG, risk engine, registry, validation, decision log, experiment log, model card, limitations, user guide, status |
| `data_manifests/`, `model_registry/`, `reports/`, `experiments/` | audit outputs, catalogues, compiled results, run manifests |

Not in the repository, by licence: price data, chart caches, embeddings, article or book text, model weights and bundles (`artifacts/`, git-ignored).

## Run it

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md). In short: create the environment from `requirements-colab.txt`, `pip install -e .`, run `pytest`, then `streamlit run app/streamlit_app.py`. The data stages need the TriFrame data store, the SEC study outputs and a Hugging Face login with access to the private TriFrame repositories and the gated news corpus.

## Governance

* The period from 2023-01-03 is sealed and read only through a single-use, owner-keyed, hash-chained unlock; it was not opened. It is not a virgin test (TriFrame opened it twice; it is the SEC study's spent locked test).
* The specification and hypotheses were committed before any fusion model was trained (`docs/PROJECT_SPECIFICATION.md`, commit `1b17168`). Every run, including failures, is in `docs/EXPERIMENT_LOG.md`.
* Survivor-only universe from one vendor; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

Author: Krishna Harsha Tangirala. AI assistance: built with Claude Code (Anthropic); every number in the documents is generated from saved artifacts.

# NEXUS Alpha

**Neural Exchange for Unified Signal Intelligence.** A point-in-time multimodal research system for US stocks that combines weekly, daily and hourly chart images, technical indicators, financial news and SEC filings, forecasts calibrated UP / SIDEWAYS / DOWN probabilities for six horizons (1, 2, 5, 10, 20 and 21 sessions), explains them with sourced evidence, and passes them to a separate, deterministic risk engine that outputs LONG, SHORT, NO TRADE or NO SIGNAL with entry, stop, targets and size.

Research and paper-trading decision support only. Nothing here connects to a broker.

NEXUS builds on two finished projects and leaves both untouched: **TriFrame CV** (chart images and 33 numeric features, 979 stocks) and the **SEC filings price-impact study** (10-K, 10-Q and 8-K text). It reuses their verified data, chart embeddings, fine-tuned weights, filing timestamps and conventions after checking each contract (renderer byte-identical, embeddings cosine ≥ 0.99998, labels 99.3% identical, indicators within 1e-9).

## Results in brief

Development data: 236,520 decisions on 500 stocks, 2003-2022:
* train 2003-2015
* selection 2017-18
* calibration 2019
* test 2020-22

The test years were already read by TriFrame and have been re-scored several times here, so they are a historical diagnostic. Full tables are in [`reports/RESULTS.md`](reports/RESULTS.md), figures in `reports/NEXUS_Audit_Report.html`, and findings in [`docs/CHIEF_AUDIT_REPORT.md`](docs/CHIEF_AUDIT_REPORT.md).

1. **"Will it move" is predictable; "which way" barely is.** Every model beats the class prior in log loss, mostly by separating SIDEWAYS from a move (AUROC about 0.58-0.61), a volatility effect.
   * The deployed model's up-vs-down AUROC on 2020-22 is 0.49-0.51.
   * Over 15 walk-forward years (2008-2022), a structured LightGBM improved log loss in 15 of 15 years at 1 and 5 days, with a small but persistent direction AUROC of 0.517 (p < 0.01 across years). At 20 days it swings from 0.43 to 0.57.
   * Placebo controls show that about a third of the short-horizon gain is market-wide timing, and that there is no leakage: a global label shuffle gains nothing, and a planted leak is found.
2. **Multimodal fusion adds little.**
   * None of the five pre-registered hypotheses is supported after Holm correction: tri-frame, news + SEC, indicators, trainable fusion vs late fusion, and calibration.
   * With five seeds, images, news and filings add 0.0005-0.0012 log loss over numbers alone, about one seed SD.
   * Charts alone carry some information: better than random-weight controls, with the hourly view best.
3. **The audited model is better, not profitable.**
   * The audit fixed 21 issues, including two that changed trading results.
   * The deployed v1.1 model blends 0.6 fusion (5 seeds) + 0.3 logistic (certified global optimum) + 0.1 LightGBM, refit on 2003-2018. Test log loss is 1.0620 vs v1's 1.0662, significant at 1-10 days.
   * In the fixed backtest (E04 v1.1), no policy beat equal-weight buy-and-hold of the same survivors (Sharpe 0.53). The gated live planner had 0.31 at 5D (41 trades) and −0.14 at 20D. All intervals include zero, and deflated Sharpe ≈ 0.
   * The app attaches this evidence to every trade plan it shows.
4. **Retrieval works for explanation.** Knowledge-base retrieval over 8 books and 270 rules finds the right rule first 85% of the time and the right book page in the top 5 for 86% of queries. RAG never changes a probability (tested).

These replicate the two parent projects on a new, six-horizon, multimodal design:
* TriFrame: images add ~0 to volatility R².
* SEC study: models know whether a stock moves, not which way.

## What is in the repository

| Path | Contents |
|---|---|
| `src/nexus_alpha/` | package: seal, data, indicators, targets, splits, CV, NLP, rules, fusion, calibration, evaluation, trading, RAG, explainability, inference |
| `app/streamlit_app.py` | dashboard (forecast with return ranges, charts, news, SEC, indicators, explanation, trade planner, diagnostics, backtest v1/v1.1 with equity curves, knowledge base, configuration, audit figures) |
| `scripts/` | audits, data builds, experiments E01-E10, RAG build and evaluation, verification (parity, quantile coverage), report compilers (`make_results.py`, `audit_report.py`), pipeline orchestrator |
| `notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb` | Colab front end (Drive persistence, Colab Secrets, run profiles) |
| `configs/` | YAML settings (tested against the code) |
| `tests/` | 70 tests: seal, leakage, labels, splits, news timing, rules, fusion masking, training curves, calibration, trading (P&L path, kill-switch), schema, inference (incl. return ranges), app |
| `docs/` | **chief audit report (findings, SWOT, optimisation, results after fixes)**, audit, coverage, specification and pre-registration, point-in-time policy, labels, architecture, fusion, RAG, risk engine, registry, validation, decision log, experiment log, model card, limitations, user guide, status |
| `data_manifests/`, `model_registry/`, `reports/`, `experiments/` | audit outputs, catalogues, compiled results (`reports/RESULTS.md`), the visual audit report (`reports/NEXUS_Audit_Report.html`, 27 figures and tables), run manifests |

Not in the repository, by licence: price data, chart caches, embeddings, article or book text, model weights and bundles (`artifacts/`, git-ignored).

## Run it

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md). In short: create the environment from `requirements-colab.txt`, `pip install -e .`, run `pytest`, then `streamlit run app/streamlit_app.py`. The data stages need the TriFrame data store, the SEC study outputs and a Hugging Face login with access to the private TriFrame repositories and the gated news corpus.

## Governance

* The period from 2023-01-03 is sealed and read only through a single-use, owner-keyed, hash-chained unlock; it was not opened. It is not a virgin test (TriFrame opened it twice; it is the SEC study's spent locked test).
* The specification and hypotheses were committed before any fusion model was trained (`docs/PROJECT_SPECIFICATION.md`, commit `1b17168`). Every run, including failures, is in `docs/EXPERIMENT_LOG.md`.
* Survivor-only universe from one vendor; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

Author: Krishna Harsha Tangirala. AI assistance: built with Claude Code (Anthropic); every number in the documents is generated from saved artifacts.

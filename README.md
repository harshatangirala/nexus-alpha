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

## Status: v1.1, v2, market_e19, v3_fund and v3_alpha are frozen and in the forward test

* **v1.1 is frozen** (`registry/frozen_models/v1.1.json`, SHA-256 of every bundle file) and is in a **forward shadow test**:
  every evening it forecasts all 500 stocks before the next open, into a hash-chained log in `shadow/v1.1/`
  ([`docs/FORWARD_SHADOW_TEST.md`](docs/FORWARD_SHADOW_TEST.md)). A golden test pins its outputs. v2, market_e19, v3_fund and v3_alpha are frozen too and logged by the same nightly run.
* **v2** ([`docs/V2_ARCHITECTURE.md`](docs/V2_ARCHITECTURE.md), pre-registered before any v2 fit) adds:
  * S&P 500 (SPY) and sector, rates, credit and commodity ETF bars at 1H/1D/1W, from the same minute source as TriFrame;
  * VIX, Treasury yields, dollar, oil, gold and copper;
  * Global Macro Database annual data, publication-lagged;
  * FRED/ALFRED real-time vintages;
  * FinBERT text features;
  * every trade length the dashboard offers: intraday (1D), swing of 2-20 days, and 4 weeks, 3 months, 6 months and 1 year.
* **Trade Desk** (first dashboard tab):
  * Inputs: trade type, three chart screenshots and 1-10 texts.
  * Model output: the frozen model's forecast and risk plan.
  * Context: market and macro context.
  * Screenshot check: legend values checked against real bars.
  * Optional Gemini read: live and unvalidated, logged for forward scoring.
* **What v2 has found so far** ([`docs/V2_RESULTS.md`](docs/V2_RESULTS.md)):
  * **Stock-level model.** Market and macro blocks did not beat the v1.1 numeric features on validation once tested
    against that baseline (D-033). v2's tabular component is the v1.1 feature set. Macro reaches v2 only through the
    fusion network's context and a separate market read.
  * **Market level (E19).** SPY technicals plus cross-asset features rank SPY's **20-session direction**: AUROC 0.72 on
    2017-18 and 0.68 on 2020-22. It is frozen as `market_e19`, logged every evening in the forward test, and shown in the
    Trade Desk as a lean, because its 2020-22 probabilities were no better than the base rate.
  * **Screenshots (E17).** Gemini read every legend value of 57 real TradingView screenshots exactly to the cent.
* **Sealed 2023+ evaluation: ready, not opened.** The scorer (`scripts/sealed/run_sealed_eval.py`, CPU, about 3-5
  hours) passed its rehearsal on 2022:
  * features exact;
  * fresh chart embeddings at cosine >= 0.99997 against TriFrame's stored ones;
  * v1.1 and v2 reproduce their stored predictions to 0.0006.

  It scores v1.1, v2 and the E19 market read once. The owner opens it with the commands in
  [`docs/SEALED_EVALUATION_PLAN.md`](docs/SEALED_EVALUATION_PLAN.md) section 7b.
* **Market-regime rules and trading-process tools** from two MIT skill repositories (D-034):
  * O'Neil/IBD rules were tested in E20: they add nothing to the market read, so they are shown as context only.
  * The Trade Desk gained live headlines and 8-Ks, a drawdown circuit breaker, a pre-trade discipline gate and sizing.
* **Fundamental analysis** ([`docs/FUNDAMENTALS.md`](docs/FUNDAMENTALS.md)): SEC XBRL for all 979 tickers, point in
  time from 2009, with 31 pre-registered ratios across valuation, profitability, growth, leverage, efficiency and
  quality (Piotroski F, Altman Z, SUE), a fixed-assumption DCF, P/E, EV/EBITDA, P/S, P/B and P/FCF.
  * **Data audit:** split basis, ADS/foreign filers, share classes, weighted-share and stale-count bugs, and XBRL scale
    errors are all guarded. Every hard check passes, including megacaps within 15% of public values.
  * **Experiments:**
    * E22 (universe ranks) was **not kept**: its placebo also passed.
    * E23 (industry-neutral ranks) was **kept**. Ranking improved at swing and long horizons on 2017-18, but only the
      swing gain replicated on 2020-22.
  * **Forward test:** `v3_fund` (LightGBM on the v1.1 features plus industry-neutral fundamentals) is frozen and
    logged nightly. A pre-registered rank-IC criterion decides whether it is ever shown as a signal.
  * **Trade Desk:** the three statements, all ratios with universe and industry percentiles (live from a nightly
    cross-section), Piotroski and Altman zones, and an adjustable DCF.
  * **Audit:** [`docs/AUDIT_3_4_2026-11-07.md`](docs/AUDIT_3_4_2026-11-07.md).
* **Kronos foundation model** ([`docs/KRONOS.md`](docs/KRONOS.md)): the MIT Kronos model for daily candles,
  vendored at a pinned commit, with weights at pinned revisions and SHA-256.
  * **Sampler:** NEXUS's own path sampler, bit-identical to Kronos's predictor, with an exact cache that makes it
    17-19x faster (2.9 s per stock on CPU). Probabilities use the NEXUS label rule.
  * **Contamination:** its pre-training data covers US daily stock bars from 2000 to June 2024, so it is never a model
    feature. A 2017-18 test (E24, pre-registered) can only rule it out.
  * **E24 result:** on 2017-18, inside its training data, Kronos-small had no ranking skill and unusable probabilities
    (log loss 1.27-1.96 vs about 1.05 for the class prior). It is **not** in the forward test; it stays in the Trade
    Desk as labelled analysis.
  * **Trade Desk:** a fan chart of sampled paths.
* **QVeris MCP** ([`docs/QVERIS.md`](docs/QVERIS.md)): the public discovery endpoint is connected; paid calls need the
  owner's one-time `/mcp` sign-in.
* **Audit 5 (2026-11-10)** ([`docs/AUDIT_5_2026-11-10.md`](docs/AUDIT_5_2026-11-10.md)): end-to-end review of the integrations; findings A-53 to A-62 fixed. Every selection trial is counted in [`docs/TRIAL_LEDGER.md`](docs/TRIAL_LEDGER.md), and forward tests are Holm-corrected (D-051).
* **Audit 6 (2026-11-12)** ([`docs/AUDIT_6_2026-11-12.md`](docs/AUDIT_6_2026-11-12.md)): every requirement and
  resource mapped to code and evidence; findings A-63 to A-68.
  * **The whole pipeline in one notebook:**
    [`notebooks/NEXUS_Alpha_End_to_End.ipynb`](notebooks/NEXUS_Alpha_End_to_End.ipynb) (open it in Colab, A100). It
    contains every step with its code inline:
    * labels, splits and indicators re-derived;
    * logistic regression with a global-minimum certificate;
    * LightGBM and the multimodal fusion network with training and validation curves;
    * ConvNeXt fine-tuned on charts;
    * calibration, all metrics with confusion matrices, bootstrap, placebo and Holm, and the model comparison;
    * the risk engine, RAG and the forward-test chains.
  * **Dashboard:**
    * double-click `RUN_DASHBOARD.cmd`;
    * the new **Compare 1-5** tab;
    * entry, stop and targets drawn with a reason card in every trade view.
* **External integrations (D-046, [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md)).**
  * **Qlib Alpha158 (E25).** Microsoft Qlib's Alpha158 factor library, re-implemented (157 of 158 features) and
    checked against Qlib's own engine: identical definitions on 122,797 stock-days. Tested as **E25** under the pre-registered rule: kept for cross-sectional ranking only (LightGBM long-horizon rank-IC gain +0.0134 [+0.0071, +0.0215] on 2017-18; no placebo passes), with no gain in probabilities and a partial 2020-22 replication. `v3_alpha` (LightGBM on F0 + A158) is frozen and in the forward test from session 2026-11-09, at 5D-1Y only: there its live Yahoo inputs reproduce the vendor inputs it was trained on, and at 1D/2D they do not (D-049). The forward test decides.
  * **Mission Control** (the dashboard's first tab):
    * header counters and an evidence tape;
    * forward-test tracker and system status;
    * an evidence board in which every number quotes its source document (tested);
    * the data flow;
    * a globe of the forward-test companies over Global Macro Database data.

    It uses globe.gl, and the Magic UI and Arwes styles recreated in plain HTML/CSS. Licences are listed in
    [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
  * **TradingAgents** (dashboard tab M) is an on-demand LLM committee on the owner's Gemini key. Its analysts,
    bull/bear debate, trader, risk team and portfolio manager end in a five-tier rating, which is logged publicly
    and scored forward only (E27). It never feeds a model or the risk engine.
  * **OpenBB** (dashboard tab N) shows CBOE's option chains: implied volatility and implied moves next to the
    model's range. A nightly snapshot feeds a pre-registered forward test (E26): does NEXUS's "will it move" add
    anything beyond option prices?
* **Keys the owner sets** (environment variables, never committed):
  * `FRED_API_KEY` for ALFRED vintages;
  * `GEMINI_API_KEY` for the Trade Desk's LLM read.

## What is in the repository

| Path | Contents |
|---|---|
| `src/nexus_alpha/` | package: seal, data, indicators, targets, splits, CV, NLP, rules, fusion, calibration, evaluation, trading, RAG, explainability, inference |
| `app/streamlit_app.py` | dashboard: Mission Control (`app/mission_control.py`, `app/globe_view.py`, `app/ui.py`), Trade Desk with a price chart of the plan, forecast with return ranges, charts, news, SEC, indicators, explanation, trade planner, diagnostics, backtest v1/v1.1 with equity and drawdown curves, knowledge base, configuration, audit figures; theme in `.streamlit/config.toml` |
| `scripts/` | audits, data builds, experiments E01-E10, RAG build and evaluation, verification (parity, quantile coverage), report compilers (`make_results.py`, `audit_report.py`), pipeline orchestrator |
| `notebooks/NEXUS_Alpha_End_to_End.ipynb` | **the presentation notebook**: the whole pipeline with the code inline (built by `scripts/colab/make_end_to_end_notebook.py`; inputs staged by `scripts/colab/stage_notebook_inputs.py`) |
| `notebooks/NEXUS_Alpha_Colab_RunAll.ipynb`, `notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb` | one-pass Colab reproduction of E22/E23/E25; Colab front end (Drive persistence, Colab Secrets, run profiles) |
| `RUN_DASHBOARD.cmd` | double-click to open the dashboard at `http://localhost:8501` |
| `configs/` | YAML settings (tested against the code) |
| `tests/` | 169 tests: seal, leakage, labels, splits, news timing, rules, fusion masking, training curves, calibration, trading (P&L path, kill-switch), schema, inference (incl. return ranges), app, comparison and reason cards, fundamentals (point in time, share basis, vendor data contract), forward-test runner, done marker and keep-awake helper |
| `docs/` | **chief audit report (findings, SWOT, optimisation, results after fixes)**, audit, coverage, specification and pre-registration, point-in-time policy, labels, architecture, fusion, RAG, risk engine, registry, validation, decision log, experiment log, model card, limitations, user guide, status |
| `data_manifests/`, `model_registry/`, `reports/`, `experiments/` | audit outputs, catalogues, compiled results (`reports/RESULTS.md`), the visual audit report (`reports/NEXUS_Audit_Report.html`, 27 figures and tables), run manifests |

Not in the repository, by licence: price data, chart caches, embeddings, article or book text, model weights and bundles (`artifacts/`, git-ignored).

## Run it

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md); for Google Colab see [`docs/COLAB.md`](docs/COLAB.md) (fundamentals, the E22/E23 reproduction and E25 run on a CPU runtime; the nightly forward test stays on the owner's PC). In short: create the environment from `requirements-colab.txt`, `pip install -e .`, run `pytest`, then `streamlit run app/streamlit_app.py`. The data stages need the TriFrame data store, the SEC study outputs and a Hugging Face login with access to the private TriFrame repositories and the gated news corpus.

## Governance

* The period from 2023-01-03 is sealed and read only through a single-use, owner-keyed, hash-chained unlock; it was not opened. It is not a virgin test (TriFrame opened it twice; it is the SEC study's spent locked test).
* The specification and hypotheses were committed before any fusion model was trained (`docs/PROJECT_SPECIFICATION.md`, commit `1b17168`). Every run, including failures, is in `docs/EXPERIMENT_LOG.md`.
* Survivor-only universe from one vendor; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

Use: the code is published for reading and review; no licence is granted (all rights reserved), as for the TriFrame and
SEC repositories. Data and weights are not included (see [`SECURITY.md`](SECURITY.md)). Work that uses the Global Macro
Database features must cite Mueller, Xu, Lehbib and Chen (2025), NBER Working Paper 33714, and follow its Research Use Terms.

Author: Krishna Harsha Tangirala. AI assistance: built with Claude Code (Anthropic); every number in the documents is generated from saved artifacts.

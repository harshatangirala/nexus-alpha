# Phase 1: source audit

Audited 5 November 2026. Every statement below was checked against the files themselves, by a script in `scripts/audit/` whose output is in `data_manifests/` or `model_registry/`. Where the handoff documents and the files disagree, the files win and the difference is listed in section 8.

## 1. Workspace preflight (Phase 0)

| Check | Result | Evidence |
|---|---|---|
| All source paths named in the master prompt exist | Yes, every one (Drive `G:` is not mounted, so the Drive mirror could not be checked) | `scripts/audit/p0_source_inventory.py` |
| Source inventory with hashes | 153 + 153 + 463 + 17 files hashed; 17 large trees summarised by count and size | `data_manifests/source_inventory.json` |
| TriFrame workspace copy vs git copy | **Byte-identical** (153 of 153 files, excluding `.git`, caches) | `data_manifests/triframe_copy_comparison.json` |
| Git state of source repositories | `triframe-cv` clean at `c8a16b1` (untracked: `docs/PROJECT_HANDOFF.md`, `docs/cqo_review/`, both owner-local); `sec-filings-price-impact` clean at `0093b29`, in sync with origin; `us_stocks_ohlcv` local-only repo at `4a91b1b` with uncommitted owner edits (left untouched) | `git status` |
| New project location | `C:\Users\harsh\nexus-alpha\` (did not exist). Chosen over `Downloads\...` because an unrelated `Downloads\CLAUDE.md` would load into any agent session opened under Downloads (it once leaked a wrong affiliation into a paper) | decision D-001 |
| Writes into source trees | None. TriFrame modules imported for verification are loaded with bytecode writing disabled; a before/after listing of `us_stocks_ohlcv/scripts` is identical | `reports/verification/cv_parity.json` |

## 2. TriFrame CV project (charts)

**What it is.** 979 surviving US stocks; daily and regular-session hourly bars 2001-01-02 to 2026-10-09 from one vendor lineage (Finnhub-derived 1-minute data on Hugging Face, no declared licence, plus Yahoo for gaps and splits). Each decision date has three binary **OHLC-bar charts (not candlesticks)**: 104 weekly, 126 daily and 140 hourly bars, 128 px high, 3 px per bar, log-scaled to each window, plus 33 numeric "visible-window" features. Seven ImageNet backbones (ResNet-18, EfficientNet-B0, ViT-B/16, ConvNeXt-T, Swin-T, DINOv2-S, MobileNetV3) and three random-weight controls produce embeddings; tree and linear heads forecast 20-day volatility, a 10% drawdown and 5-day relative direction.

**Verified results that matter here.** Images add no significant pooled R² (+0.003, 95% CI −0.018 to +0.026); a within-month gain (+0.020) was found after the fact and is unconfirmed. No 5-day directional edge (AUC 0.510). Volatility is predictable (test R² ~0.50).

**Reusable, verified components.**

| Component | Status in NEXUS | Verification |
|---|---|---|
| Frozen bars `final/1day|1hour|1week` | Read through NEXUS's sealed readers | coverage audit (section 4) |
| Renderer r1 (`tf_render.py`) | Vendored unchanged as `nexus_alpha/cv/render.py` (+ r1.1 early-close rule) | 20 of 20 re-rendered triplets byte-identical to TriFrame's cached training images |
| Stored s500 embeddings (private HF `triframe-cv-runs`, 236,520 chart sets) | Reused for 5 backbones (frozen ConvNeXt-T, ResNet-18, DINOv2-S, random ResNet-18, chart-fine-tuned ConvNeXt-T) | re-embedding on CPU matches stored vectors: cosine ≥ 0.99998, max relative error ≤ 0.8% (fp16 storage); fine-tuned ConvNeXt 0.03% |
| Fine-tuned ConvNeXt-T weights (private HF `triframe-cv-model-s500`) | Loaded through `nexus_alpha.cv.backbones` | same parity check |
| 33 visible-window features (`feat_visible.py`) | Vendored unchanged as `indicators/triframe_visible.py`; stored values read through TriFrame's sealed `ImageStore` | 40 random samples recomputed, max relative error < 1e-5 |
| Indicator conventions (`tf_data.indicators`) | Re-implemented in `indicators/core.py` | equals the frozen indicator file on 4 tickers to < 1e-9 relative error |
| Holdout guard design | Re-implemented in `nexus_alpha/seal.py` (single-use, hash-chained owner unlock) | 5 tests |

**Not reusable as is.** The 2020-2022 test years and the 2023-2026 period have been read (the pilot and 120-stock runs opened 2023+), so NEXUS cannot call them fresh. Fine-tuned weights saw TriFrame targets through 2014 and early-stopped on 2015-2016, so their embeddings may only be evaluated on 2017 onward (enforced in code). TriFrame's `tests/` are scripts that pytest cannot collect.

## 3. SEC filings project (NLP)

**What it is.** A 25-stage pipeline (S00-S24) testing whether 10-K, 10-Q and 8-K text predicts next-day reactions. The 150-stock S&P mini study is complete and its locked test (2023 to July 2026) is spent. Headline: models predict whether a stock moves (AUC 0.635), not which way (0.494); transformer embeddings add nothing over classical text (L7 − L6 = −0.001).

**Reused.** The 500-firm CPU-stage outputs in `Downloads\sec_inputs_staging\full` (never re-run here): 190,947 events from 500 firms (141,300 8-K, 37,135 10-Q, 12,512 10-K), each with an acceptance time, an assumed public time (+5 min) and a decision-ready time (+35 min in total). Timestamp quality A for 96.2%, B (date only) 2.5%, C (uncertain) 1.3%. 54% of events are filed after the close. Event text (8-K press-release exhibits, MD&A sections; tickers masked as `[TICKER]`) is capped at 12,000 characters. All 500 firms are in the 979-stock universe; **262 of TriFrame's 500 s500 stocks have SEC coverage**. The Loughran-McDonald 1993-2025 dictionary is read in place (licence restricts redistribution). See `data_manifests/sec_audit.json`.

**Not reused.** The SEC models themselves (different target: one-day reaction to an event), the spent locked test, the CARs with the pre-event alpha term (D-079 artifact).

## 4. Price data coverage (development period only)

See `docs/DATA_COVERAGE_AUDIT.md`. In short: 979 tickers, 5,143,695 daily and 35,693,557 hourly bars before the seal; 0 missing sessions, 0 duplicates, 0 OHLC violations; weekly bars equal aggregated daily bars exactly (0 of 1,066,802 weeks differ); daily and hourly bars agree exactly on minute-sourced days. Survivors only.

## 5. News corpus

`Brianferrell787/financial-news-multisource`, revision `c25780f3`, gated ("auto"), licence "other": non-commercial research, no redistribution of article text. **57,114,479 rows in 24 subsets, 1990-01-01 to 2025-09-06** (the true maximum, not "end of 2025"). Six subsets carry explicit ticker tags; 3,947,549 unique articles are tagged with at least one universe ticker. Universe-tagged coverage is thin before 2010 (5-11 thousand articles a year), dense 2010-2023, nearly empty in 2024 (258 articles: FNSPID ends 2024-01-09) and resumes March-September 2025 (Yahoo feed). 2.92 million FNSPID rows labelled minute precision are stamped exactly at midnight UTC; they are treated as day-level. See `data_manifests/news_*.json`.

## 6. Knowledge base

| Document | Pages | Low-text pages | Embedded outline |
|---|---|---|---|
| The Complete Guide to Trading | 116 | 4 | no |
| TSaM | 1,232 | 27 | yes (883 entries) |
| Nison, Japanese Candlestick Charting Techniques | 298 | 20 | no |
| TA workbook | 172 | 7 | no |
| Identifying Chart Patterns (file name spelled "Idenitfying") | 49 | 9 | no |
| Fundamentals of Investments, week 2 lecture (Kanu Jain) | 67 | 35 (slides) | no |
| Essentials of Quantitative Finance | 259 | 4 | no |
| Dobelman and Williams, Quantitative Financial Analytics | 621 | 14 | yes (199) |

Rulebooks: 150 entry rules (E001-E150) and 120 exit rules (X001-X120) in 27 families. 260 of the 270 rules share boilerplate text for evidence, decision use, model fields and validation; only E141-E150 have rule-specific text. The decision-engine, labelling, training, data-contract, Colab and verification specs were read and used. Low-text pages are flagged, never filled in. See `data_manifests/books_audit.json`, `model_registry/rule_catalog.csv`.

## 7. Models and compute

| Asset | Where | Size |
|---|---|---|
| TriFrame s500 bundle (fine-tuned ConvNeXt-T + heads) | private HF `KrishnaHarshaTangirala/triframe-cv-model-s500` | 116 MB |
| TriFrame run outputs incl. all s500 embeddings and 15 fine-tuned checkpoints | private HF `KrishnaHarshaTangirala/triframe-cv-runs` | 28.9 GB |
| Cached locally | ResNet-18, Swin-T, DINOv2-S; FinBERT, FinBERT-tone, SEC-BERT, BGE-base, ChronoBERT-2010, Qwen2.5-0.5B | 0.04 to 1 GB each |

This machine: AMD Ryzen 5 5500U (6 cores), 15.3 GB RAM, **no NVIDIA GPU**, 29 GB free disk, about 5 MB/s from Hugging Face. Consequences: re-embedding 236k chart sets is not feasible locally (hence reuse); transformer text encoders over ~650k articles would take many hours on CPU (hence classical text features in the local profile and GPU adapters for Colab); training is limited to fusion heads on cached features.

## 8. Discrepancies between the handoff/prompt and the files

| Statement | What the files show |
|---|---|
| "900+ stocks spanning more than 25 years" of weekly, daily and hourly data | 979 stocks; raw bars 2001-01-02 to 2026-10-09 (25.8 years), but development data must stop at 2022-12-30, so **22 years** are usable; 764 tickers span all of 2001-2022; the s500 sample uses 500 stocks |
| "Candlestick charts" | TriFrame's charts are **OHLC bars**, binary, without overlays |
| News "ends in 2025" | Ends **2025-09-06**; universe news is nearly absent in 2024 |
| 2023-2026 is a sealed holdout | Sealed for NEXUS development, but **already opened twice** by TriFrame and spent by the SEC study: semi-confirmatory at best |
| The calendar covers the data span | TriFrame's calendar ends 2026-02-27; extended with `pandas_market_calendars` after an exact overlap check |
| README (1).md lists docs 07-10, a risk-management spec and a `src/trading_ml/` scaffold | **Not supplied**; not used |
| Rulebooks list | Candlestick definitions, strategies, breakout and test-vector specs exist in `TriFrame_Workspace/rulebooks_and_specs/` but were not in the supplied list; the candle detectors use their own documented numeric definitions |
| SEC "900+ stock universe" | The SEC study covers 500 firms (150 in the finished mini study) |

## 9. Private data and credentials

A cached Hugging Face login on this machine can read the private TriFrame repositories and the gated news corpus; the token was never printed or copied. The holdout key was not requested and never created. `.env.example` lists variable names only.

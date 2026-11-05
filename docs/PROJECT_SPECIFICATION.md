# NEXUS Alpha: research specification and pre-registration

Version 1.0, 5 November 2026. This file fixes the questions, targets, samples, splits, metrics, selection rule and holdout protocol **before** the multimodal fusion models are trained. It is committed to git before experiment E03 runs; the commit timestamp is the registration record. Changes after that commit are listed in the change log at the end, never edited in place.

## 1. Research question

Does a point-in-time multimodal model that combines tri-timeframe chart representations, structured technical indicators, financial news and SEC filings improve calibrated multi-horizon stock-direction forecasts, and net-of-cost decisions, relative to strong unimodal, structured-data and simpler-fusion baselines?

A null or negative answer is a valid result and will be reported as such.

## 2. What the two earlier projects already established (priors, not results of this project)

| Prior finding | Source |
|---|---|
| Chart images add ~0 to pooled 20-day volatility R² over 33 numeric features (+0.003, 95% CI −0.018 to +0.026); a within-month gain (+0.020) was found after the fact and is unconfirmed | TriFrame paper, `triframe-cv/README.md` |
| No 5-day directional edge (relative-direction AUC 0.510, CI 0.500 to 0.521) | TriFrame s500 |
| Filing-based models predict whether a stock moves (AUC 0.635), not which way (0.494) | SEC study locked test |
| Transformer text embeddings add nothing over classical text (L7 − L6 = −0.001) | SEC study |
| Naive market-wide features hurt TriFrame's volatility model | TriFrame CQO audit 08 |

These priors shape the design (risk-first, NO TRADE default, matched-sample tests) and are why a direction claim needs strong evidence.

## 3. Sample, timing and targets

* **Sample frame:** TriFrame's 500-stock "s500" chart sets (236,520 stock-dates, 2001-10-22 to 2022-12-01, one decision every ~10 sessions per stock, staggered), reused so the stored chart embeddings can be used without re-rendering. 500 of 979 survivor stocks. See `docs/DATA_COVERAGE_AUDIT.md`.
* **Timing (D-003):** as-of session t = last completed daily bar. Decision timestamp = 09:00 New York on session t+1. Bars are complete through the close of t; news and SEC filings count if available by the decision time; simulated entry = official open of t+1.
* **Targets (D-005, `docs/LABEL_ENGINEERING.md`):** for h ∈ {1, 2, 5, 10, 20, 21 ("1M")} sessions, R_h = close[t+h] / open[t+1] − 1; τ_h = max(0.5·σ_t·√h, 10 bps) with σ_t the EWMA (λ 0.94) daily volatility at t; UP if R_h > τ_h, DOWN if R_h < −τ_h, else SIDEWAYS. TriFrame's frozen 5-session rule generalised; not tuned for balance. Price-only (no dividends).

## 4. Splits (D-006)

| Block | As-of dates | Use |
|---|---|---|
| train | 2003-01-01 to 2015-12-31 | fitting |
| inner | 2016 | early stopping only |
| valid | 2017-01-01 to 2018-12-31 | model and configuration selection |
| calib | 2019 | probability calibration and decision thresholds |
| test | 2020-01-01 to 2022-12-01 | development test, reported once per frozen configuration |
| sealed | from 2023-01-03 | final evaluation only with the owner's key (section 8) |

Each block is purged against the next: a row is kept only if its 21-session label end plus a 5-session embargo is before the next block's first session (`nexus_alpha.splits`, tested). The 2020-2022 test years were already read by TriFrame for its own targets, so the test block is a **historical diagnostic**, not a virgin test. A walk-forward protocol (annual test folds 2008-2022, expanding training window) is used for the cheap tabular models to include crisis years.

**Fine-tuned chart weights:** the chart-fine-tuned ConvNeXt-Tiny embeddings were trained on TriFrame targets with data to 2014 and early-stopped on 2015-2016. They may be evaluated only on 2017 onward (`cv.store.assert_eval_allowed`), which the primary split respects.

## 5. Primary endpoint and selection rule

* **Primary metric:** mean multiclass log loss over the six horizons, on calibrated probabilities (calibrator fitted on the calib block only).
* **Selection rule:** among candidate configurations, choose the lowest mean **validation** log loss (2017-2018). Within 0.0005 of the best, choose the simpler model (fewer modalities, then fewer parameters). Test metrics never enter selection.
* **Calibration:** none / temperature / vector scaling chosen per horizon on chronological halves of the calib block, then refitted on the whole block. Isotonic is excluded (TriFrame audit: it worsened log loss).

## 6. Hypotheses (one family, Holm-corrected, month-block bootstrap, one-sided α = 0.05)

Each is a paired difference in mean six-horizon log loss on the test block, positive = improvement.

| ID | Comparison | Rows |
|---|---|---|
| H1 tri-frame value | images W+D+H vs the best single view, same backbone and head | all test rows |
| H2 NLP value | images + numbers vs images + numbers + news + SEC | test rows with news present or SEC covered |
| H3 indicator value | images only vs images + numbers | all test rows |
| H4 fusion value | selected trainable fusion vs late fusion, same modalities | all test rows |
| H5 calibration value | calibrated vs uncalibrated selected model (log loss and ECE) | all test rows |
| H6 RAG role | RAG is explanation-only; it cannot change probabilities (tested: identical outputs with RAG on/off). Retrieval quality is measured separately. Rule features as a model input are ablation 17 | n/a |

Secondary (reported, not part of the family): per-horizon log loss, up-vs-down AUROC among non-sideways outcomes, sideways AUROC, macro-F1, balanced accuracy, ECE, per-year results, modality-dropout and missing-modality tests, the 17 ablations of master spec 11.6, and the economic tests below.

## 7. Economic evaluation (secondary)

Pre-specified policies, backtested with `nexus_alpha.trading.backtest` conventions (entry at next open, gap-through stops at the open, stop-first ambiguity, 10/15/20 bps round trip, 0.5%/yr borrow):

* P0 rule-only technical baseline (weekly permission + daily setup + hourly trigger, structural stop, first-obstacle target);
* P1 model-only: trade when |P(UP) − P(DOWN)| ≥ θ with ATR stops; θ ∈ {0.05, 0.10, 0.15, 0.20} chosen on the valid block's net Sharpe and frozen;
* P2 NEXUS policy: P1 signal AND a same-side rule setup and trigger, structural levels;
* benchmark: equal-weight buy-and-hold of the survivor universe (survivorship-biased; labelled as such).
Reported with Sharpe (month-block CI), PSR, deflated Sharpe using the number of policies and thresholds tried, and PBO where enough configurations exist. The live decision engine additionally requires validated directional skill per horizon before it can output LONG or SHORT.

## 8. Sealed final evaluation

The sealed period (2023-01-03 onward) is never read during development. One evaluation after the owner provides `NEXUS_HOLDOUT_UNLOCK` = sha256 of the frozen configuration file; the access is logged in a hash-chained file. Honest status: 2023-2026 was opened twice by TriFrame (pilot, s120) and is the SEC study's spent locked test, and the news corpus is nearly empty in 2024 (258 articles) and ends 2025-09-06. Results on it will be called **semi-confirmatory**. A clean test needs forward data after the freeze date.

## 9. Disclosed deviations

1. Experiments E01 (structured baselines) and E02 (chart branch) printed test-block metrics before this registration was committed. They involve baseline and single-branch models, not the fusion comparisons registered here, and their selection used validation only; they are reported as exploratory.

## Change log

| Date | Change |
|---|---|
| 2026-11-05 | v1.0 registered |

# Validation and backtesting methods

Results are compiled in `reports/RESULTS.md` (generated from artifacts by `scripts/report/make_results.py`). This file describes how they were produced.

## Leakage and contract tests (automated, `tests/`)

| Requirement | Test |
|---|---|
| OHLCV/timeframe alignment; weekly = aggregated daily | `p1_ohlcv_coverage.py` (0 of 1,066,802 weeks differ); `test_weekly_window_partial_flag_uses_calendar` |
| No future bars in features or charts | `test_structured_features_ignore_future_bars`, `test_pivots_and_patterns_ignore_future_bars`, renderer parity with TriFrame's audited images |
| News publication-time filtering | `test_article_after_decision_time_is_excluded`, `test_day_level_article_...`, inference test with a later article |
| SEC acceptance-time filtering | availability rule in `data/sec.py` (documented; uses the SEC study's tested timestamps) |
| Duplicate news handling | `test_near_duplicate_headlines_cluster_to_the_earliest_copy` |
| Indicator warm-up and causality | `test_indicator_warmup_and_ranges`, `test_indicators_match_frozen_triframe_file_on_dev_period` |
| Purge and embargo | `test_primary_blocks_are_purged_and_embargoed`, `test_verify_integrity_catches_a_leaking_row`, `test_walk_forward_folds_cover_crises` |
| Sealed period never read | `test_sealed_readers_never_return_sealed_rows`, `test_sealed_dates_are_refused`, seal unit tests |
| Checkpoint loading and adapter shapes | `scripts/verify/cv_parity.py` (all five backbones), `test_shapes_and_probabilities` |
| Missing-modality handling | `test_masked_modality_has_no_influence` (all variants), `test_prediction_contract_and_missing_news` |
| Probability sums and finite outputs | `check_probs`, schema validators, inference tests |
| Calibration separation | calibrators fitted on the calib block only (E03 code); `test_calibrators_reduce_log_loss_of_an_overconfident_model` |
| Stop direction, R:R, sizing, costs | `tests/test_trading.py` |
| Inference schema | `test_schema_rejects_bad_plans_and_probabilities` |
| RAG cannot change probabilities | `test_future_article_is_excluded_and_rag_never_changes_probabilities` |
| Configs match code | `tests/test_configs.py` |

## Metrics

Per horizon: accuracy, balanced accuracy, macro and per-class precision/recall/F1 with supports and predicted counts, confusion matrix, one-vs-rest AUROC and PR-AUC with base rates, log loss, multiclass Brier, top-label ECE (15 equal-width bins) and class-wise ECE, plus the AUROC of P(UP) − P(DOWN) for UP versus DOWN among non-sideways outcomes (the direction question). Return heads: pinball loss and interval coverage functions exist (`evaluation/metrics.py`); the quantile heads are trained but their outputs are not yet surfaced as expected returns (left None in the app).

## Inference

* Month-block bootstrap (calendar months of the as-of date, 1,000 resamples for hypotheses): keeps same-date cross-sections and overlapping label windows together; never row-level i.i.d. inference.
* Paired differences for model comparisons on identical resampled rows; registered hypotheses Holm-corrected.
* Walk-forward: each of the 15 test years is one observation (t-tests across years).
* Backtests: Sharpe with a month-block bootstrap interval, probabilistic Sharpe against 0, deflated Sharpe with the number of validation configurations tried, PBO by combinatorially symmetric cross-validation over all configurations.

## Backtest conventions (D-010)

Decision before the open of t+1; market entry at that open, cancelled on a gap beyond the chase limit or beyond the stop; gap-through stops fill at the open; stop-first when a bar touches both levels; time exit at the close of the horizon's last session; 10/15/20 bps round trip plus 0.5%/yr borrow on shorts; risk 0.5% of equity per trade, at most 10% per position and 100% gross. Benchmark: equal-weight buy-and-hold of the survivor universe (biased upwards by survivorship). Not modelled: partial fills, market impact, borrow availability, halts beyond what the bars show, intraday stop timing beyond the stop-first rule.

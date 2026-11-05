# Point-in-time policy

Hard validity rules. Each has the code that enforces it and the test that checks it.

## 1. The sealed period (D-002)

* First sealed session: **2023-01-03** (inherited from TriFrame's `HOLDOUT_START`). Bar readers push `date < 2023-01-03` (daily), `timestamp < 2023-01-03 00:00 New York` (hourly) and `last_day < 2023-01-03` (frozen weekly) into the parquet scan, so sealed rows are never returned (`nexus_alpha.data.bars`; test `test_sealed_readers_never_return_sealed_rows`).
* A development sample needs its as-of date AND every used label end before the seal (`seal.dev_mask`); labels whose window would need sealed bars are NaN and masked per horizon.
* The news index and SEC events are filtered to `available_utc < 2023-01-03 05:00 UTC` for development.
* Unlocking: owner sets `NEXUS_HOLDOUT_CONFIG` (frozen config file) and `NEXUS_HOLDOUT_UNLOCK` (its sha256). One unlock per configuration hash; a second needs `NEXUS_HOLDOUT_REUSE_REASON`; every unlock is appended to `governance/holdout_access_log.csv` with a hash chain (`seal.unlock`, 3 tests). NEXUS never calls TriFrame's own unlock, which would write into the source project.
* Status: sealed for NEXUS, but not virgin (opened by TriFrame twice; SEC locked test spent). Any result on it is "semi-confirmatory".

## 2. Decision timing (D-003)

| Item | Convention |
|---|---|
| As-of session t | last completed daily bar |
| Decision timestamp | 09:00 New York on session t+1 (`calendar.decision_time_utc`) |
| Daily and hourly inputs | bars of sessions up to and including t (all complete at 09:00 on t+1) |
| Weekly inputs | completed weeks only for weekly indicators; the week in progress is used only as (a) the last bar of the weekly chart image, built from days up to t (as TradingView shows it and as TriFrame trained), and (b) the week-to-date return feature |
| News and SEC | count only if available by the decision timestamp |
| Simulated entry | official open of session t+1 |
| Label window | [open t+1, close t+h] |

Using 09:00 rather than the previous close lets after-hours earnings filings and overnight news count, and they are known before the entry price forms. Test: `test_offset_and_decision_time`.

## 3. News availability (D-004, `nexus_alpha.data.news`)

| Timestamp type | available_at |
|---|---|
| minute precision | publication time (UTC) + 15 minutes processing latency |
| "minute" stamped exactly 00:00:00 UTC | treated as day precision (2.92 million such rows) |
| day precision (date known, time not) | the next calendar day 08:00 UTC = end of the day in US Pacific time |

The corpus's own `date_trading` anchor is kept but not used. Lookback: articles with `decision − 5 days < available_at ≤ decision`. Exact duplicates (normalised text hash) are dropped; near-duplicate headlines (3-word shingle Jaccard ≥ 0.8 within 2 days) form a cluster represented by its earliest copy, so syndication never makes a story look newer or more numerous. At most 7 cluster representatives per sample (the app also accepts 1-7). Tests: `test_minute_day_and_midnight_availability`, `test_article_after_decision_time_is_excluded`, `test_day_level_article_dated_asof_day_is_usable_next_morning_but_not_same_day`, `test_near_duplicate_headlines_cluster_to_the_earliest_copy`.

Mapping: explicit ticker tags only, normalised (`BRK-B` → `BRK.B`); never from company names. Retrospective labels, sentiment columns or later metadata in the corpus are not used as inputs.

## 4. SEC availability (D-007, `nexus_alpha.data.sec`)

available_at = the SEC study's `decision_time` = acceptance + 5 minutes (assumed public) + 30 minutes (processing). For timestamp quality C (derived/uncertain) the later of that and the filing date at 18:05 New York is used (a filing dated D was accepted by 17:30 on D under EDGAR's cut-off), so an uncertain time can only make a filing later. Quality B (date only) already sits at 17:30 + 35 minutes on the filing date. Period-end and report dates are never used as availability times.

## 5. Features and charts

* Indicators at bar t use bars ≤ t (`test_structured_features_ignore_future_bars`: corrupting every bar after the last as-of date leaves all features identical).
* Weekly features at a mid-week as-of date equal those of the previous Friday's completed week (`test_weekly_features_use_completed_weeks_only`).
* Hourly features exist only if the as-of session itself has hourly bars; they are never carried forward from an older session.
* Swing pivots become known only after their confirmation bars and are stored at the confirmation bar (`test_pivot_is_known_only_after_confirmation`).
* Chart images: re-rendered triplets are byte-identical to TriFrame's training images, whose own look-ahead audit perturbed every future bar in 318 cases with 0 changes.

## 6. Fitting and evaluation

Scalers, PCA, imputation, TF-IDF vocabularies and SVD axes are fitted on training rows only (text: articles and filings available before 2016-01-01). Early stopping uses the 2016 `inner` block, model selection the 2017-2018 `valid` block, calibration and decision thresholds the 2019 `calib` block. Splits are purged by the 21-session label end plus a 5-session embargo (`test_primary_blocks_are_purged_and_embargoed`, `test_verify_integrity_catches_a_leaking_row`). Fine-tuned chart embeddings may only be evaluated from 2017 (`cv.store.assert_eval_allowed`).

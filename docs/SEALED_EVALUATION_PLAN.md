# Sealed evaluation plan (2023-01-03 onward): v1.1 and v2, opened once

Status: **DRAFT, written 2026-11-06 before v2 is frozen and before any sealed row has been read.** It becomes final when
v2 is frozen. At that point the bundle hashes, code commit and scorer hash are filled in (section 8), and this file's
SHA-256 goes into the frozen config that the owner's unlock checks. Changes after finalisation are logged in
`docs/DECISION_LOG.md` and void the "pre-specified" label for whatever they touch.

Owner decision D-018: the seal is opened **once**, after v2 is frozen, scoring v1.1 and v2 together.

## 1. What this evaluation can and cannot show

* It is **semi-confirmatory, not virgin.**
  * TriFrame opened 2023+ twice for its own models (`docs/LIMITATIONS.md` item 17), although its s500 run never did
    (TriFrame `docs/PROJECT_HANDOFF.md`).
  * 2023+ is the SEC study's spent locked test.
  * Our design choices were made by people who lived through 2023-2026 markets.
  * No NEXUS model, feature choice or threshold has been fitted, tuned or selected on any 2023+ row; the read-time seal
    enforced this (`src/nexus_alpha/seal.py`).
* Only the forward shadow test (`docs/FORWARD_SHADOW_TEST.md`) is fully clean.
* A null result here is informative. A positive result is encouraging but must still be confirmed forward.

## 2. What is scored

| Model | Bundle | Horizons |
|---|---|---|
| v1.1 | `registry/frozen_models/v1.1.json` (frozen 2026-11-06) | 1D, 2D, 5D, 10D, 20D, 1M (21 sessions) |
| v2 | `registry/frozen_models/v2.json` (frozen at E14) | 1D-20D, 3M (63), 6M (126), 1Y (252) |

Both bundles are used exactly as frozen, with their file hashes verified before scoring. There is **no** refitting,
recalibration, re-thresholding, feature re-selection or skill-gate change. The frozen backtest policies (E04 for v1.1,
E15 for v2) are used unchanged.

## 3. Period and sample

* As-of sessions run from 2023-01-03 to the last session whose label window for that horizon ends on or before the last
  bar in the source files. The end date is read once, at opening, and is reported.
* Universe: the same 500 survivor stocks as development. This is survivorship-biased, as in development, and is stated.
* Sample design: TriFrame's s500 rule, about one as-of date every 10 sessions per stock with staggered offsets, continued
  from each stock's last development offset. A second, denser read (every session) is reported for the backtest only.
* Labels: the training label rule unchanged (`targets.py` for v1.1, `targets_v2.py` for v2).

## 4. Inputs, built once after the unlock, by code frozen before it

| Modality | Source for 2023+ | Note |
|---|---|---|
| Bars, numeric features, indicators | `us_stocks_ohlcv` final parquet (known only as counts until opening) | Same builders as development |
| Chart embeddings | **Must be produced at opening.** TriFrame's s500 sets end 2022-12-01 and s500 never opened its holdout | Render with the training renderer (byte parity verified) and embed with the frozen backbones. CPU or Colab GPU (owner's choice at opening; the sealed bars then pass through the owner's Drive) |
| News | Local corpus | Coverage collapses in 2024 (258 universe-tagged articles, against 207k in 2023 and 79k in 2025). The models' missing-news mask handles this; results are also reported per year |
| SEC filings | SEC study inputs | Same availability rule (acceptance + 35 min) |
| Market and macro (v2) | ETF minute bars 2023+ (download stopped at 2022-12 by design), Yahoo market series, ALFRED vintages, GMD 2026_09 | The same point-in-time rules as development (`docs/V2_ARCHITECTURE.md` section 2) |
| FinBERT (v2, if kept by E16) | Fixed weights, same PCA | |

Every builder runs from the frozen commit. The number of sealed rows returned is written to the access log.

## 5. Endpoints (pre-specified)

**Primary** (one per horizon group, for each model):
mean log-loss gain against the training class prior, averaged over the group's horizons (v1.1: 1D; 2D-20D; 1M. v2: 1D;
2D-20D; 3M-1Y). The 95% CI comes from a month-block bootstrap (1000 resamples, seed 11). Holm correction is applied
across the three groups per model.

**Key secondary:**
* Up-vs-down AUROC per horizon, with month-block bootstrap CI. This is the direction-skill question.
* v2 minus v1.1 log loss on the shared horizons (1D, 2D, 5D, 10D, 20D) on identical rows.
* Calibration: expected calibration error and a reliability table per horizon.
* 10-90% quantile coverage per horizon (target 0.80).
* Frozen backtests (E04 v1.1, E15 v2) under base, stressed and severe costs. Benchmarks: SPY total return and the
  equal-weight universe. Reported: return, volatility, Sharpe with a deflated Sharpe ratio using the logged trial
  count, maximum drawdown, turnover and trade count.

**Descriptive:**
* Per-year results (2023, 2024, 2025, 2026 to date).
* Results with and without news-covered rows.
* Results by market regime (SPY above or below its 200-day average at the as-of date).

## 6. Decision rules stated now

* The primary result is reported whatever it is, in full, in `reports/sealed/`, including negative results.
* "Direction skill confirmed on the sealed period" is claimed for a horizon only if the AUROC CI lies above 0.5 **and**
  the same horizon passed the validation skill gate.
* No model change motivated by sealed results is ever scored on the sealed period again.
* If a bug is found during scoring:
  * it is fixed, the fix is logged, and both the pre-fix and post-fix results are reported;
  * the post-fix result is labelled "after a sealed-period bug fix".

## 7. Execution protocol

1. At v2 freeze, finalise this file (section 8), commit it, and commit the scorer `scripts/sealed/run_sealed_eval.py`.
2. Write the frozen config `governance/sealed_eval_config.json`. It holds the commit, this plan's SHA-256, both bundle
   record hashes, the scorer's SHA-256 and the planned outputs.
3. **The owner**, in their own terminal, sets:
   * `NEXUS_HOLDOUT_CONFIG` to that file;
   * `NEXUS_HOLDOUT_UNLOCK` to its SHA-256.

   Then they run the scorer. Code never creates or guesses the key.
4. The scorer:
   * verifies the bundles, the plan hash and a clean git tree;
   * calls `seal.unlock` once (hash-chained access log);
   * builds the inputs, scores both models and writes `reports/sealed/`.
5. Results are committed with the access-log row.

## 8. To fill in at v2 freeze

| Item | Value |
|---|---|
| v2 bundle id and record SHA-256 | (E14 run) |
| Code commit | |
| Scorer SHA-256 | |
| Trial count for the deflated Sharpe ratio | (from `experiments/`) |
| Embedding route (CPU / Colab) | owner |

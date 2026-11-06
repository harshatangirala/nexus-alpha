# NEXUS Alpha v2: results (interim, 2026-11-06)

Pre-registration: `docs/V2_ARCHITECTURE.md` section 4. Decisions and disclosures: `docs/DECISION_LOG.md` (D-026 to D-032).

**How to read the numbers:**
* Log loss is multinomial, 3 classes (down / sideways / up, vol-scaled), so lower is better.
* "Gain" is old minus new log loss, so positive means the new model is better.
* Validation is 2017-2018 and is used for selection.
* "Test" is 2020-2022, a **reused diagnostic** that never drives a choice.
* Intervals are 95% month-block bootstrap intervals.

Status: E11, E12, E13 and E19 are complete. E16, E18, E14 (blend and freeze) and E15 (backtest) are running in sequence.

## E11: market and macro blocks in the stock model (tabular)

Nested sets:
* F0: v1.1 numeric features (80);
* F1: + `rel_*` (stock vs market);
* F2: + `mkt_*`;
* F3: + `gmd_*`;
* F4: + `fred_*`.

The placebo permutes whole dates of the date-level blocks. Nine representative horizons (D-026).

Group gains (validation | test), as registered: one-month bootstrap blocks; the long group averages 3M, 6M and 1Y on both splits. D-032's recomputation, which uses horizon-length blocks and only 3M in the validation long group, gives the same keep decisions.

| Comparison | Family | 1D | 2D-20D | 3M-1Y |
|---|---|---|---|---|
| F1 vs F0 (stock-relative) | LightGBM | +0.0001 \| -0.0005 | +0.0010 \| +0.0021 | +0.0000 \| +0.0024 |
| F2 vs F1 (market) | LightGBM | **-0.0069** \| -0.0047 | -0.0087 \| +0.0002 | **-0.0037** \| -0.0040 |
| F3 vs F2 (GMD) | LightGBM | +0.0021 [+0.0001, +0.0043] \| **-0.0033** | -0.0008 \| +0.0030 | +0.0083 \| **-0.0556** |
| F4 vs F3 (FRED) | LightGBM | +0.0003 \| +0.0026 | +0.0056 [+0.0008, +0.0118] \| -0.0107 | +0.0061 \| **-0.0386** |
| F4 vs F0 (all blocks) | LightGBM | **-0.0043** \| -0.0058 | -0.0029 \| -0.0054 | +0.0108 \| **-0.0958** |
| F4 vs F0 (all blocks) | logistic | **-0.0315** \| -0.2046 | **-0.2600** \| -0.3992 | **-0.3275** \| -0.7031 |
| F4 placebo vs F1 | LightGBM | -0.0049 \| -0.0038 | -0.0048 \| -0.0027 | -0.0010 \| -0.0006 |

**Keep rule (as registered, and unchanged under D-032's horizon-length blocks):**
* LightGBM keeps F3 (GMD) and F4 (FRED); every other rung is dropped.
* The placebo passes for both families.
* Carried set (union rule): F0 + GMD + FRED, 138 columns. `mkt_n` is never carried (audit A-24, D-030).

**Reading.**
* The GMD keep rests on a 1D lower bound of +0.0001.
* Both kept blocks lose on the 2020-2022 diagnostic at long horizons.
* Linear models with date-level features overfit date effects severely (2D F4 logistic: 1.138 vs F3 1.059). With about 3,200 training dates, a stock-level model can memorise dates.
* This repeats v1's finding (D-016) with a broader block.

## E19: does market and macro information predict the market itself? (one row per session, SPY's own label)

Nested sets:
* P: training prior;
* M1: SPY technicals;
* M2: + cross-asset market features;
* M3: + GMD;
* M4: + FRED.

Bootstrap blocks are max(1, ceil(h/21)) months. 6M and 1Y have fewer than 4 non-overlapping validation windows, so they are reported only.

* Keep rule: logistic keeps M1, M2 and M4; LightGBM keeps M3. Both placebos sit on the prior.
* **Logistic M2, up-vs-down AUROC per horizon** (validation | test):

  | Horizon | Validation | Test |
  |---|---|---|
  | 1D | 0.541 | 0.572 |
  | 10D | 0.604 | 0.685 |
  | 15D | 0.643 | 0.640 |
  | 20D | **0.720 [0.525, 0.857]** | **0.683 [0.536, 0.814]** |
  | 3M | 0.780 [0.498, 0.959] | 0.567 [0.281, 0.765] |

* Only **20D** passes the D-015 direction gate on validation and replicates on test.
* Even at 20D, the 2020-2022 log loss was no better than the base rate (1.086 vs 1.084). The model ranks direction but its probabilities were overconfident in that period.
* GMD and FRED rungs of the linear model lost 0.2-1.4 nats in 2020-2022 at swing and long horizons (macro extremes of 2020 extrapolated).

**Deployed (D-031):** the SPY 20D direction lean, frozen as `registry/frozen_models/market_e19.json`.
* It is logged daily in the forward shadow test (`shadow/market_e19/`).
* It is shown in the Trade Desk with this evidence and caveat.
* It does not change the stock models.

Live parity (`reports/verification/e19_live_parity.csv`, 2022-09 to 2022-12):
* every feature correlates > 0.98 with development;
* 978 of 979 universe tickers had data;
* the read correlates 0.998 with signs agreeing 100%, with the SPY hourly features held equal (Yahoo keeps 60 days of hourly bars).

First read, session 2026-11-05: P(up / sideways / down) = 0.424 / 0.259 / 0.317.

## E12: v2 fusion (v1 modalities + carried context, 23 heads, 5 seeds)

| Horizon | 1D | 5D | 10D | 20D | 3M | 6M | 1Y |
|---|---|---|---|---|---|---|---|
| Validation log loss | 1.0407 | 1.0637 | 1.0667 | 1.0723 | 1.0744 | 1.0604 | 1.0444 |
| Validation AUROC (up vs down) | 0.527 | 0.531 | 0.552 | 0.556 | 0.608 | 0.614 | 0.585 |
| Test AUROC | 0.510 | 0.528 | 0.554 | 0.533 | 0.475 | 0.497 | 0.390 |

* Best epochs by seed: 1, 6, 2, 1, 2.
* Comparable to the tabular F0 baseline at short horizons (1D 1.0347; 20D 1.0713).
* Its long-horizon validation skill does not survive 2020-2022.

## E13: sequence models over 63 daily bars of the stock and SPY (3 seeds each)

| Horizon | 1D | 5D | 20D | 3M | 1Y |
|---|---|---|---|---|---|
| TCN validation AUROC | 0.491 | 0.512 | 0.501 | 0.486 | 0.495 |
| GRU validation AUROC | 0.513 | 0.532 | 0.530 | 0.549 | 0.516 |
| GRU test AUROC | 0.516 | 0.494 | 0.465 | 0.410 | 0.488 |

* TCN: no skill.
* GRU: slight validation skill at short horizons that does not hold on test.
* TCN best epochs were 1-2: it learned nothing beyond the first epoch.

## Operational notes from this run

* **D-027 (compute only, model unchanged):** logistic fits were about 80x slower under BLAS thread oversubscription (one cell took 4.1 hours). Fixed with one BLAS thread; E11 was restarted from scratch.
* **D-028:** the shadow forecaster hung for six hours on one Yahoo request. It now has per-ticker timeouts, retry passes and a decision-time deadline.
* **Bundle selection bug:** the app loaded "the newest bundle folder", which became the E19 market model. It now loads the frozen v1.1 record, and a regression test covers it.

# NEXUS Alpha v2: results (2026-11-06)

Pre-registration: `docs/V2_ARCHITECTURE.md` section 4. Decisions and disclosures: `docs/DECISION_LOG.md` (D-026 to D-032).

**How to read the numbers:**
* Log loss is multinomial, 3 classes (down / sideways / up, vol-scaled), so lower is better.
* "Gain" is old minus new log loss, so positive means the new model is better.
* Validation is 2017-2018 and is used for selection.
* "Test" is 2020-2022, a **reused diagnostic** that never drives a choice.
* Intervals are 95% month-block bootstrap intervals.

Status: complete. v2 frozen 2026-11-06. E17 has 3 of 22 stocks pending the next Gemini free-tier reset.

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

## E16: text encoders on top of the carried set B (F0 + GMD + FRED)

| Comparison (LightGBM, validation) | 1D | 2D-20D | 3M-1Y |
|---|---|---|---|
| + FinBERT vs B | **+0.0028 [+0.0003, +0.0052]** | -0.0007 [-0.0023, +0.0010] | -0.0008 [-0.0021, +0.0006] |
| + v1 news features vs B | +0.0021 [-0.0008, +0.0051] | -0.0017 [-0.0037, -0.0000] | +0.0049 [+0.0017, +0.0087] |

* The keep rule kept FinBERT (LightGBM, 1D) and the "both" set; logistic kept nothing.
* E16 chose B + FinBERT.

## D-033 baseline guard: is the chosen set better than F0?

* Comparing B + FinBERT with F0 under the D-032 bootstrap, the keep rule **fails for both families**.
* **The v2 tabular component is therefore F0** (the v1.1 numeric features), fitted on all 23 horizons in E18. Its 1D and 2D validation log losses reproduce E11's F0 exactly (1.03472, 1.05250).
* Market and macro blocks reach v2 only through the E12 fusion context (GMD + FRED), weighted by E14's validation blend, and through the separate E19 market read.
* FinBERT on top of F0 is untested: it is a candidate for a later version (new trial).

## E17: Gemini reads the legends of real TradingView screenshots

19 of 22 stocks (57 screenshots, 228 O/H/L/C values):
* every value read, **all exact to the cent**, on all three timeframes;
* ticker named correctly on 94.7% of the screenshots it read.

The other three stocks (TTEC, URI, WERN) are scheduled for 2026-11-07 13:45 IST, after the free-tier reset. Reads are cached, so a re-run never re-spends.

This validates D-019's screenshot check: legend values typed in, or read by Gemini, are compared with real bars.

The free tier allows 20 requests per model per day, and on this key only `gemini-3.6-flash` answered. E17 used that day's quota, so the Trade Desk's Gemini read was unavailable until the next reset.

## E14: v2 blend, refit on 2003-2018, calibration on 2019 (frozen as `registry/frozen_models/v2.json`)

| Group | LightGBM (F0) | Logistic (F0) | Fusion | TCN | GRU | Validation log loss |
|---|---|---|---|---|---|---|
| 1D | 0.3 | 0.5 | 0 | 0 | 0.2 | 1.03306 |
| 2D-20D | 0.2 | 0.3 | 0.2 | 0 | 0.3 | 1.06311 |
| 3M-1Y | 0 | 0.1 | 0.7 | 0 | 0.2 | 1.05888 |

* **Validation direction gate (D-015, blend AUROC 95% CI above 0.5):** passed at 1D (lower bound 0.5005), 5D, 10D, 15D, 20D, 3M, 6M and 1Y; failed at 2D.
* **2020-2022 diagnostic:**

  | Horizon | 1D | 5D | 20D | 3M | 6M | 1Y |
  |---|---|---|---|---|---|---|
  | Log loss | 1.0276 | 1.0625 | 1.0758 | 1.0882 | 1.0668 | 1.0767 |
  | Training prior | 1.0487 | 1.0753 | 1.0836 | 1.0842 | 1.0783 | 1.0634 |
  | Up-vs-down AUROC | 0.510 | 0.509 | 0.488 | 0.479 | 0.517 | 0.418 |

* **Reading:** v2's probabilities beat the base rate at 1D-20D, but its direction ranking is at chance. The validation gate did not carry over to 2020-2022, and the Trade Desk shows this AUROC beside the flag.
* **10-90% band coverage on 2020-2022:** 0.74-0.84.

**v2 vs v1.1 on identical 2020-2022 rows** (log-loss gain of v2; month-block 95% intervals):

| Horizon | Gain of v2 | 95% interval | AUROC v1.1 / v2 |
|---|---|---|---|
| 1D | +0.0012 | [+0.0004, +0.0020] | 0.511 / 0.510 |
| 2D | -0.0005 | [-0.0017, +0.0006] | 0.514 / 0.515 |
| 5D | +0.0014 | [-0.0001, +0.0027] | 0.508 / 0.509 |
| 10D | +0.0013 | [-0.0008, +0.0036] | 0.497 / 0.502 |
| 20D | -0.0016 | [-0.0047, +0.0016] | 0.495 / 0.488 |

v2 is essentially v1.1 on the shared horizons. What it adds is every swing length plus 3M, 6M and 1Y.

**Train/serve parity** (`reports/verification/v2_inference_parity.csv`, 24 rows x 23 horizons through the dashboard / forward-test path):
* rows whose training sample had no corpus news reproduce to <= 0.00024;
* rows with corpus news differ by up to 0.019, because inference takes news only from user texts.

So in the forward test (no live news feed), v1.1 and v2 both run with the news modality missing.

## E15: v2 backtest (E04 engine; thresholds frozen on 2019 with 50 trials; 2020-2022 reused diagnostic)

| Horizon | Model + ATR stops: total return (base / severe costs) | Sharpe (base / severe) | Trades (long / short) |
|---|---|---|---|
| 5D | +38.2% / +19.1% | 0.62 / 0.38 | 1870 / 277 |
| 10D | +60.3% / +46.0% | 0.82 / 0.68 | 1348 / 81 |
| 20D | +39.5% / +25.1% | 0.63 / 0.46 | 880 / 22 |
| 3M | +60.8% / +53.0% | 0.73 / 0.67 | 790 / 0 |

* Benchmarks: equal-weight survivors +39.7% (Sharpe 0.53, max drawdown -41%); SPY total return +24.6% (Sharpe 0.42, max drawdown -34%).
* The model + ATR-stop policy is almost entirely long on a survivor universe. The fair comparison is equal-weight survivors, which it beats on Sharpe at 10D and 3M before deflation.
* **After deflation it is not significant** (audit A-33):
  * 10D: Sharpe 0.82, 95% interval [-0.25, 2.02], deflated Sharpe ratio 0.0001 after 50 trials;
  * the same holds at the other horizons;
  * 2020-2022 has also been reused many times.
* No economic edge is demonstrated.
* The full rules engine (`nexus`, `nexus_gated`) is flat to slightly negative with few trades.
* The rules alone lose at every horizon.

## E20: classic market-regime rules (O'Neil/IBD, from tradermonty/claude-trading-skills, MIT)

Rule block R:
* distribution days for SPY and QQQ (counts over 5, 15 and 25 sessions, plus the risk class);
* the follow-through-day state (the skill's own state machine, vendored unchanged);
* power trend.

Spot checks match the known follow-through days (2009-03-23, 2020-04-02, 2022-06-24). Tested on E19's design.

| Comparison | Family | Decision | Detail |
|---|---|---|---|
| R on top of the deployed market features (M2 + R vs M2) | logistic and LightGBM | **dropped** | e.g. logistic 2D-20D -0.0086 [-0.0246, +0.0089] |
| R on its own (R vs P) | logistic | kept | 3M +0.0038 [+0.0010, +0.0064] on 2017-18; +0.0003 (n.s.) on 2020-22 |
| R on its own (R vs P) | LightGBM | dropped | |

* In 2020-22 the rules alone did show a 1D gain (+0.027 [+0.015, +0.040]) that was absent on 2017-18.
* The placebo showed no gain.
* Per the pre-registration, no new market read is frozen. The rule states appear in the Trade Desk as descriptive
  context with this evidence.

## Trade Desk tools adapted from the two skill repositories (D-034; tools, not signals)

* **Live texts:** recent Yahoo Finance headlines (timestamped) and, when the owner sets `SEC_USER_AGENT`, EDGAR 8-Ks. Ticked
  items join the texts.
* **Market-regime panel:** distribution days, follow-through state and power trend.
* **Section 7, trade discipline** (paper only; a local, git-ignored journal):
  * drawdown circuit breaker (2% daily, 2-loss cooldown, 5% weekly, 8% monthly);
  * pre-trade discipline gate (written plan, predefined stop, size, risk dollars, revenge window, breaker);
  * fixed-risk sizing.

## Forward test and sealed period

* **Logged every evening before the next open** (hash-chained), all three starting with session 2026-11-05:
  * v1.1 (`shadow/v1.1/`);
  * v2 (`shadow/v2/`);
  * the E19 SPY 20D read (`shadow/market_e19/`).
* **Sealed 2023+ period:** opened once, by the owner, under `docs/SEALED_EVALUATION_PLAN.md`. That plan is final for the models; the scorer and the embedding route are still to be fixed before opening.

## Operational notes from this run

* **D-027 (compute only, model unchanged):** logistic fits were about 80x slower under BLAS thread oversubscription (one cell took 4.1 hours). Fixed with one BLAS thread; E11 was restarted from scratch.
* **D-028:** the shadow forecaster hung for six hours on one Yahoo request. It now has per-ticker timeouts, retry passes and a decision-time deadline.
* **Bundle selection bug:** the app loaded "the newest bundle folder", which became the E19 market model. It now loads the frozen v1.1 record, and a regression test covers it.

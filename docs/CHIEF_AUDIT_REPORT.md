# NEXUS Alpha: chief audit report

Audit date: 6 November 2026. Scope: the whole system as built (data lineage and point-in-time rules, labels, splits, features, the three model families, training and optimisation, calibration, statistics, backtest, RAG, inference and app, code quality and governance). Method: line-by-line review of the risky code paths, re-measurement of every claim against artifacts, targeted experiments where a claim could not be checked by reading (E06 optimisation, E07 tuning and curves, E08 final assembly, E09 convexity certificate, E10 controls, E04 v1.1 backtest). Figures and tables: `reports/NEXUS_Audit_Report.html` (also the app's "L Audit" tab).

Severity: **High** = changes a reported result or a trade; **Medium** = weakens a conclusion or a control; **Low** = hygiene; **Info** = context a reviewer must know.

## 1. Verdict

The point-in-time machinery is sound: no leakage into the test years was found, the seal was never opened, and every reused artifact reproduces exactly.

The audit found 21 issues (register below):
* **Changed trading results:** two real bugs. A-01 target levels and A-02 portfolio accounting.
* **Documented but missing:** one control, the A-04 daily-loss kill-switch.
* **Weakened selection:**
  * The backtest's threshold selection could pick degenerate configurations (A-06).
  * A tuning script let the 2017-2018 validation block cap LightGBM's tree count at two horizons (A-21), a choice the protocol reserves for 2016.

All are fixed, covered by tests or asserts, and the affected experiments were re-run.

The v1 modelling was under-optimised: constant learning rate, short patience, one seed, untuned trees, no refit. E06-E08 address this, and the convex head is now certified at its unique global minimum. The deployed model improved significantly at short horizons.

**The research conclusion does not change:** the signal is mostly "will it move", direction is weak and regime-dependent, and there is no demonstrated trading edge.

## 2. Findings register

| ID | Sev. | Area | Finding | Evidence | Fix and verification | Status |
|---|---|---|---|---|---|---|
| A-01 | High | Rules / targets | "First obstacle" targets used the 252-day high INCLUDING the current bar, so a breakout's target was often that bar's own high (median 1.32 ATR away, 10th percentile 0.07 ATR); reward-to-risk then failed for mechanical reasons | distance distribution of all setups' targets | obstacles from prior bars only; an obstacle must be >= 0.5 ATR away (rule X021 "material"); otherwise a measured move (X024). After: breakouts median 4.86 ATR, 10th pct 1.47; pullbacks >= 0.5 ATR. Rule events rebuilt | fixed |
| A-02 | High | Backtest | Daily position returns were price ratios applied to a fixed entry weight, so a multi-day trade's daily P&L did not sum to its actual return | new regression test on a moving path | daily P&L = weight × (price change / entry price); test `test_daily_pnl_sums_to_trade_return_on_a_moving_path` | fixed |
| A-03 | Low | Train/serve | Inference marks a chart view with < 40% of its bars as missing; training always marked views present | per-view valid-bar counts from TriFrame's sealed index | counts added to the frame and the same rule applied in training. Measured: 0 rows affected in the s500 frame (TriFrame only cached full windows), so no result changes | fixed, no impact |
| A-04 | Medium | Risk control | The risk document said the portfolio simulator enforces the 2% daily-loss limit; it did not | code review | day-by-day portfolio with the kill-switch (X093) and gross cap; test `test_portfolio_kill_switch_and_gross_cap` | fixed |
| A-05 | Low | Metrics | Sortino used the standard deviation of negative days only | code review | downside deviation over all days | fixed |
| A-06 | Medium | Backtest selection | Threshold selection on validation Sharpe let a 2-trade configuration win (NEXUS 5D, theta 0.20) | E04 v1 table | v1.1 eligibility: >= 30 validation trades; both v1 and v1.1 reported | fixed |
| A-07 | Low | Reproducibility | Run manifests flagged a dirty tree but not which files or what diff | manifests | manifests now store dirty files and a SHA-256 of the working-tree diff | fixed |
| A-08 | Medium | Training | (caught in new code before it ran) a fixed-length refit would have restored the epoch with the best loss on the 2019 calibration block, a silent selection on calibration data | review of E08 | `restore_best=False` for refits; test | fixed |
| A-09 | Medium | Optimisation | v1 fusion used a constant learning rate, patience 3 and one seed; most models stopped after 5-9 epochs, so "converged" was not demonstrated | E03 histories | E06: learning-rate sweep, cosine schedule with warm-up, patience 6, capacity/regularisation variants, 5 seeds, loss landscape | done |
| A-10 | Medium | Trees | LightGBM used fixed parameters and was never refit on 2016-2018 data | E01 config | E07: 25-trial TPE search on validation, per-iteration curves, learning curves, importance, refit | done |
| A-11 | Medium | Calibration | Calibration fitted on one calm year (2019) made 2020-2022 log loss slightly worse (H5: −0.0004, CI excludes 0) | E03 | E08 compares none / static 2019 / walk-forward recalibration on the test years | done |
| A-12 | Medium | Decision engine | The direction-skill gate is statistical (AUROC CI > 0.5 on validation) and let through trades that lost at 20D on 2020-22 | E04 v1 | not retuned on test data; every LONG/SHORT plan carries the backtest evidence; an economic gate is a pre-registration candidate for the next study | mitigated |
| A-13 | Medium | Inference | The bundle supported one fusion model only, so a better ensemble or blend could not be deployed | code review | bundle v1.1: fusion seeds averaged, LightGBM per horizon, convex logistic, blend weights, calibration last | fixed |
| A-14 | Low | Inference | The in-sample notice assumed training ended in 2015 also for refit bundles | code review | notice follows the bundle's training span | fixed |
| A-15 | Info | Evaluation governance | The 2020-2022 test block has now been scored by E01, E02, E03, E06, E07 and E08; it is a heavily reused diagnostic, not evidence of generalisation | experiment log | stated in every report; only the sealed 2023+ period (semi-confirmatory) or forward data can confirm | open by nature |
| A-16 | Info | Model outputs | Quantile heads were trained but never validated or shown | code review | coverage of the deployed bundle measured on validation (0.83-0.84) and test (0.82-0.84); ranges shown when within 0.80 ± 0.08 | fixed |
| A-17 | Info | Comparability | E02-E03 used ConvNeXt-T although DINOv2-S was marginally better images-only on validation | E02 | recorded in the registry; the difference (≈ 0.001-0.003 log loss) is below the noise between seeds measured in E06 | recorded |

No leakage was found in the checks below; they are part of the automated suite: future-bar perturbation of features, charts, pivots and patterns; seal readers; purge and embargo (including a planted violation); news and SEC timing; label parity with TriFrame (99.32%); indicator parity (< 1e-9); renderer byte parity (20 of 20); embedding parity (cosine ≥ 0.99998); masked-modality zero influence (all fusion variants); RAG cannot change probabilities.

## 3. SWOT

**Strengths**
* Point-in-time discipline end to end: read-time seal, calendar-based label windows, availability rules for news and filings, purged and embargoed splits, all tested.
* Reuse is verified, not assumed: byte-exact chart parity, embedding parity, label and indicator parity with the parent projects.
* Honest pre-registration (commit `1b17168`), full experiment log including failures, artifacts behind every number.
* Proper scoring rules and dependence-aware inference (month-block bootstrap, walk-forward across 15 years, Holm, PSR/DSR, PBO).
* Separation of forecast and decision; hard gates a model score cannot bypass; paper-only.
* Certified global optimum for the convex heads (E09), measured convergence and seed stability for the networks (E06).

**Weaknesses**
* Survivor-only universe from one vendor; 2020-2022 test re-used many times; one development test window dominated by 2020.
* The economically relevant quantity (direction) is near chance; most predictive gain is volatility-driven SIDEWAYS discrimination.
* Classical text features only (transformer encoders not run); news coverage thin before 2010 and nearly absent in 2024.
* Chart images are scale-free OHLC bars resized to 224²; resolution and candlestick variants untested.
* The direction-skill gate is not an economic gate (A-12).

**Opportunities**
* Forward shadow test from the freeze date (the only clean test of direction).
* Volatility/"will it move" products: the predictable part (SIDEWAYS AUROC ≈ 0.58-0.61) supports risk sizing and option-style use cases better than direction.
* GPU profile: transformer text encoders, native-resolution charts, 979-stock universe, delisted names.
* Economic gate and rolling recalibration as pre-registered changes for the next study.

**Threats**
* Multiple testing across ~150 configurations: any single "win" can be selection noise (DSR ≈ 0 in the backtest).
* Regime dependence: direction AUROC swings from 0.43 to 0.57 by year at 20 days; calibration drifts in crises.
* Data licences (bars without a declared licence, research-only news) block any commercial or public release of data or weights.
* Over-trust in attractive dashboards: the app must keep showing NO TRADE and the evidence warnings by default.

## 4. Optimisation and the "global minimum" question

* **Convex heads (multinomial logistic on the numeric features): certified global minimum.** Strictly convex objective (reference-class parameterisation, L2); L-BFGS from 5 random starts reaches the same objective (spread ≤ 2e-14) and the same parameters (≤ 4e-6); gradient norm ≤ 7e-8; Hessian positive definite (min eigenvalue 1e-3). Validation log loss 1.0338 (1D), 1.0620 (5D), 1.0699 (20D), as good as or better than the default LightGBM. (`reports/e09_convexity_certificate.json`)
* **Gradient-boosted trees:** greedy, no global optimum exists to certify. The per-iteration curves show the training loss falling monotonically while the held-out losses reach a minimum and flatten or rise; the number of trees is chosen at the 2016 minimum (E07).
* **Neural fusion:** non-convex; no method can certify a global minimum. Evidence offered instead: learning-rate sweep, cosine schedule, early stopping at the held-out minimum, five seeds landing within a narrow band, and a smooth basin in the 1-D seed-to-seed path and the 2-D landscape (E06).
* **Accuracy vs loss.** Maximising raw accuracy is trivial here (always predict SIDEWAYS, ≈ 41-49%); models are selected on log loss, a proper scoring rule, and accuracy, balanced accuracy, macro-F1, AUROC and PR-AUC are reported beside it.

## 5. Results after the fixes

All numbers come from saved artifacts. Log loss is averaged over the six horizons unless stated. Lower is better. The class prior scores 1.0733 on test. Figures are in `reports/NEXUS_Audit_Report.html`.

### 5.1 Model ladder

| Model | Valid 2017-18 | Calib 2019 | Test 2020-22 | Note |
|---|---|---|---|---|
| Class prior | | | 1.0733 | |
| E01 LightGBM, default settings | | | 1.0632 | v1 baseline |
| E07 LightGBM, TPE-tuned (tree counts after A-21) | 1.0604 | 1.0457 | 1.0640 | tuning gain on validation did not carry to test |
| E03 gated fusion, 1 seed, calibrated (v1 bundle) | | | 1.0662 | v1 deployed |
| E06 fusion, best settings, 5-seed ensemble | 1.0580 | 1.0446 | 1.0629 | |
| E06 numbers-only fusion, 5-seed ensemble | 1.0593 | 1.0451 | 1.0634 | images, news and SEC add 0.0012 on validation, 0.0005 on test (about one seed SD) |
| **E08 blend 0.1 LightGBM + 0.6 fusion + 0.3 logistic, uncalibrated** | **1.0573** | 1.0442 | **1.0613** | best on validation among 68 candidates |
| E08 blend, calibrated on 2019 (selection stage) | | | 1.0625 | |
| **E08 deployed: refit 2003-2018, calibrated** | | | **1.0620** | v1.1 bundle; beats v1 at 1D-10D (CIs exclude 0) |

Per horizon, the deployed model beats the prior at 1D (+0.0200, CI [+0.015, +0.026]), 2D (+0.0134) and 5D (+0.0114). At 10D, 20D and 1M the gain is positive, but the confidence intervals include zero.

### 5.2 Convergence and the minimum (E06, E07, E09)

* **Convex head.** Certified global minimum: objective spread across 5 starts ≤ 2e-14, gradient ≤ 7e-8, Hessian positive definite.
* **Networks.**
  * Learning-rate sweep: 3e-4 best; 3e-3 lost 0.005.
  * With cosine decay and patience 6, the curves flatten before stopping. Five seeds give validation 1.0591 ± 0.0010.
  * The 2-D landscape around the selected model is a smooth bowl. The trained point is within 0.0003 of the grid minimum, which is less than the seed spread.
  * The straight path between two seeds rises by 0.009, so the seeds sit in different, equally good basins. The ensemble averages over these basins and gains 0.0010 over the mean seed on validation.
* **Trees.** In the per-iteration curves, training loss falls monotonically while 2016 and validation loss reach a minimum and then flatten. TPE found settings 0.0007 better on validation (1D/5D/20D), but 0.0009 worse on test than the E01 defaults (six-horizon mean 1.0640 vs 1.0632): the gain was noise.
* **Conclusion.** The models are at or within noise of their attainable minimum on this data. Further optimisation of the same inputs is unlikely to help. The ceiling is the information in the features, not the optimiser.

### 5.3 Controls (E10, LightGBM, test LL minus prior; negative = better)

| Horizon | Real | Placebo within date | Placebo global | Canary (planted leak) |
|---|---|---|---|---|
| 1D | −0.0184 | −0.0058 | +0.0004 | −0.1029 |
| 5D | −0.0077 | −0.0033 | 0.0000 | −0.1007 |
| 20D | −0.0069 | −0.0020 | +0.0001 | −0.0999 |

* The global placebo finds nothing, so there is no leakage.
* The canary is found, so the pipeline can detect real signal.
* The within-date placebo keeps 29-43% of the real test gain. That part is market-wide timing: the class mix of a date is predictable from features shared by all stocks on that date (volatility regime). The rest is cross-sectional.

### 5.4 Calibration (A-11)

On the test years, no calibration (1.0613) beat static 2019 calibration (1.0625) and annual walk-forward recalibration (1.0617). Temperature scaling was chosen only at 10D-1M, on chronological halves of 2019. The procedure is not retuned on test. For the next study, "none" or walk-forward recalibration is the registered candidate. The reliability diagrams (equal-count bins, 95% intervals) show why. 1D-5D are well calibrated. At 10D-1M, DOWN is under-predicted by about 3-5 points across the range on 2020-2022: the March 2020 crash and the 2022 bear market against a model calibrated on 2019, a strong up year. This is a class-prior shift that a static calibrator cannot follow.

### 5.5 Direction gate (A-12)

The gate is closed at 1D and 2D and open at 5D-1M on validation evidence (AUROC 0.522-0.542, CIs above 0.5). On 2020-2022, up-vs-down AUROC was 0.508 at 5D and 0.494-0.497 at 10D-1M. 2020 was below 0.46 at 10D-1M, and 2021 was above 0.53. Direction skill at long horizons is regime-dependent and did not persist.

### 5.6 Return ranges (A-16)

The deployed quantile heads (5 seeds averaged, sorted) cover the realised volatility-scaled return 0.83-0.84 of the time on validation and 0.82-0.84 on test, against a nominal 0.80. The bands are slightly conservative, and no quantile crossings occur. Ranges are now shown in the app and the schema.

### 5.7 Economics (E04 v1.1)

Fixed engine: A-01 targets, A-02 P&L, A-04 kill-switch and gross cap, A-05 Sortino. Thresholds were frozen on 2017-2018 with at least 30 validation trades (A-06). Probabilities for 2017-2019 come from the E08 selection-stage blend (trained on 2003-2015). Probabilities for 2020-2022 come from the deployed refit (trained on 2003-2018). Every scored row is out of sample. Test 2020-2022, base costs; Sharpe intervals from month-block bootstrap.

| Policy | Horizon | θ | Trades | Sharpe [95% CI] | CAGR | Max DD | Expectancy (R) | DSR | v1 Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| Rule only | 5D | | 1,386 | −0.87 [−2.11, 0.34] | −8.1% | −29.1% | −0.036 | 0.00 | −0.96 |
| Model + ATR stops | 5D | 0.10 | 1,325 | 0.02 [−0.76, 1.14] | −1.1% | −25.0% | −0.015 | 0.00 | 0.29 |
| NEXUS (model + rules) | 5D | 0.05 | 245 | 0.04 [−1.11, 1.12] | 0.1% | −6.6% | −0.016 | 0.00 | 0.31 (2 trades, θ 0.20) |
| NEXUS gated (live planner) | 5D | 0.10 | 41 | 0.31 [−0.64, 2.08] | 0.8% | −5.1% | −0.010 | 0.01 | 0.20 |
| Rule only | 20D | | 761 | −0.34 [−1.61, 0.71] | −4.4% | −24.4% | −0.014 | 0.00 | −0.98 |
| Model + ATR stops | 20D | 0.05 | 1,110 | 0.16 [−0.96, 1.47] | 1.0% | −37.8% | +0.018 | 0.01 | 0.04 |
| NEXUS gated (= NEXUS at 20D) | 20D | 0.10 | 322 | −0.14 [−1.19, 0.77] | −1.6% | −14.2% | −0.041 | 0.00 | −0.64 |
| Equal-weight buy-and-hold, same survivors | | | | 0.53 [−0.41, 1.72] | 11.8% | −40.9% | | | 0.53 |

* PBO over the 20 validation configurations is 0.12, so selection overfitting is low. There is simply no edge to select.
* Stressed (15 bps) and severe (20 bps) costs lower every Sharpe by 0.02-0.4.
* The kill-switch was live: NEXUS policies had 1-3 sessions below −2% and the model-ATR 20D policy had 61, each blocking new entries on the next session.
* Two of the fixes changed the rule policy directly:
  * The fixed targets (A-01) let more rule setups pass the reward-to-risk gate: test trades rose from 1,075 to 1,386 at 5D and from 686 to 761 at 20D. With the corrected P&L (A-02), rule-only 20D moved from −0.98 to −0.34. It still loses.
  * The trade-count floor (A-06) removed v1's degenerate 2-trade selection.
* **Verdict.** No policy is distinguishable from zero, and none beats buy-and-hold of the same survivors on a risk-adjusted basis. The better forecasts (E08) did not create an edge. The economic conclusion of v1 stands.

### 5.8 What changed and what did not

* **Changed.**
  * Four bugs were fixed: target levels, P&L accounting, the missing kill-switch, and degenerate threshold selection.
  * The model is better: 1.0620 vs 1.0662 test log loss, significant at 1D-10D.
  * Every network result now has seeds, curves and a landscape. The convex head is certified.
  * The deployed bundle is the validated blend and carries return ranges.
* **Unchanged.** The signal is mainly "will it move". Direction is weak and unstable across regimes. Images, news and filings add 0.0005-0.0012 over numbers alone, about one seed SD. There is no demonstrated trading edge.
* **Recommended next step.** Freeze v1.1 and run a forward shadow test (no data until it is scored). Optionally, the owner can authorise the single sealed 2023+ evaluation as a semi-confirmatory read.

Additional findings logged during the fixes:

| ID | Sev. | Area | Finding | Fix | Status |
|---|---|---|---|---|---|
| A-18 | Low | Backtest script | A loop variable shadowed the parsed arguments, so E04 v1.1 crashed after computing (no wrong number was written) | renamed; re-run | fixed |
| A-20 | Low | Backtest provenance | The gated policy's skill flags were read from the E03 bundle even when another model's probabilities were passed | `--bundle` option and the source recorded in the manifest. E08's flags agree with E03's at 5D and 20D (both open), so v1.1 results are unchanged | fixed |
| A-21 | Medium | Protocol (E07) | The final tuned LightGBM fits recorded their curves inside the early-stopped run. LightGBM's early stopping watches every non-training set, so the 2017-18 validation block ended the run at 1D and 2D, capping the tree count and leaking validation data into a choice reserved for 2016 (5D-1M unaffected) | tree count from 2016-only early stopping, then a fixed-rounds fit records the curves; an assert checks the 2016 minimum is unchanged; E07, E08, the bundle, coverage and E04 v1.1 re-run | fixed |
| A-22 | High | Live inference | Live mode requested Yahoo daily bars with `period="3y"`, which is not a valid Yahoo range and returns no rows, so every live forecast failed ("no daily data"). Found 2026-11-06 when starting the forward shadow test | explicit start date (3 years + 7 days); the frozen weights are unchanged; a golden test now pins v1.1's historical outputs | fixed |
| A-23 | Medium | Live inference | Share-class tickers (BRK.B) were sent to Yahoo with a dot (Yahoo uses BRK-B), so they always failed live; too-short Yahoo histories surfaced as a bare AssertionError | `yahoo_symbol()` mapping in live inference and shadow scoring; explicit "insufficient history" error; the shadow forecaster retries empty responses once and logs late-recovered stocks in chained addendum files | fixed |
| A-24 | Medium | Features (v2) | `mkt_n` (the number of universe stocks with data on the date) rises monotonically from 786 (2001) to 979 (2022): it is a calendar index, not market information. Trees can use it to separate eras, and validation/test values (958-979) lie at or beyond the top of the training range. It is in E11's pre-registered F2 (and F3, F4); `gmd_data_year` had been excluded for the same reason. Found 2026-11-06 while E11 was running, before any keep decision was computed | E11 is reported as run (disclosed). D-030: if any block from F2 onward is carried, `mkt_n` is removed from the carried set; E19 never uses it | open (decision D-030) |
| A-25 | High | App | `latest_bundle()` returned the newest bundle folder of any kind; once the E19 market bundle existed the app's stock predictor tried to load it and the dashboard could not run an analysis (caught by `tests/test_app.py`, 2026-11-06). The forward test was unaffected (it always loads the frozen record) | v1 stock bundles only; the app loads the frozen v1.1 record by name with its hash check; regression test | fixed |
| A-26 | Medium | Live inference (v2) | v2 inference set the decision time to the next calendar day 14:00 UTC: Fridays mapped to Saturday and summer dates were an hour late, so texts and filings were judged against the wrong cut-off. Found in review before v2 was frozen | the v1/training rule (09:00 New York on the next session, calendar-based); test | fixed |
| A-27 | Medium | Forward test | The scheduled task did not load the owner's key files, so v2's FRED context would have been silently missing (imputed) every night once v2 joined the forward test | git-ignored `scripts/shadow/local_env.cmd` (paths only) called by `run_daily.cmd`; verified that the keys load in the task environment | fixed |
| A-28 | Medium | Forward test (labels) | The shadow scorer fetched Yahoo history from 60 days before the first forecast; the label threshold's EWMA volatility (alpha 0.06) kept ~8% weight on its start value. Measured on Jan-Mar 2022: label agreement with training 98.4% | 3 years of history (as live inference). Re-measured: 99.2% agreement (disagreement halved); the rest is the Yahoo vs vendor price difference (median 2.6 bp) | fixed |
| A-29 | Low | Data (v2 frame) | `build_v2_frame.py` dropped v1 `y_`/`z_`/`end_` but not `ret_`, so the v2 frame's `ret_*` columns were duplicated (`_x`/`_y`) or empty and the per-horizon purge was not applied to returns. No reported result used them | Builder fixed; frame repaired (the 46 label columns verified byte-identical) | fixed |
| A-30 | Low | Live inference (v1) | Beyond the market calendar, v1 fell back to 'next calendar day 14:00 UTC' (the bug fixed in v2 as A-26) | tz-aware next-business-day rule; golden test unchanged | fixed |
| A-31 | Medium | Reporting (v2 skill flags) | E14's direction gate used one-month bootstrap blocks; 2017-18 holds fewer than 4 independent 6M windows and about one 1Y window, so the 'validated' flag at 6M/1Y is not supportable | The Trade Desk shows 6M/1Y as 'not validatable'; the frozen bundle is unchanged (its flags are logged as recorded) | fixed (display) |
| A-32 | Low | Operations | The market calendar ended 2027-12-31 | Extended to 2030-12-31 after the overlap check against pandas_market_calendars (0 differences) | fixed |
| A-33 | Medium | Reporting (E15) | V2_RESULTS reported E15's Sharpe 'before deflation' without the deflated value: 10D DSR 0.0001 after 50 trials, Sharpe 95% CI [-0.25, 2.02] | Stated in V2_RESULTS: no economic edge is demonstrated | fixed |
| A-34 | High | Fundamentals (prices) | As-traded price applied splits after the vendor's split basis (KLAC 10:1 in 2026-06: market cap 10x on every development date) | `VENDOR_SPLIT_BASIS` = 2026-02-27; data-contract test (KLAC, GOOG, NVDA as-traded closes) | fixed |
| A-35 | High | Fundamentals (prices) | S03 split table covered only part of the universe (GOOG 20:1 missing: price 20x low before 2022-07) | Split history from the vendor's Yahoo files + yfinance + S03; 0 disagreements 2009-2026-02 | fixed |
| A-36 | High | Fundamentals (shares) | 20-F/40-F filers: ordinary shares x ADS price, local-currency statements (TSM $3.1tn) | Foreign-filer guard; non-USD reporters keep no money facts (0 leaking rows) | fixed |
| A-37 | High | Fundamentals (shares) | Weighted diluted shares differenced as flows (Q4 = FY - 9M: 0 or negative) | Averages never differenced; test | fixed |
| A-38 | High | Fundamentals (shares) | Spin-offs recorded as splits (ABT x2.08), split-restated filings before the effective date (AOS), counts in thousands (AKR x1000), 100-share shell (ETN): 144 share-basis jumps | Point-in-time continuity guard + 1M floor: 0 of 193 one-day moves > 50% left | fixed |
| A-39 | Medium | Fundamentals (shares) | Stale cover-page counts reused (TAP: a 2010 value) | Count must be dated within the latest period; test | fixed |
| A-40 | Medium | Fundamentals (shares) | BRK.A/B share one count; GOOG/META had no cover count | Class guard; diluted and NI/EPS fallbacks; megacaps within 15% of public values | fixed |
| A-41 | Medium | Dashboard (live) | Live snapshot had no filer form, so the foreign guard never fired (TSM $9.8tn live) | Latest form on or before the date; test | fixed |
| A-42 | Medium | Operations | Yahoo unreachable: every ticker fails slowly, no deadline is set, run killed at the 3-hour limit | Preflight, circuit breaker, done marker (D-041) | fixed |
| A-43 | Low | Code | ruff F/E9/B: unused imports/locals, mutable dataclass default | Fixed outside the sealed scorer and experiment records; tests and golden test pass | fixed |
| A-44 | Low | Repository | Mixed line endings (196 CRLF / 164 LF) | Restored per file; no global normalisation (hash-chained logs, sealed-scorer hash) | accepted |
| A-45 | Info | Data source | SEC companyfacts lags some combined registrants (AEP) | Latest filing and days since filing shown | accepted |
| A-46 | Info | E23 | SIC codes are current, not point in time | Disclosed in the pre-registration | accepted |
| A-47 | High | Operations (found in audit 4) | A patch wrote a vertical-tab byte (`\v`) into the v3_fund script path in `run_daily.cmd`: the step would have failed silently every night | Fixed; `tests/test_shadow_runner.py` (byte-clean CRLF, scripts exist, scored versions frozen and monitored) | fixed |
| A-48 | High | Inference (found in audit 4) | `latest_bundle()` treated any bundle without a `kind` field as a v1 stock bundle, so the frozen v3_fund bundle became the app's default predictor and failed to load (caught by two `test_inference` errors) | A v1 bundle must hold its preprocessor; new fundamentals bundles carry `"kind": "fund"`; 109 tests pass | fixed |
| A-49 | Medium | RAG explanations (found by the new hygiene test) | Three rule-to-book regexes in `rag/explain.py` held literal backspace bytes instead of `\b` word boundaries (since commit cc2e75e), so `support`, `resistance` and `hammer` never matched a book page and those rules had no book citation. Probabilities were never affected (RAG never changes them; tested) | Bytes replaced with `\b`; `tests/test_repo_hygiene.py` fails on any control character in a tracked text file (it also caught a `\v` in this register) | fixed |
| A-50 | High | Operations (incident, 2026-11-07) | A bar cache filled before the last close was served to the next night's run, which forecast an already-logged session and wrote nothing for v1.1 and v2; the done marker was written anyway (no retry), v3_fund logged a 1-stock file from partial input, and the scorer crashed on an all-late version | `expected_asof()` (last 16:05 ET close on the calendar); the cache is refetched if it lacks that session; the marker requires v1.1, v2 and market_e19 to have logged it; v3_fund refuses fewer than 450 stocks or a stale session; the scorer returns on an all-late version; SPC acknowledges the permanent junk file by SHA-256 (tested). Makeup run logged the session on time (D-045) | fixed |
| A-51 | Medium | Colab notebook and docs (found by Antigravity's independent review, 2026-11-07) | Notebook cell 20 passed the newest bundle folder to `e04_backtest.py` as the E03 run id; the newest bundle is now `v3_fund`, so the cell would fail with FileNotFoundError. README, LIMITATIONS and IMPLEMENTATION_STATUS still said 109 tests, v2 "being built", no sealed script, no screenshot read, FinBERT not run | Cell 20 now selects the `*_e03_fusion_*` predictions run by name; the three documents corrected (119 tests). A third claim in the same review, that the pins in `requirements-colab.txt` do not exist, is wrong: they equal the local environment and all resolve on PyPI | fixed |
| A-52 | Medium | Colab path (second independent review by Antigravity, 2026-11-07) | Valid: "Run all" on the orchestrator stops at its first v1-pipeline cell when only the minimal fundamentals set is staged (cell 10 runs the whole suite, `test_seal_gate.py` needs hourly bars); cells 12-20 stop without the cost opt-in; cell 20 raised IndexError when no E03 run exists; there was no clear message when Drive had not synced. Not valid: the repository is public, so the clone works (the status table still said private; corrected); the torch reinstall concern is moot because the new path installs no torch. Section 13 GPU timings in that review are estimates, nothing was run | A separate run-all notebook (`NEXUS_Alpha_Colab_RunAll.ipynb`, 5 cells) calls one driver, `scripts/colab/reproduce.py`: Drive check with a clear message, copy to local disk, layout check, tests, build, audit, E22, E23, comparison, frozen check, then one scrubbed summary file. Minimal pins in `requirements-colab-repro.txt` (no torch). `test_stale_bar_cache_is_refetched` skips without torch; E22/E23 need torch (through the v1.1 bundle loader), so the script checks for it first. New `tests/test_colab_runall.py` (4 tests). Rehearsal from a fresh clone in a clean environment against the staged Drive copy: **REPRODUCED** (E22 and E23 differences 0.00000, audit 7 of 7, about 21 minutes on 12 cores) | fixed |
| A-53 | Low | Forward-test runner settings (found while fetching SEC addresses for the dashboard, 2026-11-10) | The git-ignored `scripts/shadow/local_env.cmd` had LF-only line endings and a corrupted tail: a fourth key-file path that does not exist (a mangled copy of the `local_sec.env` path with its backslashes lost) and a stray fifth line that cmd tried to run as a command, printing an error at the start of every nightly run. The three real key files (FRED, Gemini, SEC User-Agent) were still listed first, so every nightly run had its keys: the defect was latent. `test_repo_hygiene.py` could not see it because the file is not tracked | Rewritten with CRLF line endings and only the three existing paths, unchanged; backup in `artifacts/logs/`. `call local_env.cmd` now exits cleanly, and the address fetch that exposed it ran with it | fixed |
| A-54 | Medium | Nightly runner (found by `tests/test_shadow_runner.py`, 2026-11-10) | Adding the v3_alpha lines to `run_daily.cmd` through a shell heredoc turned the path text `\v3_alpha` into a vertical-tab byte (0x0B), the same failure as A-47: cmd.exe would have failed that line every night. The byte-cleanliness test failed at once, before any run used the file | Rewritten from a patch file that builds every backslash with `chr(92)`; the three lines verified byte by byte (CRLF, no control bytes) and run through cmd.exe exactly as the task runs them. Rule restated: never patch Windows paths through a heredoc | fixed |
| A-55 | Low | Live-parity check (audit 5) | D-049's criteria were written in the script minutes before it ran, with no separate timestamp | Disclosed; the check reads no label or return, so it cannot have been tuned to outcomes | disclosed |
| A-56 | Info | v3_alpha inputs (audit 5) | Trained on Alpha158 stored as float32, scored live in float64 | Measured on 496 stocks x 7 horizons: identical predictions | verified |
| A-57 | Medium | OpenBB / CBOE (audit 5) | Slow command layer (about 40 s per stock), an outdated certificate store, CDN throttling of bursts, and an OpenBB transform crash (TSLA) | Direct provider fetcher (about 2 s per stock), certifi bundle with verification on, sequential fetching with backoff, and a fallback parser that agrees with OpenBB on 6,360 of 6,360 contracts; 480 of 500 stocks covered | fixed |
| A-58 | Medium | Committee tab (audit 5) | Listing Gemini models on load would fail offline, including the nightly app test | Guarded | fixed |
| A-59 | Medium | Committee (audit 5) | Google 503 "high demand" on the first model | Fallback across up to three flash models on early transient errors (tested) | fixed; external risk stated |
| A-60 | Medium | Paper readiness (audit 5) | No machine-readable trial count | `docs/TRIAL_LEDGER.md` (up to 665 selection trials on 2017-18 since v2) | fixed |
| A-61 | Medium | Forward-test multiplicity (audit 5) | Two ranking candidates and four E26 tests judged separately at 5% | Holm correction (D-051), before any 20D outcome; implemented and tested | fixed |
| A-62 | Medium | Forward criteria (audit 5) | The promotion criteria existed only as text | `scripts/shadow/forward_criteria.py` nightly; tested on synthetic logs | fixed |
| A-63 | High | Forward test operations (audit 6) | Session 2026-11-10 not logged: the run started 10 s after the laptop left Modern Standby and logged nothing; the next attempt coincided with a Windows Update restart (0xC0000142) | Session 2026-11-11 made up on time (D-052); `scripts/shadow/keep_awake.py` holds off idle sleep during the run; owner script `scripts/shadow/owner_fix_task.ps1` | mitigated; owner settings pending |
| A-64 | Medium | Done marker (audit 6) | Keyed by the local date: an extra trigger after midnight (before the US close) would mark the new day complete with the previous session, so the 07:00 run would skip the new session | Keyed by the as-of session, with a new file prefix so no old marker collides; tested | fixed |
| A-65 | Medium | Dashboard access (audit 6) | No launcher; the owner could not open the dashboard | `RUN_DASHBOARD.cmd` | fixed |
| A-66 | Medium | Explainability of plans (audit 6) | A rejected plan showed no levels and no reasons | Reason card and CONDITIONAL levels from the engine's own rule-only path (`trading/explain_plan.py`, `app/plan_view.py`); engine unchanged; tested | fixed |
| A-67 | Low | Comparison (audit 6) | No multi-stock comparison, although TriFrame had one | Compare 1-5 tab; tested | fixed |
| A-68 | Low | Presentation notebook (audit 6) | The Colab notebook called scripts instead of showing the code | `notebooks/NEXUS_Alpha_End_to_End.ipynb`, code inline, executed locally before delivery | fixed |
| A-19 | Low | Quantile heads | Heads are unconstrained, so q10 > q90 was possible | measured crossing rate 0.0%; monotone rearrangement (sorting) added in inference and in the coverage check | fixed |

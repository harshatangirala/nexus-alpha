# NEXUS Alpha v3 research (candidates beyond the frozen v1.1 / v2)

v1.1, v2 and `market_e19` stay frozen and in the forward test. Anything here is a new candidate: pre-registered, selected
on validation (2017-2018) only, with 2020-2022 as a reused diagnostic and the sealed period untouched.

## Why v3 is about ranking (audit 2026-11-06, Six Sigma "Analyze")

The pooled up-vs-down AUROC of v1.1 and v2 is about 0.50: they cannot say which way a stock goes. The same models do
rank stocks against each other on the same date. Mean daily Spearman rank IC on validation (moving-block bootstrap 95%
interval) for the selection-stage F0 LightGBM (E18):
* 5D 0.019 [0.0005, 0.039];
* 10D 0.020 [0.002, 0.038];
* 20D 0.032 [0.003, 0.062].

Frozen v2 on 2020-2022 is 0.014-0.028 (intervals touch 0). A cross-sectional rank IC is market-neutral by construction (a
common market move changes no ranks), so this is skill at relative performance, the quantity a market-neutral book
trades. The models were trained for 3-class direction, not for ranking.

## E21 cross-sectional ranker (pre-registered 2026-11-06, before any E21 code or result)

**Disclosure.** Written after the rank-IC diagnostic above, which measured existing models only.

**Target.** For each as-of date, the cross-sectional rank of the stock's h-session return (close[t+h] / open[t+1] - 1)
among that date's samples, mapped to a normal score (rank-Gaussianised); dates with fewer than 10 labelled samples are
dropped.

**Candidates.** Each objective is fitted on 2003-2015 with early stopping on inner 2016 and E07's LightGBM parameters:
* R1: LightGBM regression on the rank-Gaussianised target, features F0 (the 80 v1.1 numeric features);
* R2: R1 with F0 + `rel_*` (stock-vs-market beta, correlation, idiosyncratic volatility, relative strength, inferred
  sector);
* R3: LightGBM `lambdarank`, query = as-of date, relevance = within-date quintile (0-4), features F0 + `rel_*`.

**Baseline.** B0: the existing selection-stage F0 LightGBM classifier score p_up - p_down (E18).

**Horizons.** 1D, 5D, 10D, 20D, 3M. Groups: 1D | 5D-20D | 3M.

**Metric.** Mean daily Spearman rank IC on validation. The interval is a moving-block bootstrap over months, with blocks
of max(1, ceil(h/21)) months, 1000 draws, seed 11.

**Keep rule.** A candidate beats B0 if the paired daily IC difference (candidate minus B0), averaged within each group,
has a 95% lower bound above 0 in at least one group and an upper bound of at least 0 in every group. Among candidates
that pass, the one with the highest validation mean IC over the five horizons is chosen. If none passes, B0 stays the
ranking signal.

**Economic diagnostic** (reported, not a selection criterion):
* the within-date top-minus-bottom quintile spread of h-session returns (equal weight; 10 bps per leg round trip
  deducted), mean and 95% interval;
* on validation, and on 2020-2022 as a reused diagnostic.

This is a signal spread, not a tradable backtest: positions overlap, and borrow and capacity are not modelled.

**Use.** If a ranker is chosen (or B0 stays):
* it is refitted on 2003-2018 with its selected tree count scaled by rows;
* it is frozen as `ranker_v3`;
* it is logged nightly in the forward test as daily cross-sectional ranks of the universe. The scorer reports daily
  rank IC and quintile spread.

No trading engine uses it until the forward test shows a positive IC.

**Trial budget.** 3 candidates x 5 horizons, counted with the other experiments for deflation.

## E21 results (2026-11-06)

**Keep rule (validation, paired daily IC vs B0).** No candidate passed:

| Candidate | 1D | 5D-20D | 3M |
|---|---|---|---|
| R1 | +0.012 [-0.004, +0.028] | -0.007 | -0.032 |
| R2 | +0.009 | -0.011 | -0.035 |
| R3 (lambdarank) | -0.017 | -0.038 | -0.042 |

So **B0 stays the ranking signal**.

**Validation IC of B0 (2017-2018):**

| Horizon | IC |
|---|---|
| 1D | 0.009 |
| 5D | 0.019 |
| 10D | 0.020 |
| 20D | 0.032 |
| 3M | 0.034 |

**Within-date top-minus-bottom quintile spread of B0, after 20 bp per spread** (validation, 95% interval):

| Horizon | Spread |
|---|---|
| 1D | **-10.3 bp [-17.4, -3.4]** |
| 5D | -1.7 bp [-23.0, +19.1] |
| 10D | +10.2 bp [-25.2, +47.1] |
| 20D | +36.1 bp [-36.6, +104.6] |

On 2020-2022 (a reused diagnostic) the spreads are: 1D -10.5 bp [-18.7, -1.1]; 5D -1.3; 10D +4.5; 20D +3.6; and 3M -301.5 bp [-810, +83]. The 3M ranking is unreliable.

**Reading.** The ranking skill is statistically real at 5-20D, but too small to cover trading costs reliably at the
sample sizes we have. A short-horizon (1D) long-short would lose money.

**Deployed for measurement only (D-038).** `rank_score` (v2's frozen F0 LightGBM component, 1D-20D) is logged every
night for all ~500 stocks. The forward scorer reports daily rank IC across the whole universe, roughly ten times more
stocks per date than the development samples, which gives a far more precise IC estimate within weeks.

## E22 fundamentals (pre-registered 2026-11-06, before any fundamentals data was downloaded or built)

**Data.**
* Source: SEC XBRL `companyfacts` (us-gaap and dei) for the 979-ticker source universe, using the owner's declared
  User-Agent.
* Seal: only facts **filed before 2023-01-03** are kept.
* Point in time: a fact filed on date F is used for as-of sessions t >= F (the decision is at 09:00 New York on t+1,
  after any filing dated F). For each concept and period, the value used is the latest one filed on or before t, so a
  restatement counts only once it is public.
* Quarterly values: 3-month facts directly; Q4 = fiscal year minus the 9-month year-to-date.
* TTM: the sum of the last four consecutive quarters. Balance-sheet items: the latest instant on or before t.
* Price: as-traded close, meaning the vendor's split-adjusted close times every later split factor (SEC study S03
  corporate actions). Split factors carry no return information; they only undo the vendor's scaling.
* Earnings surprise: SUE from the SEC study's S06 (seasonal random walk), by its public time.

**31 ratios.** Yields rather than multiples, so negative earnings stay ordered.

| Group | Ratios |
|---|---|
| Valuation (8) | earnings yield (TTM net income / market cap); sales yield; book-to-price; EBITDA/EV; FCF yield (CFO - capex); dividend yield; shareholder yield ((dividends + buybacks) / market cap); DCF value / price (fixed assumptions: FCF growth = TTM revenue growth clipped to [-5%, 15%] for 5 years, then 2.5%; discount 9%) |
| Profitability (7) | gross, operating, net and EBITDA margins; ROE; ROA; ROIC (EBIT x (1 - 21%) / (debt + equity - cash)) |
| Growth (5) | revenue, operating income and diluted EPS (TTM vs TTM four quarters earlier); asset growth; share issuance (diluted shares year on year) |
| Leverage and liquidity (5) | debt/equity; net debt/EBITDA; interest coverage (EBIT / interest); current ratio; cash/assets |
| Efficiency (2) | asset turnover; receivables / sales |
| Quality and events (4) | accruals ((net income - CFO) / assets); Piotroski F-score (9 signals); Altman Z (original public formula); SUE |

Also: days since the latest 10-Q/10-K, and a fundamentals-present flag.

**Features.** Each ratio becomes its **within-session cross-sectional percentile** among the universe stocks that have
it. Ranks are unit-free, robust to outliers and stable across regimes. Raw values are for the dashboard only. P/E,
EV/EBITDA and the other multiples are also shown there.

**Experiment.**
* Rows: as-of 2009-2022. Blocks: train 2009-2015, inner 2016, valid 2017-18, test 2020-22 as a reused diagnostic.
* Sets: B = F0 (80 v1.1 numeric features); B + FUND (the ranked ratios, days since filing, present flag).
* Placebo: the FUND ranks permuted **across tickers within each as-of date**, which keeps the date mix and breaks the
  stock link.
* Models: LightGBM (E07 parameters, early stopping on inner) and multinomial logistic, through the E11 engine, on the
  9 representative horizons.

**Endpoints.**
1. The log-loss keep rule (D-032 bootstrap and eligibility) for B + FUND vs B.
2. The daily rank-IC keep rule (E21 metric and bootstrap) for B + FUND vs B.

FUND is kept if either endpoint passes for a family whose placebo shows no gain on that endpoint. Both endpoints are
reported, with the two-endpoint multiplicity disclosed.

**Use.**
* If kept, a v3 stock-model candidate (LightGBM, F0 + FUND) is refitted on 2009-2018, frozen as `v3_fund`, and added to
  the forward test with nightly point-in-time fundamentals.
* Whatever the result, the Trade Desk gains a fundamentals panel as a tool: all ratios and multiples with universe
  percentiles, the three statements, an adjustable DCF, Piotroski and Altman.

**Trial budget.** 3 sets x 2 models x 9 horizons x 2 endpoints.

**Amendment before any E22 fit (D-040, audit 3).** The data audit found share-basis defects. They were corrected
before E22 was run; the sets, endpoints, placebo and keep rule are unchanged.
* The as-traded price applies only the splits the price vendor applied (those up to 2026-02-27). The split history
  comes from the vendor's own Yahoo files, plus yfinance and S03 where those lack a ticker, because S03 covered only
  part of the universe.
* Weighted diluted shares are averages: a missing Q4 takes the fiscal-year average rather than FY minus 9 months.
  Flows are still differenced as registered.
* Price-based ratios are missing for 20-F/40-F filers, for tickers whose CIK has another listed class, and where the
  cover-page count and the diluted-share reference disagree by more than 1.5x.
* Companies that report mostly in another currency keep no money facts.
* Share counts below 1 million, or breaking continuity (> 1.25x from the past year's accepted counts on one split
  basis, unless confirmed by a filing at least 75 days later), are missing: spin-offs recorded as splits, filings
  restated for a split before its effective date, and counts tagged in thousands.

## E22 results (2026-11-07; run `experiments/20261107T001413_e22_fundamentals_a9900a`)

Rows: 149,485 (as-of 2009-2022). Fundamentals present: train 67%, inner 86%, valid 86%, test 87%.

**Decision under the pre-registered rule: FUND is not kept** (all four family x endpoint cells drop). So no `v3_fund`
model is frozen.

| Family, endpoint | B + FUND passes | Placebo passes | Kept |
|---|---|---|---|
| LightGBM, log loss | no | no | no |
| LightGBM, rank IC | no | no | no |
| Logistic, log loss | no | no | no |
| Logistic, rank IC | **yes** | yes (barely) | no |

Validation gains of B + FUND over B (95% block-bootstrap intervals):

| Family | Endpoint | Intraday (1D) | Swing (2-20D) | Long (3M-1Y) |
|---|---|---|---|---|
| LightGBM | log loss | -0.0009 [-0.0018, -0.0001] | +0.0004 [-0.0005, +0.0014] | -0.0058 [-0.0078, -0.0037] |
| Logistic | log loss | -0.0006 [-0.0011, -0.0000] | +0.0008 [+0.0000, +0.0017] | +0.0024 [-0.0023, +0.0078] |
| LightGBM | rank IC | -0.0000 [-0.0107, +0.0119] | +0.0052 [-0.0068, +0.0165] | -0.0122 [-0.0576, +0.0309] |
| Logistic | rank IC | +0.0066 [-0.0062, +0.0186] | **+0.0159 [+0.0058, +0.0255]** | **+0.0531 [+0.0044, +0.1081]** |
| Logistic placebo | rank IC | +0.0029 [-0.0031, +0.0089] | +0.0037 [+0.0002, +0.0072] | +0.0050 [-0.0053, +0.0139] |

**What this means.**
* **Probabilities.** Fundamentals do not improve calibrated up/sideways/down probabilities. LightGBM gets worse at
  long horizons (overfitting 33 more features on 6 training years).
* **Ranking.** The logistic model's cross-sectional ranking improves a lot. Mean daily rank IC on 2017-18, B vs
  B + FUND (placebo):

  | Horizon | B | B + FUND | Placebo |
  |---|---|---|---|
  | 10D | 0.011 | 0.031 | 0.010 |
  | 20D | 0.020 | 0.044 | 0.024 |
  | 3M | -0.006 | 0.047 | -0.001 |
  | 6M | 0.004 | 0.066 | 0.005 |
  | 1Y | 0.035 | 0.118 | 0.030 |

  The gain repeats on the 2020-22 diagnostic: swing +0.0079 [+0.0020, +0.0141], long +0.0267 [+0.0041, +0.0546],
  where the placebo shows none (swing +0.0010 [-0.0019, +0.0037]).
* **Why it is still dropped.** The placebo passed the same "lower bound above zero in any group" test, by +0.00015 on
  swing. The rule says drop, and it is applied as written.
* **The rule's weakness (lesson for future pre-registrations).** Testing the placebo with the same any-of-three-groups
  criterion gives it its own false-positive rate. Comparing the candidate with its placebo directly would have been
  sharper. This is recorded rather than used: the rule cannot change after the result.

## E23 industry-neutral fundamentals (pre-registered 2026-11-06, before any E22 result and before any E23 code)

**Why.** Valuation and quality ratios differ structurally by industry: a bank's book-to-price, a utility's leverage and
a software firm's margins are not comparable. A universe-wide rank partly measures industry membership. The
literature's industry-adjusted versions of these signals are the standard remedy.

**Design.** The same as E22 except for one feature set.
* B_FUND_IND: each of the 31 ratios and days since filing, ranked within the stock's **SIC major group** (first two
  digits, SEC study S02 `universe.parquet`; S02 lacks 236 tickers, which take the code from SEC's submissions
  endpoint for the same CIK, 235 found; added before any E22 or E23 result) on each session. SIC is the filer's
  current code, so industry membership is not point in time (a small, disclosed look-ahead: codes rarely change). When fewer than 8 stocks in the group have the ratio that
  day, the rank is taken within the SIC division (A-J) instead, and failing that within the universe. Present flag as
  in E22.
* Placebo: the B_FUND_IND ranks permuted across tickers within each as-of date.
* Same rows, blocks, models, 9 horizons, two endpoints and keep rule as E22.
* Comparisons: B_FUND_IND vs B (keep rule), with B_FUND_IND vs B_FUND reported descriptively.

**Use.** If B_FUND_IND passes and E22's B_FUND does not, the industry-neutral set replaces it in the `v3_fund`
candidate. If both pass, the one with the larger validation log-loss gain is used. Either way the choice and the trial
count (E22 + E23) are disclosed.

**Trial budget.** 2 sets (B_FUND_IND and its placebo) x 2 models x 9 horizons x 2 endpoints, on top of E22's.

## E23 results (2026-11-07; run `experiments/20261107T002423_e23_industry_fundamentals_2d062c`)

**Decision under the pre-registered rule: FUND_IND is kept.** No placebo passes on any endpoint.

| Family, endpoint | B + FUND_IND passes | Placebo passes | Kept |
|---|---|---|---|
| LightGBM, log loss | yes | no | **yes** |
| LightGBM, rank IC | yes | no | **yes** |
| Logistic, log loss | no | no | no |
| Logistic, rank IC | yes | no | **yes** |

Validation gains over B (95% block-bootstrap intervals; 2020-22 diagnostic in the last column):

| Family | Endpoint | Group | 2017-18 | 2020-22 |
|---|---|---|---|---|
| LightGBM | log loss | long | +0.0037 [+0.0020, +0.0059] | -0.0007 [-0.0054, +0.0062] |
| LightGBM | rank IC | long | +0.054 [+0.022, +0.089] | +0.003 [-0.072, +0.061] |
| LightGBM | rank IC | swing | +0.005 [-0.002, +0.012] | +0.007 [+0.002, +0.012] |
| Logistic | rank IC | swing | +0.010 [+0.001, +0.018] | +0.007 [+0.002, +0.011] |

Mean daily rank IC of LightGBM on 2017-18:

| Horizon | B | B + FUND_IND | Placebo |
|---|---|---|---|
| 10D | 0.013 | 0.020 | 0.015 |
| 20D | 0.020 | 0.029 | 0.018 |
| 3M | -0.015 | 0.039 | 0.003 |
| 6M | -0.020 | 0.047 | -0.033 |
| 1Y | 0.073 | 0.080 | 0.005 |

Against E22's universe ranks (descriptive), the industry-neutral set lowers LightGBM's validation log loss by 0.0096
(3M), 0.0072 (6M) and 0.0018 (1Y). Industry adjustment is what made fundamentals useful to the tree model.

**Honest reading.**
* **What validation shows:** a real ranking gain at swing and long horizons, with placebos clean.
* **What replicated on 2020-22:** the swing ranking gains of both families, not LightGBM's long-horizon gain. The
  intervals there are wide (2020-22 holds only a few independent 3M-1Y windows).
* **Trials:** E22 + E23 together are 5 sets x 2 families x 9 horizons x 2 endpoints. The keep came from the second of
  two related experiments, so the forward test, not this table, is the evidence that matters.

## v3_fund in the forward test (D-042)

By E22's and E23's use clauses, with no choice left open:
* **Model:** LightGBM (E07 parameters) on F0 + FUND_IND, refitted on as-of 2009-2018 for the 9 horizons. Tree counts
  follow E08's convention: E23's early-stopping counts scaled by rows.
* **Freeze:** frozen as `v3_fund` (`registry/frozen_models/v3_fund.json`).
* **Nightly inputs:** v2's assembled F0 row for each stock (forecasts identical with and without this output, A/B
  tested) and the live fundamentals cross-section of the same session.
* **Logging:** hash-chained in `shadow/v3_fund/`, with `rank_score` = p_up - p_down. **No trading engine uses it.**

**Forward criterion (pre-registered 2026-11-07, before any v3_fund forecast exists).** v3_fund becomes a shown signal
in the Trade Desk (still not traded) only if, on on-time rows after at least 120 scored sessions at 20D:
1. its mean daily rank IC at 20D has a date-block bootstrap 95% lower bound above 0; and
2. the paired daily rank-IC difference against `ranker_v3` (v2's F0 LightGBM `rank_score`) on the same dates and
   stocks has a lower bound above 0.

1D-15D results and the log loss against the class prior are reported, but they are not part of the criterion. 3M-1Y
are reported once they mature.

## E24 Kronos zero-shot (pre-registered 2026-11-07, before any Kronos forecast on project data was scored)

**What Kronos is.** A family of decoder-only foundation models for K-lines. The code is MIT
(shiyu-coder/Kronos, commit 67b630e, vendored in `src/nexus_alpha/kronos/vendor/`) and so are the weights (Hugging
Face `NeoQuasar/*`, safetensors, pinned revisions, SHA-256 recorded). Paper: Shi et al., arXiv:2508.02739, AAAI 2026.

**Contamination (the deciding fact).** Verbatim from the paper:
* "The pre-training data for Kronos extends up to June 2024. Consequently, our test period for all tasks begins in July
  2024" (p. 14).
* Table, p. 21: "Nasdaq Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 8,725 | 2,478,662,459 | 2000/1/1" and
  "New York Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 7,073 | 2,133,143,549 | 2000/1/1".

So the daily bars of US-listed stocks from 2000 to June 2024 were in its training data, and almost certainly so were
our 979 tickers. Every development year (2003-2022) and the first 18 months of the sealed period lie inside it. Three
rules follow:
1. **Never a feature.** Kronos output never becomes a feature of any trained NEXUS model: it would carry pre-training
   knowledge of our training and validation labels into the fit, and that knowledge disappears live.
2. **History can only rule it out.** A result on our history cannot show skill. Contamination can inflate skill but
   not hide it, so a historical test is used only as a necessary condition.
3. **Only the forward test is evidence.** It is clean by construction. (The sealed period from July 2024 is also clean
   for Kronos, but the sealed plan is fixed. Adding Kronos to it would be an owner decision taken before opening.)

**Configuration (one, fixed before any result; no tuning).**
* **Model:** Kronos-small (24.7M parameters, the smallest model the paper evaluates) with Kronos-Tokenizer-base.
  Kronos-base is about 4x slower and does not fit the nightly CPU budget.
* **Context:** the last 400 daily bars up to and including the as-of session: open, high, low, close, volume, and
  amount = volume x mean(OHLC). Split-adjusted vendor bars in development, Yahoo bars live. Z-scored per window, as in
  the paper.
* **Sampling:** temperature 1.0 and top-p 1.0, so the paths come from the model's own predictive distribution, which
  probabilities need. The paper's T = 0.6 is for point forecasts. 32 paths, 20 sessions ahead, seeds from (ticker,
  as-of date).
* **Engine:** NEXUS's sampler is bit-identical to the vendored predictor under deterministic decoding and to its own
  uncached reference under seeded sampling (`scripts/kronos/parity_and_speed.py`). Kronos-small takes 2.8 s per stock on
  this CPU.
* **Outputs per horizon (1D, 2D, 5D, 10D, 15D, 20D):**
  * P(down / sideways / up) by the v2 label rule applied to each path: R_h = close[t+h] / open[t+1] - 1, threshold
    max(0.5 sigma_t sqrt(h), 10 bps), probabilities (count + 0.5) / (32 + 1.5);
  * `rank_score` = the mean path return;
  * the path quantiles.

**E24 contaminated diagnostic (necessary condition).**
* **Rows:** the v2 frame's validation block (2017-18), all rows on every 5th as-of session that has rows (sessions
  numbered in date order; the rule is deterministic). Rows without 400 prior bars are skipped and counted.
* **Endpoints** (the project's existing metrics and block bootstrap, D-032 blocks):
  1. the log-loss gain over the training class prior, swing group (2D-20D) and 1D;
  2. the mean daily rank IC of `rank_score` against the realised R_h, swing group.
* **Reference, descriptive only:** v2's F0 LightGBM (B0) on the same rows, from E22's stored B predictions.
* **Decision.**
  * The necessary condition is met if endpoint 1's swing-group interval lies above 0, or endpoint 2's does.
  * If met, Kronos (`kronos_zs`) is frozen (weight hashes) and logged nightly in the forward test. A result here is
    still not evidence: it may be memory.
  * If not met, Kronos is dropped from the forward test. A model without skill on data it was trained on has none to
    find live. The Trade Desk view stays, labelled accordingly.

**Forward criterion (`kronos_zs`, only if E24's condition is met).** On on-time rows, after at least 120 scored
sessions at 20D, Kronos becomes a shown signal in the Trade Desk (still not traded) only if:
1. its mean daily rank IC at 20D has a date-block 95% lower bound above 0; and
2. its paired daily rank-IC difference against `ranker_v3` on the same dates and stocks has a lower bound above 0.

The forward IC minus the 2017-18 IC is reported as the estimated inflation from contamination.

**Trial budget.** 1 configuration x 6 horizons x 2 endpoints. Nothing is selected on 2020-22, which E24 does not read.

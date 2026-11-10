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
* **Rows:** the v2 frame's validation block (2017-18), all rows whose as-of session falls in January, March, May,
  July, September or November. Rows without 400 prior bars are skipped and counted.
  * **Amendment before any Kronos run.** The rule first registered ("every 5th as-of session") would have kept only
    110 of 489 stocks, because the frame samples each stock every 10th bar in a fixed phase.
  * The alternate-month rule keeps every stock, and full daily cross-sections of about 48 stocks: 244 sessions,
    11,675 rows.
  * Estimated 9.5 CPU hours (Kronos-small at 2.9 s per stock), inside the 12-hour CPU rule. It runs at below-normal
    priority, so the nightly forward test keeps precedence.
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

## E24 results (2026-11-07; run `experiments/20261107T125909_e24_kronos_zero_shot_32c34f`)

11,666 of the 11,675 registered rows were scored (9 had under 400 prior bars): 489 stocks, 244 sessions, 2017-18.

**Decision under the pre-registered rule: the necessary condition is NOT met.** Kronos is not frozen and does not enter the
forward test. It stays in the Trade Desk as labelled analysis.

| Endpoint (2017-18, contaminated) | Kronos zero-shot | Reference B0 (F0 LightGBM) |
|---|---|---|
| Log-loss gain over the class prior, swing 2D-20D | **-0.678 [-0.768, -0.581]** | +0.011 [+0.008, +0.015] |
| Log-loss gain over the class prior, 1D | -0.220 [-0.248, -0.191] | +0.015 [+0.010, +0.020] |
| Mean daily rank IC, swing | +0.005 [-0.028, +0.039] | +0.022 [-0.001, +0.043] |
| Mean daily rank IC, 1D | -0.002 [-0.022, +0.017] | -0.005 [-0.027, +0.016] |

**Reading.**
* **Probabilities are unusable.** Kronos's log loss is 1.27 at 1D and 1.96 at 20D, against about 1.05-1.10 for the
  class prior. The path distribution is far too wide or too shifted for the label rule's classes (the level-reversion
  behaviour documented in `docs/KRONOS.md`).
* **Ranking shows nothing.** The swing rank IC is +0.005 with an interval spanning zero, on data Kronos was trained on.
  If it had memorised those years, the IC would have been high.
* **Therefore** there is no skill to look for live, and the forward test is not spent on it.
* **Not tested:** other configurations (Kronos-base, T = 0.6, mean-path point forecasts). None was registered, and
  trying them now would be a post-result search. A new configuration would need its own pre-registration.

## E25 Qlib Alpha158 factor block (pre-registered 2026-11-10, before the Alpha158 features of the frame were built and before any E25 fit)

**Why (owner decision D-046).** Microsoft Qlib's most used asset is its Alpha158 factor library, the default feature
set of its benchmark models. Testing it under this project's rules asks one question: do the 80 hand-built F0 features
miss price-volume information that a widely used public factor library captures?

**Features.** The 157 Alpha158 features of `nexus_alpha.indicators.alpha158`:
* definitions from Qlib's `Alpha158DL` with the `Alpha158` handler's configuration; VWAP0 is left out because the bars
  have no VWAP;
* computed from the sealed daily bars up to and including the as-of session (causal; the same split-adjusted vendor
  bars as F0); +/-inf set to missing; no other processing;
* parity with Qlib 0.9.7's own expression engine on 23 tickers and 122,797 stock-days: the definitions are identical.
  The 10 correlation features differ by at most 3.1e-5 because Qlib computes their inputs in float32; with float32
  inputs they agree on every row (`reports/verification/alpha158_qlib_parity.json`). The check ran before this
  pre-registration and reads no label.

**Rows and blocks.** The v2 frame's train (2003-2015), inner (2016), valid (2017-18) and test (2020-22) rows, as in
E21. Selection uses valid only; test is a reused diagnostic; the sealed period is not read.

**Sets.**
* B = F0 (the 80 v1.1 numeric features);
* B + A158;
* placebo: B + A158 with the A158 values permuted across tickers within each as-of date (seed 11), as in E22 and E23;
* A158 alone: descriptive only (a public factor library against the hand-built features), no keep decision.

**Models, horizons, endpoints and keep rule: unchanged from E22 and E23.**
* LightGBM (E07 parameters, early stopping on inner) and the multinomial logistic, through `evaluation/ladder.fit_eval`.
* The 9 representative horizons.
* Endpoint 1: the log-loss keep rule (D-032 bootstrap and eligibility). Endpoint 2: the daily rank-IC keep rule (E21
  metric and bootstrap).
* A158 is kept if either endpoint passes for a family whose placebo does not pass that endpoint.

**Sharper check (E22's lesson; reported, not part of the decision).** The paired gain of B + A158 over its placebo, by
group and endpoint. A keep where that gain's interval does not lie above 0 in the passing group is reported as weak.

**Qlib-style report (descriptive).** For every set, family, horizon and split: IC (daily Pearson correlation of
p_up - p_down with the h-session return), ICIR (mean / standard deviation of the daily IC), rank IC and rank ICIR.
Qlib's published benchmarks use Chinese A-shares (CSI300) and a 1-day label, so their levels are not comparable.

**Use.**
* If kept: a candidate `v3_alpha` (LightGBM on B + A158 if the LightGBM family passes, otherwise the logistic) is
  refitted on 2003-2018 with E08's tree-count scaling and frozen. It joins the forward test after a live-parity check
  (features from live Yahoo bars equal the development computation on overlapping sessions). Forward criterion: the
  same as `v3_fund`'s (20D rank IC above 0, and above `ranker_v3`, after 120 sessions).
* If not kept: no model is frozen, and Alpha158 stays in the package for analysis.

**Disclosure.** The validation block (2017-18) has been read by every experiment since E11. Each new keep decision on it
is more optimistic than the last, so the forward test, not this table, is the evidence.

**Trial budget.** 4 sets x 2 families x 9 horizons x 2 endpoints, added to the project's trial count for deflation.

## E25 results (2026-11-10; run `experiments/20261110T144028_e25_alpha158_951bbf`)

Rows: 213,103 (train 2003-2015, inner 2016, valid 2017-18, test 2020-22). The run's manifest records the SHA-256 of
the pre-registration text above as it stood when the run started (`476d72df...`); the run began at 14:40, five
minutes after that text was written, and before any E25 feature or fit existed.

**Decision under the pre-registered rule: Alpha158 is kept, on the ranking endpoint only.** No placebo passes.

| Family, endpoint | B + A158 passes | Placebo passes | Kept | Beats its own placebo |
|---|---|---|---|---|
| LightGBM, log loss | no | no | no | - |
| LightGBM, rank IC | yes | no | **yes** | yes |
| Logistic, log loss | no | no | no | - |
| Logistic, rank IC | yes | no | **yes** | yes |

Rank-IC gains of B + A158 over B (95% block-bootstrap intervals; 2020-22 is a reused diagnostic):

| Family | Group | 2017-18 | 2020-22 |
|---|---|---|---|
| LightGBM | intraday | +0.0037 [-0.0070, +0.0144] | +0.0076 [+0.0002, +0.0147] |
| LightGBM | swing | +0.0042 [-0.0022, +0.0102] | +0.0056 [+0.0001, +0.0107] |
| LightGBM | long | **+0.0134 [+0.0071, +0.0215]** | +0.0003 [-0.0097, +0.0188] |
| Logistic | intraday | +0.0076 [-0.0025, +0.0181] | +0.0033 [-0.0079, +0.0149] |
| Logistic | swing | **+0.0145 [+0.0042, +0.0252]** | +0.0075 [-0.0039, +0.0221] |
| Logistic | long | +0.0055 [-0.0110, +0.0209] | -0.0024 [-0.0151, +0.0102] |

The sharper check (B + A158 against its own placebo, in the passing group): LightGBM long +0.0094 [+0.0029, +0.0159];
logistic swing +0.0213 [+0.0096, +0.0334]. For the logistic, part of that margin is the placebo hurting (swing
-0.0068 [-0.0164, +0.0017] against B): random columns cost a linear model ranking quality.

Log loss: no horizon group improves for either family on 2017-18 (every interval includes zero). On 2020-22 the
logistic's long group is worse (-0.0024 [-0.0038, -0.0002]).

Mean daily rank IC of LightGBM, 2017-18 / 2020-22:

| Horizon | B | B + A158 | Placebo | A158 alone |
|---|---|---|---|---|
| 1D | 0.009 / 0.016 | 0.013 / 0.023 | 0.014 / 0.017 | 0.019 / 0.028 |
| 5D | 0.019 / 0.013 | 0.015 / 0.017 | 0.008 / 0.011 | 0.016 / 0.020 |
| 10D | 0.020 / 0.018 | 0.032 / 0.028 | 0.018 / 0.015 | 0.027 / 0.031 |
| 20D | 0.032 / 0.020 | 0.041 / 0.019 | 0.032 / 0.020 | 0.035 / 0.023 |
| 3M | 0.034 / 0.016 | 0.047 / 0.014 | 0.038 / 0.020 | 0.023 / 0.014 |

**A158 alone against the 80 hand-built features (descriptive).** In log loss it is worse for both families on 2017-18
(LightGBM intraday -0.0045 [-0.0057, -0.0033], long -0.0042 [-0.0053, -0.0032]). In LightGBM ranking it is no
different. For the logistic it ranks better at swing on 2017-18 (+0.0164 [+0.0042, +0.0289]), not on 2020-22
(+0.0044 [-0.0160, +0.0262]).

**Qlib-style figures (LightGBM, 20D).**

| Split | Set | Rank IC | Rank ICIR |
|---|---|---|---|
| 2017-18 | B | 0.0317 | 0.180 |
| 2017-18 | B + A158 | 0.0408 | 0.225 |
| 2020-22 | B | 0.0205 | 0.106 |
| 2020-22 | B + A158 | 0.0185 | 0.094 |

The full table is in the run's metrics (`qlib_style`). Qlib's published CSI300 benchmarks are not comparable.

**Honest reading.**
* **Probabilities:** Alpha158 does not improve them at any horizon.
* **Ranking:** it improves cross-sectional ranking on validation, with clean placebos, and both families beat their
  own placebo directly.
* **Replication is partial.** On 2020-22, LightGBM's intraday and swing gains are positive with lower bounds just
  above zero (+0.0002 and +0.0001). The long-group gain that carried the LightGBM keep does not replicate, and the
  logistic's swing interval spans zero.
* **The hand-built features already hold most of the price-volume information.** The public library alone is worse
  in log loss and no better in tree ranking. What it adds is a small amount of cross-sectional information on top.
* **Trials.** E25 adds 4 sets x 2 families x 9 horizons x 2 endpoints, and the keep came from 2 of the 4
  family-endpoint cells on a validation block that every experiment since E11 has read. The forward test is the
  evidence.

**Use, per the pre-registration (D-047).** The candidate `v3_alpha` is LightGBM on F0 + A158, refitted on 2003-2018
with E08's tree-count scaling. It will be:
1. frozen;
2. given a live-parity check: A158 from Yahoo bars against the vendor's on the same 2021-22 sessions, because the
   nightly cache does not reach back into the development years;
3. added to the forward test with `v3_fund`'s criterion.

No trading engine uses it. Timing follows the owner's scope decision of 2026-11-10: the nightly run stays unchanged
in the presentation week. (Superseded the same day by D-049: the owner asked for every integration at once, and
v3_alpha joined the forward test after its live-parity check; see the next section.)

## v3_alpha in the forward test (D-049)

By E25's use clause, with the live-parity gate applied per horizon:
* **Model.** LightGBM (E07 parameters) on F0 + A158, refitted on as-of 2003-2018, with E25's early-stopping tree
  counts scaled by rows (E08's convention). Bundle `20261110T171026_v3_alpha_refit_a14783`, frozen as `v3_alpha`.
* **Live-parity check.** `scripts/verify/alpha158_live_parity.py`, criteria written before it ran; result in
  `reports/verification/alpha158_live_parity.json`. Yahoo against vendor bars on 2022 (39 of 40 sampled stocks;
  BZH has no Yahoo history; 9,789 stock-days):

  | Check | Result | Criterion |
  |---|---|---|
  | Price features (117) | median correlation 0.9989, minimum 0.961 | pass |
  | Volume features (40) | median correlation 0.920, minimum 0.722 | reported |
  | rank_score Spearman, 5D-1Y | 0.971 (5D) to 0.999 (6M) | pass |
  | rank_score Spearman, 1D / 2D | 0.935 / 0.947 | fail (below 0.95) |

* **Deployed horizons:** 5D, 10D, 15D, 20D, 3M, 6M and 1Y. 1D and 2D are not logged.
* **Nightly inputs.**
  * v2's assembled F0 row for each stock (as for v3_fund).
  * The 157 Alpha158 features of the as-of session, from the same cached Yahoo bars v1.1 and v2 used that night
    (`alpha158.features_at` refuses stale or short histories).
* **Logging:** hash-chained in `shadow/v3_alpha/` with `rank_score` = p_up - p_down. No trading engine uses it.
* **Forward criterion (as v3_fund's, from E25's use clause).** v3_alpha becomes a shown signal (still not traded)
  only if, on on-time rows after at least 120 scored sessions at 20D, both hold:
  1. its mean daily rank IC has a date-block bootstrap 95% lower bound above 0;
  2. its paired daily rank-IC difference against `ranker_v3` has a lower bound above 0.
* **First logged session:** 2026-11-09, logged on 2026-11-10 before that session's decision time (19:30 IST).

## E26 implied-volatility benchmark (forward only; pre-registered 2026-11-10, before any options snapshot was taken)

**Question.** The models' one robust skill is "will it move" (sideways against a move). The options market prices
the size of future moves. Does NEXUS's P(move) carry information beyond option-implied volatility? If it does not,
the volatility skill is already in market prices.

**Data, forward only.**
* **When.** Every nightly run, after the forecasts and before 09:00 New York on the next session (the same decision
  clock).
* **What.** A snapshot of each forward-test stock's listed options from CBOE's delayed quotes, through OpenBB
  (`obb.derivatives.options.chains`, provider `cboe`).
* **Where.** The snapshot is stored in `artifacts/iv_snapshots/` (CBOE's data terms rule out publishing it). Its
  SHA-256 goes into the public hash chain `shadow/iv_snapshot/chain.csv`, which fixes the time of each snapshot.
* **History.** No historical options data is used, so nothing here can be overfitted.

**Implied-volatility measure (fixed now).**
* **Contracts.** Only those with a positive bid and ask, and an implied volatility between 0.01 and 5.0.
* **ATM IV per expiration.** For each expiration at least 2 calendar days away: the mean of the call and put implied
  volatilities at the listed strike closest to the underlying price, or the one that exists.
* **30-day IV (IV30).** Linear interpolation in total variance (sigma^2 x days) between the two expirations that
  bracket 30 calendar days. If every expiration is beyond 30 days, the nearest is used; if every one is shorter, the
  longest is used. A stock with no valid expiration has no IV.
* **Horizon scaling.** For h sessions, sigma_h = IV30 x sqrt(h / 252). This includes the first overnight, which the
  label's window (open of t+1 to close of t+h) excludes; that is a stated approximation.

**IV-implied probabilities.** With a zero-drift normal model for the label return and the label rule's own
threshold tau_h (from `shadow.realized_labels`):
* P_IV(sideways) = Phi(tau_h / sigma_h) - Phi(-tau_h / sigma_h);
* P_IV(move) = 1 - P_IV(sideways).

**Endpoints.** At 5D and 20D, for v1.1 and v2, on on-time rows of stocks that have both a forecast and an IV:
1. **Raw comparison (descriptive).** Binary log loss of "move" under NEXUS (P(move) = 1 - p_side) and under IV. The
   paired per-row difference (IV minus NEXUS) is given with a date-clustered bootstrap 95% interval.
2. **Incremental value (decides).**
   * Model: a logistic regression of the move indicator on logit(P_IV(move)) and logit(P_NEXUS(move)), fitted on the
     first half of the scored forward sessions and evaluated on the second half.
   * Baseline: the IV-only logistic, fitted and evaluated the same way.
   * Statistic: the paired log-loss difference, with the same bootstrap.

**Decision, after at least 120 scored sessions at 20D.**
* NEXUS "adds to implied volatility" only if endpoint 2's lower bound is above 0.
* Otherwise the reading is "NEXUS's volatility skill is already priced".
* Endpoint 1 is reported only, because raw IV is not calibrated to the label rule.
* Option coverage (the share of stocks with a valid IV) is reported.

**Trial budget.** 2 models x 2 horizons x 2 endpoints, all forward. Nothing is selected.

## E27 LLM committee (TradingAgents; forward only; pre-registered 2026-11-10, before any committee run)

**What.**
* **The tool.** TauricResearch TradingAgents v0.7.1 (Apache-2.0) runs in its own environment on the owner's Gemini
  key. It is run on demand from the dashboard, never on a schedule, and never on history.
* **Why never on history.** Its language model has read about the outcomes of 2003-2022, so a historical score would
  measure memory, not skill (D-020).
* **Each run.** A five-tier rating (Buy, Overweight, Hold, Underweight, Sell) for one stock as of the last closed
  session.

**Log.**
* **Public part.** Every run is appended to the hash chain `shadow/committee/chain.csv`: ticker, as-of session, run
  time, decision time, rating, models, and the SHA-256 of the full reports.
* **Private part.** The reports themselves (LLM text built from news and filings) stay in `artifacts/llm_committee/`.
* **Late runs.** A run after the decision time (09:00 New York on the next session) is logged and flagged late, and
  never scored.

**Scoring (forward).**
* **Mapping.** Buy and Overweight count as up, Hold as sideways, Underweight and Sell as down. REVIEW (no readable
  rating) counts as a failure.
* **Labels.** At 5D and 20D with the training label rule.
* **Metrics.**
  * the up-vs-down hit rate on rows whose outcome is not sideways, against 50%;
  * the agreement with v1.1's lean on the same stock and session;
  * the share of Hold calls against the share of sideways outcomes.
* **One vote per stock per window.** Runs on the same stock within 5 sessions of an earlier scored run count once:
  the first one.

**Decision.**
* No claim of any kind before 100 scored runs.
* The runs are chosen by the user, so the sample is not random. A hit rate is reported with that caveat and a
  date-clustered bootstrap interval.
* The committee is never a model feature, never blended into a probability, and never an input to the risk engine.

## Forward-test multiplicity (amendment D-051, 2026-11-10, before any 20D forward outcome and before the first E26 snapshot)

**The problem.** The forward test now holds two ranking candidates, `v3_fund` (E23) and `v3_alpha` (E25), each
judged against `ranker_v3`, and E26 runs four incremental tests (v1.1 and v2, at 5D and 20D). Judging each at 5%
inflates the chance that at least one passes by luck.

**The amendment** (no 20D outcome of any model had matured, and no option snapshot existed, so it cannot be tuned to
a result):
1. **v3_fund and v3_alpha.** Each candidate's forward criterion is unchanged in substance: 20D rank IC above 0 and
   above `ranker_v3` after 120 sessions.
   * It is decided on one-sided date-block bootstrap p-values (the share of bootstrap means at or below 0).
   * A candidate passes only if both of its p-values are at or below its Holm threshold across the two candidates
     (0.025 for the smaller p-value, 0.05 for the other), on the larger of each candidate's two p-values.
2. **E26.** The four endpoint-2 tests are Holm-corrected the same way (thresholds 0.0125, 0.0167, 0.025, 0.05).
   * "NEXUS adds to implied volatility" is read only for a test that passes after correction.
3. **Not corrected.** E27 is descriptive, so it is not corrected. The market read (`market_e19`) and the three-class
   models are not ranking candidates.

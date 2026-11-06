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

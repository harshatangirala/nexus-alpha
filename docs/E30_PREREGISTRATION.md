# E30 pre-registration (2026-11-14): do published fundamental anomalies work in this universe, point in time?

**Status: REGISTERED** on 2026-11-14 (IST, before 07:00), before any statistic below has been computed. Decision D-060.
The owner asked for financial metrics and a deeper fundamentals layer, with full authority.

## Why this test, and why on history

The project already computes 31 point-in-time ratios and 7 multiples (P/E, EV/EBITDA, P/S, P/B, P/FCF, Piotroski F,
Altman Z, accruals, SUE and others; `fundamentals/ratios.py`). They were tested inside fitted models (E22 not kept; E23
industry-neutral kept as v3_fund, now in the forward test). What has never been tested is the plain question a factor
researcher asks first: **do the published composite scores, exactly as their authors defined them, rank this
universe's stocks at all?**

A forward-only test cannot answer it in reasonable time. 120 sessions of overlapping 20-session returns hold about six
independent observations. A realistic monthly information coefficient of 0.02, with a cross-sectional IC standard
deviation around 0.12, would need years to separate from zero. History can answer it: 2005-2022 gives about 215 monthly
cross-sections, so the standard error of a mean monthly IC is near 0.008.

Using the development years again is acceptable here for one reason only: **nothing is fitted or chosen.** Every
definition was fixed by its authors years ago (references in `fundamentals/extended.py`) and is frozen in code committed
with this file. The only multiplicity is the six composites below, and Holm corrects for it.

## Definitions (frozen; `src/nexus_alpha/fundamentals/extended.py`)

Six primary composites, each "higher = more attractive" by its own published logic:

| Composite | Source | Score |
|---|---|---|
| value_composite | classic value | mean universe percentile of earnings yield, book-to-price, EBITDA/EV, FCF yield, sales yield |
| quality_composite | after Asness, Frazzini and Pedersen (2019) | mean of profitability, one-year growth of profitability, and safety parts (no beta, no earnings volatility) |
| gross_profitability_rank | Novy-Marx (2013) | percentile of (revenue - COGS) / total assets |
| magic_formula | Greenblatt (2005) | percentile of the mean of the EBIT/EV and EBIT/(NWC + net PP&E) percentiles |
| piotroski_rank | Piotroski (2000) | percentile of the F-score (already in `ratios.py`) |
| beneish_safety_rank | Beneish (1999) | percentile of minus the M-score (low manipulation risk = attractive) |

Financial firms (SIC 60-67) get no Beneish M, Altman Z or magic-formula inputs, because those sources exclude them
(`extended.FINANCIAL_SIC2`). The other composites include them.

## Sample and timing

* **Signal dates:** the last session of each month, January 2005 to November 2022 (the last date whose 20-session
  return ends before the seal, 2023-01-03; the sealed period is never read).
* **Information:** each stock's latest point-in-time snapshot filed on or before the signal date. A filing dated on
  the signal date was public before the next open, the entry. Price-based fields come from the stored panel on the
  signal date (`artifacts/fundamentals/panel.parquet`, built by `scripts/data/build_fundamentals.py`).
* **Return:** R = close[t+20] / open[t+1] - 1 (the project's label rule: enter at the next open, exit 20 sessions
  later), from the sealed development bars (split-adjusted, not dividend-adjusted).
* **Universe:** every ticker in the fundamentals panel with a finite R and a finite score on that date, at least 100
  stocks per date for the date to count.

## Endpoint and decision

* **Statistic:** per signal date, the Spearman rank correlation (IC) between the composite and R.
* **Test:** one-sided, mean monthly IC > 0, moving-block bootstrap over months (blocks of 6 months, 10,000 resamples,
  seed 30), `evaluation.forward_criteria.block_p_value`.
* **Multiplicity:** Holm across the six composites.
* **Decision:** a composite **replicates** if its Holm-adjusted p-value is below 0.05 and its mean IC is positive.
  "Replicates" means it is a candidate input for the next frozen model generation, which would be forward-tested like
  every other. It changes no frozen model, forecast, risk rule or dashboard verdict.

## Reported, deciding nothing (descriptive)

* Top-minus-bottom quintile spread of R, equal-weighted, gross of costs.
* Sub-periods 2005-2011, 2012-2016, 2017-2022, and before and after each composite's publication year.
* Industry-neutral IC (scores ranked within two-digit SIC), and IC after removing size (market-cap rank).
* The composites' correlation matrix, and the quality parts and the Montier C-score on their own.
* Coverage by date.

## Known biases, stated in advance

* **Survivorship:** the universe is today's survivors. Firms that were weak and later recovered are over-represented,
  which flatters junk and works against quality and safety scores.
* **Dividends:** returns are price-only, which works against high-yield value stocks by about the dividend yield times
  20/252 per month.
* **Tag coverage:** COGS, receivables and operating income are tagged for only 52-67% of firm-dates in the stored
  snapshots. Composites needing them cover fewer stocks. Widening the tag map would change the frozen v3_fund model's
  inputs, so it is deferred to the next model generation (finding A-112).
* **Reused years:** 2017-18 and 2020-22 were read before by fitted experiments. Without fitting or selection that
  matters only through the six tests, which Holm handles. Reported as a replication, not a discovery.

## Trial count

6 tests (Holm). 0 fitted parameters. 0 rows from 2023 or later.

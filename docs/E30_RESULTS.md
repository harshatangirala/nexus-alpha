# E30 results (2026-11-14): published fundamental composites in this universe

Pre-registration: `docs/E30_PREREGISTRATION.md` (D-060), committed in 3d06664 before the run. Script:
`scripts/experiments/e30_factor_replication.py`. Output: `reports/e30/e30_results.json` and `reports/e30/e30_monthly_ic.csv`.
Runtime 61 s. Nothing was fitted, and nothing on or after 2023-01-03 was read.

## Verdict: none of the six composites replicates

| Composite | Mean monthly IC | 95% interval | One-sided p | Holm p | Replicates | Q5 - Q1, 20 sessions | Stocks a month |
|---|---|---|---|---|---|---|---|
| Value composite | +0.0021 | [-0.0215, +0.0246] | 0.443 | 0.844 | no | +0.04% | 700 |
| Quality (QMJ-style) | +0.0091 | [-0.0080, +0.0261] | 0.146 | 0.439 | no | -0.22% | 705 |
| Gross profitability (Novy-Marx) | +0.0182 | [-0.0029, +0.0412] | 0.051 | 0.258 | no | +0.35% | 450 |
| Magic formula (Greenblatt) | +0.0104 | [-0.0065, +0.0271] | 0.099 | 0.397 | no | -0.45% | 478 |
| Piotroski F | +0.0137 | [-0.0006, +0.0267] | 0.040 | 0.241 | no | -0.26% | 707 |
| Beneish (low M = attractive) | +0.0017 | [-0.0125, +0.0158] | 0.422 | 0.844 | no | +0.27% | 300 |

IC: Spearman correlation between the score and the next 20-session return (next open to the close 20 sessions later),
one per month, then averaged. Interval and p: moving-block bootstrap over months, blocks of 6.

**In plain words:**
- **No composite passes.** Every interval includes zero, and none of the six survives the correction for testing six at
  once.
- **Gross profitability and Piotroski come closest.** Their point estimates are positive, but neither is distinguishable
  from zero after the correction.
- **Value is flat.** Its IC is about zero over the sample.
- **What this means for the user.** P/E, EV/EBITDA and the published scores built on them describe a company well. In
  these roughly 500-960 surviving large caps, over 20 sessions, from 2009 to 2022, they did not rank next month's
  returns reliably.

## One deviation from the plan, by the plan's own rule

The pre-registration expected about 215 months from January 2005. SEC XBRL company facts only exist from 2009, when
XBRL filing was phased in. The pre-registered rule "at least 100 stocks per date for the date to count" therefore
excluded every month before April 2009. **164 month-ends were available, April 2009 to November 2022; each composite's test uses the months with at least 100 stocks that have its score and a return: 137 to 151 months (Beneish fewest, because of tag coverage; see `months` in `reports/e30/e30_results.json`).** No definition, statistic or
threshold was changed. The "2005-2011" sub-period below is in practice 2009-2011.

## Descriptive (decides nothing)

| Composite | 2009-11 | 2012-16 | 2017-22 | Industry-neutral IC | IC with size removed |
|---|---|---|---|---|---|
| Value composite | -0.018 | +0.010 | +0.001 | +0.004 | +0.002 |
| Quality (QMJ-style) | +0.039 | +0.002 | +0.007 | +0.004 | +0.007 |
| Gross profitability | +0.051 | +0.006 | +0.020 | +0.005 | +0.017 |
| Magic formula | +0.012 | +0.009 | +0.012 | +0.005 | +0.010 |
| Piotroski F | +0.032 | +0.005 | +0.017 | +0.008 | +0.014 |
| Beneish | +0.029 | -0.006 | +0.006 | -0.001 | +0.002 |

Parts and extra scores:

| Score | Mean monthly IC | 95% interval |
|---|---|---|
| Quality: profitability part | +0.021 | [-0.001, +0.042] |
| Quality: one-year growth part | -0.012 | [-0.022, +0.001] |
| Quality: safety part | +0.009 | [-0.008, +0.025] |
| Montier C (low = attractive) | -0.010 | [-0.018, +0.002] |

The growth part and the Montier C-score lean the wrong way.

**Readings, not findings:**
- **2009-11 was the strongest stretch for quality and profitability.** That was the post-crisis recovery, when a
  survivor-only universe would favour firms that came back.
- **The industry-neutral ICs are smaller than the raw ones.** Part of each composite's signal is an industry bet.
- **Removing size changes little.**
- **Composite correlations** (mean monthly Spearman):
  - quality and gross profitability 0.57;
  - quality and Piotroski 0.55;
  - value and the magic formula 0.49;
  - value and quality -0.11.

## Why this is consistent with the literature, and what it does not show

- **Weaker after publication.** Published anomalies weaken after publication and are weaker in large, liquid stocks
  (McLean and Pontiff 2016: across 97 predictors, long-short returns are 26% lower out of sample and 58% lower after
  publication). This universe is large caps only.
- **Biases stated in advance.**
  - Survivorship flatters junk.
  - Price-only returns penalise high-yield value stocks.
  - Both work against the composites, so the true effect could be somewhat larger than measured.
- **Horizon.** The test is about 20-session returns. Many fundamental premia are documented at 1-12 month holding periods
  with annual rebalancing. A longer horizon is a different hypothesis. It is not tested here, and testing it now, after
  seeing this result, would be a post-hoc choice: it needs its own pre-registration.

## Consequences

- **No frozen model, forecast, risk rule or verdict changes.**
- **The dashboard shows the metrics as context and risk flags.** The new tab P Fundamentals has the extended metrics and
  composites, with this result next to them, so no score looks predictive.
- **For a future model generation:** gross profitability and Piotroski are the only candidates worth carrying forward,
  and only as inputs to a pre-registered, forward-tested model, not as standalone signals.
- **Trial ledger:** E30 adds 6 tests (Holm) and 0 fitted parameters.

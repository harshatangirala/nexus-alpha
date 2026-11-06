# Fundamental analysis in NEXUS Alpha

Pre-registered as E22 (`docs/V3_RESEARCH.md`); owner decision D-039. Code in `src/nexus_alpha/fundamentals/`.

## Source and access

* **Source:** SEC XBRL `companyfacts` for every ticker of the 979-ticker universe (976 have XBRL facts). XBRL exists from
  2009, so fundamentals cover 2009-2022 in development.
* **Access:** SEC requires a declared User-Agent naming the requester (a generic one is refused with HTTP 403). The owner's
  value lives only in the git-ignored `local_sec.env`, loaded through `NEXUS_ENV_FILES`. Requests go only to
  data.sec.gov, at no more than 5 per second.
* **Raw files** are cached gzip-compressed in `artifacts/fundamentals/raw/` (git-ignored, about 231 MB). They also hold
  facts filed after the seal.
* **Seal:** `sec_facts.load()` drops every fact filed on or after 2023-01-03 unless the sealed evaluation is open in the
  process (tested).
* **Live use:** `sec_facts.load_live()` returns current facts and is labelled *live inference only*, like live Yahoo bars.
  It is never used to build a training, selection or evaluation dataset.

## Point-in-time rules

| Rule | Detail |
|---|---|
| Availability | A fact filed on date F is used for as-of sessions t >= F. The decision is at 09:00 New York on t+1, after anything filed on F |
| Restatements | For each concept and period, the latest value filed on or before t (as known then; tested) |
| Quarters | Direct 3-month facts; otherwise differences of year-to-date facts with the same fiscal-year start (Q2 = 6M - 3M, Q3 = 9M - 6M, Q4 = FY - 9M). 10-Q cash-flow statements are YTD only |
| TTM | The sum of the last four consecutive quarters (gaps 75-105 days). Prior TTM = the four quarters before those |
| Balance sheet | The latest instant on or before t; the prior value is the instant 320-410 days earlier |
| Staleness | Any item whose latest period ends more than 45 days before the company's latest period is missing. This prevents a tag the company stopped using from being carried forward |
| Units | Money in USD, EPS in USD/share, share counts in shares. A company whose balance sheet is mostly in another currency (14 Canadian, Japanese, Korean, Chinese and Indian filers) keeps no money facts at all: their few USD convenience translations are not mixed in |
| Prices | As-traded close = the vendor's split-adjusted close x the later split factors **that the vendor applied**: splits up to 2026-02-27, when its Yahoo split history was taken (later splits such as KLAC 10:1 in 2026-06 are not in its prices). Split history: the vendor's own Yahoo files, Yahoo through yfinance for 38 tickers without one, plus S03 (`fundamentals/splits.py`). Cover-page share counts are moved to the session's share basis |
| Shares | Cover-page count if dated within the company's latest period (a stale count is missing); otherwise the latest quarter's weighted diluted shares; otherwise TTM net income / TTM diluted EPS (multi-class issuers such as GOOG and META report cover counts per class). Weighted shares are averages, so a missing Q4 takes the fiscal-year average, never FY minus 9 months |
| Earnings surprise | SUE from the SEC study's S06, at the later of the filing's acceptance time and the time its EPS history was public |

**Verified on Apple, FY2019 10-K (as known 2019-11-15):**

| Item | Value |
|---|---|
| Revenue | 260.2B |
| Net income | 55.3B |
| Operating cash flow | 69.39B |
| Capex | 10.5B |
| Dividends | 14.12B |
| Buybacks | 66.9B |
| Total debt | 108.0B |
| Equity | 90.5B |
| Market cap (2019-12-31) | 1.30T |
| P/E | 23.6 |

## Share-basis guards (audit 3)

Market cap, EV and every price-based ratio are missing when:

* the snapshot comes from a **foreign private issuer** (latest report a 20-F or 40-F). Their statements are annual and
  often in local currency, and the cover page counts ordinary shares, not ADS. TSM, DEO, TM, VOD, NGG and GSK were
  2-10x too large before this guard. All fundamentals are missing for these rows (`foreign_filer` = 1);
* the ticker **shares its CIK with another listed class** (BRK.A / BRK.B): one company-wide count cannot be matched to
  one class's price;
* the cover-page count and the diluted-share reference **disagree by more than 1.5x**;
* the count is **below 1 million** (ETN's 2012 shell entity reported 100 shares);
* the count **breaks continuity**: on one split basis it is more than 1.25x away from the median of the counts accepted
  over the previous year, unless a filing at least 75 days later confirms the new level (a real merger or offering).
  This catches spin-offs that Yahoo records as splits (ABT 2013, x2.08; PENN 2013), filings restated for a split before
  its effective date (AOS 2013, NEOG 2018) and counts tagged in thousands (AKR, AAP, ETR, FCEL: x1000). It uses only
  earlier sessions.

`shares_basis_ok` records whether a share count was usable.

**Audit 3 result** (`scripts/verify/audit_fundamentals.py`, `reports/verification/fundamentals_audit.json`): every
hard check passes.

| Check | Result |
|---|---|
| Point in time: 40 random cells recomputed from facts filed on or before that date | 0 mismatches |
| Every split of 2009-2022 (324): market-cap jump matching the split factor | 0 |
| One-day market-cap moves > 50% | 193, all matched by the price move; 0 share-basis jumps (144 before the continuity guard) |
| Ten megacaps at 2021-12-31 vs public values (+-15%) | all pass (AAPL 2,914, MSFT 2,525, GOOG 1,969, AMZN 1,691, TSLA 1,062, META 962, NVDA 735, UNH 473, JPM 468, JNJ 450; $bn) |
| Market cap / NI vs price / EPS within 1.5x | 98.3% of 320,362 rows (the rest: TTM EPS mixing bases around splits) |
| Non-USD reporters with any ratio | 0 rows |
| Seal | panel ends 2022-12-30, evaluation closed |

Remaining known issue: `interest_coverage` has 4.4% of values beyond +-50 (tiny interest expense); harmless for the
rank features, shown as-is in the dashboard.

Coverage (share of universe rows with the ratio): earnings yield 36% in 2010, 80% in 2012, 85-91% from 2013 on.

## The 31 ratios (model features as within-session universe percentiles)

| Group | Ratio | Definition |
|---|---|---|
| Valuation | earnings_yield | TTM net income / market cap |
| | sales_yield | TTM revenue / market cap |
| | book_to_price | equity / market cap |
| | ebitda_to_ev | (TTM EBIT + TTM D&A) / (market cap + debt - cash) |
| | fcf_yield | (TTM CFO - TTM capex) / market cap |
| | dividend_yield | TTM dividends paid / market cap |
| | shareholder_yield | (TTM dividends + TTM buybacks) / market cap |
| | dcf_to_price | equity value from a 2-stage FCF DCF (growth = TTM revenue growth clipped to [-5%, 15%] for 5 years, then 2.5%; discount 9%) / market cap |
| Profitability | gross_margin | gross profit (or revenue - COGS) / revenue |
| | operating_margin | EBIT / revenue |
| | net_margin | net income / revenue |
| | ebitda_margin | EBITDA / revenue |
| | roe | net income / equity |
| | roa | net income / assets |
| | roic | EBIT x (1 - 21%) / (debt + equity - cash) |
| Growth | revenue_growth | TTM / prior TTM - 1 |
| | operating_income_growth | (EBIT - prior EBIT) / abs(prior EBIT) |
| | eps_growth | (TTM diluted EPS - prior) / abs(prior) |
| | asset_growth | assets / assets a year earlier - 1 |
| | share_issuance | latest-quarter diluted shares / a year earlier - 1 |
| Leverage and liquidity | debt_to_equity | debt / equity |
| | net_debt_to_ebitda | (debt - cash) / EBITDA |
| | interest_coverage | EBIT / interest expense |
| | current_ratio | current assets / current liabilities |
| | cash_to_assets | cash / assets |
| Efficiency | asset_turnover | revenue / assets |
| | receivables_to_sales | receivables / revenue |
| Quality and events | accruals | (net income - CFO) / assets |
| | piotroski_f | 9 signals: ROA > 0, CFO > 0, ROA up, CFO > NI, leverage down, current ratio up, no dilution, gross margin up, asset turnover up (missing when fewer than 6 inputs exist) |
| | altman_z | 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + 1.0 Sales/TA |
| | sue | standardised unexpected earnings (seasonal random walk), stale after 120 days |

Also computed: days since the latest 10-Q/10-K, and the multiples P/E, EV/EBITDA, P/S, P/B and P/FCF, plus market cap
and EV (dashboard only). Debt is total long-term debt (or non-current + current) plus short-term borrowings and
commercial paper; operating leases are excluded.

## Dashboard (Trade Desk section 4c)

* The three statements: TTM and latest balances versus a year earlier, with the change.
* The multiples.
* All 31 ratios with the stock's universe percentile that day (historical mode).
* Piotroski and Altman with their zones, and days since filing.
* An adjustable DCF: growth, discount rate, terminal growth and years, giving value per share and margin of safety.
* Live mode uses current SEC facts and the latest Yahoo close.

## Limits

* No XBRL before 2009, so fundamentals cover 14 of the 21 development years. Smaller filers phase in during 2009-2011.
* Survivor-only universe (as everywhere in NEXUS).
* Tags vary by company, and some items are missing for some firms (banks have no gross margin or current ratio). The
  model sees missing values and the present flag.
* The DCF uses fixed assumptions as a feature; in the dashboard it is analysis, not a validated signal.

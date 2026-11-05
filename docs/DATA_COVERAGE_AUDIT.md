# Data coverage audit (development period, sealed)

Produced by `scripts/audit/p1_ohlcv_coverage.py` (per-ticker table: `data_manifests/ohlcv_coverage_by_ticker.csv`), `scripts/audit/p1_news_*.py`, `scripts/data/build_news_index.py` and `scripts/audit/p1_sec_audit.py`. All bar reads pass through the seal: rows on or after 2023-01-03 are never returned; post-seal rows are known only as counts from parquet footers.

## 1. Bars

| | Daily | Hourly | Weekly |
|---|---|---|---|
| Tickers | 979 | 979 | 979 |
| Rows before the seal | 5,143,695 | 35,693,557 | 1,066,802 |
| Rows after the seal (count only, never read) | 926,134 | 6,435,958 | n/a |
| First date range | 2001-01-02 to 2021-11-10 | same | same |
| Last date before the seal | 2022-12-30 for every ticker | 2022-12-30 | 2022-12-30 |
| Missing sessions vs the XNYS calendar | 0 | 3,094 sessions with no hourly bars (Yahoo-filled daily bars) | n/a |
| Duplicates / OHLC violations | 0 / 0 | 0 / 0 | 0 / 0 |
| Sessions with fewer hourly bars than the calendar allows | n/a | 79,422 (median ticker 0.09%; worst 33%) | n/a |
| Sessions with more hourly bars than allowed | n/a | 9,018 (post-close prints on early-close days; removed by the r1.1 rule before rendering) | n/a |
| Consistency | 5,140,601 days from 1-minute data, 3,094 from Yahoo, 0 from the 2026 Yahoo extension | equals the daily bar exactly on 1-minute days | equals aggregated daily bars exactly (0 differences) |

**Timeframes are not independently sourced.** Weekly and hourly bars are aggregations of the same 1-minute feed as the daily bars, so the three views are nested observations of one price process, not three data sources.

**Tickers with full 2001-2022 history:** 764. Per-year availability (tickers with any bar): 789 in 2001, 857 in 2003, 945 in 2008, 950 in 2012, 960 in 2017, 979 in 2021-2022. Yahoo-filled days (no hourly bars) affect 192 tickers, at most 349 days (SMCI, which traded OTC from August 2018 to January 2020; those days are not decision dates in TriFrame's sample).

**Adjustment and point-in-time status.** Prices are split-adjusted with later split information (retrospectively adjusted), not dividend-adjusted. Chart images are scale-free, so split adjustment does not change them; dollar quantities (ADV, the $5 price screen) use adjusted prices, a small look-ahead in screening noted in TriFrame's review (D5). NEXUS labels are price-only.

**Exchange and currency.** NYSE 606, NASDAQ 329, NYSE American 4, "F" 2 (as recorded in TriFrame's universe file), 38 additions without an exchange field; all USD; 32 ADRs.

## 2. Universe and survivorship

The 979 stocks are **current survivors** screened from today's listing directory (TriFrame data card): no delisted names, no historical index membership. Results are survivorship-biased: risk is probably understated and long-side returns flattered, most in the early years. TriFrame's data card estimates the universe covers 57.9% of S&P 500 member-days 2001-2026. NEXUS describes nothing as survivorship-free. TriFrame's 23 unverified delisted names (`extra/`) are not used.

## 3. The NEXUS sample frame

TriFrame's s500 chart sets: **236,520 samples, 500 stocks, as-of dates 2001-10-22 to 2022-12-01**, about one decision every 10 sessions per stock (staggered), so roughly 45 stocks share a date. Every row is in TriFrame's sealed index. After purging:

| Block | Rows | Share with news in the 5-day window |
|---|---|---|
| train (2003-2015) | 143,695 | 30% |
| inner (2016) | 10,671 | 62% |
| valid (2017-2018) | 22,783 | 64% |
| calib (2019) | 10,842 | 71% |
| test (2020-2022) | 35,954 | 45% |
| purged or before 2003 | 12,575 | n/a |

Missing features: daily 0%, hourly 0%, weekly 0.6% (warm-up of 50-week averages in early rows).

## 4. News coverage

| | Value |
|---|---|
| Corpus rows / subsets | 57,114,479 / 24 |
| Corpus publication dates | 1990-01-01 to **2025-09-06** |
| Subsets with explicit ticker tags | FNSPID, Benzinga, SentARL, Yahoo Finance (two feeds), Reddit finance |
| Unique articles tagged with a universe ticker | 3,947,549 (all 979 tickers have at least one) |
| Universe-tagged articles per year | 4-11 thousand 1998-2009; 111k-521k 2010-2019; 301k 2020; 105k 2021; 158k 2022; 207k 2023; **258 in 2024**; 79k in 2025 |
| Timestamp precision of universe articles | minute 595,205; day 430,068; "minute" stamped exactly 00:00:00 UTC (treated as day) 2,922,276 |
| s500 samples with at least one article available in the 5 days before the decision | 91,936 of 236,520 (39%); 11-17% per year before 2010, 34-71% from 2010 |

Mapping is by explicit ticker tag only (no company-name matching); renamed tickers are therefore missed (for example articles tagged FB before the META rename). General-news subsets without tags (NYT, Bloomberg/Reuters, All-the-News and others) are not mapped.

## 5. SEC coverage

190,947 events (2000 to 2026-07-31) for 500 firms, all inside the price universe; 262 of the 500 s500 stocks are covered. A sample outside SEC coverage has `sec_covered = 0` and NaN SEC features, never zero counts.

## 6. Coverage regimes used in experiments

1. **Charts + indicators**, 2003-2022, all s500 rows.
2. **Multimodal**, rows with news present and/or SEC covered (mostly 2010 onward).
3. **Matched comparisons** for H2 restrict both models to the same multimodal rows.
Missing news is never treated as neutral news: it is an explicit mask.

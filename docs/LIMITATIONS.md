# Limitations and known defects

## Data

1. **Survivorship.** 979 current survivors from one vendor lineage (Finnhub-derived 1-minute data plus Yahoo). No delisted firms, no point-in-time index membership. Long-side returns and the equal-weight benchmark are flattered; risk is understated. Nothing here is survivorship-free.
2. **Licences.** The bar data declare no licence; the news corpus is non-commercial research only with no text redistribution; the Loughran-McDonald dictionary restricts redistribution; the books are copyrighted. No data, features, embeddings, article or book text, or weights may be published. The GitHub repository, if created, must contain code and documentation only.
3. **Retrospective adjustment.** Prices are split-adjusted with later split information; dollar screens use adjusted prices. Labels are price-only (no dividends): 0.68% of 5-day labels differ from TriFrame's dividend-inclusive labels.
4. **Timeframes are not independent sources.** Weekly and hourly bars are aggregations of the same 1-minute feed as the daily bars.
5. **News coverage is uneven.** Explicit ticker tags only: renamed tickers are missed (for example FB before META), general news is unmapped, coverage before 2010 is thin (11-17% of samples have any article), 2024 is nearly empty and the corpus ends 2025-09-06. 2.92 million rows stamped at midnight are treated as day-level, so their true intraday timing is unused.
6. **SEC coverage** is the SEC study's 500 firms: 262 of the 500 sample stocks. Public availability is acceptance + 5 minutes by assumption, not observed.
7. **Market context** comes from the survivor cross-section, not from a real index or VIX.

## Design

8. **Sample frame.** NEXUS reuses TriFrame's s500 chart sets (one decision every ~10 sessions per stock) so stored embeddings can be reused. That limits the sample to 500 stocks and those dates.
9. **Charts are OHLC bars, not candlesticks**, scaled to each window (so they cannot show the volatility level), resized to 224² (TriFrame's review flags resolution loss). A candlestick renderer and native-resolution inputs are registered, not run.
10. **Fine-tuned chart embeddings** may be evaluated only from 2017; their weights saw TriFrame targets through 2014.
11. **Text.** The v1/v2 models read text through classical features (TF-IDF/SVD, lexicon tone). Transformers enter in E28 (D-053): FinBERT on headlines and ChronoBERT-2010 on filings, with fixed weights (no fine-tuning). ChronoBERT reads only the first 256 tokens of each filing after safe-harbour paragraphs are dropped, and the SEC study masks most firm names (some remain). Filing blocks exist only for the SEC study's 500 firms. The SEC price-impact expert's trees are chosen on 2016 events, so its predictions for 2016 are mildly optimistic, and 2016 is the fusion's early-stopping block. There is no live path for these blocks: the forward test has no news feed, and the expert needs the SEC study's event features.
12. **Seeds.** The E03 variant comparison used one seed per variant. The selected configuration was re-run with 5 seeds in E06 (validation spread ± 0.0010), so differences between E03 variants smaller than about 0.002 are within seed noise.
13. **The 20D and 1M horizons differ by one session**, so they are near-duplicates by construction.
14. **Hourly triggers** are evaluated only on the as-of session's completed bars; intrabar decisions (rule E141) are out of scope.
15. **Return ranges.** The quantile heads' 10-90% band is slightly too wide (coverage 0.83-0.84 on validation, 0.82-0.84 on test, nominal 0.80). It is shown as a range, not as a direction call, and the median is close to zero by construction.

## Evaluation

16. **The test years 2020-2022 were already read** by TriFrame for its own targets. NEXUS has also scored them in E01, E02, E03, E06, E07, E08 and E10. Its test results are a heavily re-used historical diagnostic, not evidence of generalisation.
17. **The sealed period 2023-2026 is not virgin**: TriFrame opened it twice and it is the SEC study's spent locked test. A clean confirmation needs forward data.
18. **E01 and E02 printed test-block metrics before the pre-registration** (disclosed in the specification).
19. **RAG gold pages are lexical**, which can favour BM25; five evaluation queries had no gold page.
20. **The backtest** has no quotes (spreads bounded by cost scenarios), no market impact, no partial fills, no borrow availability; stops inside a bar assume the stop came first.
21. **Calibration hurt on the test years.** The pre-registered calibration (temperature at 10D-1M, fitted on 2019) was slightly worse than none on 2020-2022. The procedure was not changed after seeing this.
22. **The direction-skill gate is statistical.** It opened 5D-1M on 2017-2018 evidence, and that skill was absent at every one of those horizons on 2020-2022. Since D-057 the dashboard requires both blocks, so no horizon trades direction until forward evidence re-opens it. An economic gate is a candidate for the next pre-registration.

## Engineering

23. Live mode depends on Yahoo Finance, whose hourly bars can differ slightly between requests.
24. The live mode does not fetch SEC filings.
25. The final-evaluation script for the sealed period is written and rehearsed on 2022 (`scripts/sealed/`, `docs/SEALED_EVALUATION_PLAN.md`); only the owner may run it on the sealed data, once.
26. Screenshots are not read by a trained model. Legend values were read by Gemini and checked against real bars in E17 (`docs/V2_RESULTS.md`); that read is live, unvalidated at scale, and never changes the frozen probabilities.
27. **Live inputs come from a different vendor than the training data.**
    * **The setup.** The forward test forecasts from Yahoo bars; every model was trained on the TriFrame vendor's bars.
    * **Measured on 2022 for Alpha158** (`reports/verification/alpha158_live_parity.json`):
      * price features agree closely (median correlation 0.9989);
      * volume features agree less (median 0.92, minimum 0.72), because Yahoo's consolidated daily volume is not the
        vendor's sum of minute bars.
    * **Effect on v3_alpha.** Its 1D/2D ranking agreement falls to 0.935/0.947, so those horizons are not deployed (D-049).
    * **Other models.** The F0 features of v1.1, v2 and v3_fund include volume-based indicators and face the same
      shift. It was not measured for them.
28. **News coverage is not stationary (audit 9, A-99).** Company news exists for 30% of training stock-dates, 64% of 2017-2018 and 45% of 2020-2022 (37% in 2021): the news source thins out in the test years. Any test of the text branch on 2020-2022 rests on fewer rows than validation suggests, and the forward test has no live news at all (the text branch is never exercised there). Notebook section 13A shows coverage by year.
29. **The charts add nothing once indicators are present (audit 9).** Adding the chart embeddings to indicators and language changes the mean log loss by +0.00000 on 2017-2018 and +0.00009 [-0.00123, +0.00143] on 2020-2022, well inside the seed spread. The vision branch is kept in the frozen models, which are never re-fitted.
30. **Published fundamental scores do not rank this universe at 20 sessions (E30).** Value, QMJ-style quality, gross profitability, the magic formula, Piotroski and Beneish were tested once, point in time, on 164 month-ends (137-151 months per composite, 2009-2022): none survives Holm (best: gross profitability IC +0.018, Piotroski +0.014). The Fundamentals tab shows them as description and risk flags, not as signals (docs/E30_RESULTS.md).
31. **XBRL tag coverage (A-112).** COGS, receivables, operating income and D&A are missing for a third to a half of firm-dates because companies use alternative or company-specific tags. Ratios that need them are missing, not estimated. Widening the tag map is deferred to the next model generation because the frozen v3_fund reads these fields.
32. **Price-only returns.** The development bars are split-adjusted but not dividend-adjusted, so every historical return omits dividends (about the dividend yield times h/252 for an h-session horizon).

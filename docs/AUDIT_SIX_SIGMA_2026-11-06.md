# Two-pass internal audit with SWOT, fishbone and Six Sigma DMAIC (2026-11-06)

Scope: every section of NEXUS Alpha (data, point-in-time rules, labels, features, models, calibration, evaluation,
backtests, forward test, sealed evaluation, inference and dashboard, security, operations, documentation).

**Ground rules.**
* The frozen models (v1.1, v2, `market_e19`) are not modified: they are under forward test.
* Improvements to models are new, pre-registered candidates, selected on validation only.
* The sealed 2023+ period was not touched.
* Every finding has evidence and a fix, or an explicit reason it cannot be fixed now.

## 1. SWOT #1 (before this audit)

| Strengths | Weaknesses |
|---|---|
| Strict point-in-time rules, read-time seal, purged splits; pre-registration and a decision log (D-001..D-036) | Pooled direction skill about 0.50 (AUROC) for both stock models |
| Three frozen models in a hash-chained, publicly timestamped forward test | Long horizons (3M-1Y) worse than the base rate on 2020-2022 |
| Sealed scorer rehearsed: 309 features exact, predictions reproduce to 0.0006 | Macro/market blocks never beat plain features at stock level |
| Free, reproducible data path; 94 automated tests | Survivor-only universe; one historical price vendor |
| Honest negative results (E11, E20) | Forward test runs without news or SEC text |

| Opportunities | Threats |
|---|---|
| Cross-sectional ranking (market-neutral) rather than direction | Overfitting through re-use of 2020-2022 |
| A live point-in-time news feed (QVeris / Alpha Vantage) as a new variant | Data-source drift (Yahoo vs vendor), Yahoo blocking |
| Longer-window calibration; earnings-event (PEAD) tests | Free-tier limits (Gemini 20/day); single LLM model |
| The forward test across about 500 stocks a night gives fast, clean evidence | Regime change; costs eat thin edges |

## 2. Fishbone (Ishikawa): "No validated, tradeable edge yet"

```
 DATA (materials)                 METHODS                            MEASUREMENT
 - survivor-only universe         - 3-class direction target         - pooled AUROC hid ranking skill
 - news on 46% of samples,          (wrong objective for the         - DSR not reported (A-33)
   none in 2024                     skill that exists)               - forward label warm-up (A-28)
 - one price vendor               - union carry rule (D-033)         - 6M/1Y "validated" on <4
 - v2 ret_ columns (A-29)         - 1-month blocks on overlapping      independent windows (A-31)
 - GMD single vintage               labels (D-032)
          \                       - calibration on one year (2019)        /
           \                              \                              /
            >=====================================================> NO VALIDATED, TRADEABLE EDGE
           /                              /                              \
 INFRASTRUCTURE (machines)        PEOPLE / PROCESS                   ENVIRONMENT (market)
 - CPU only (BLAS fixed D-027)    - decisions after results           - prices close to a random walk
 - Yahoo hangs (fixed D-028)        (D-031..D-033, disclosed)         - regimes: 2020 crash, 2022 bear
 - Gemini free tier, one model    - two reporting slips (fixed)       - 20 bp costs vs 1D edges
 - no hosted CI (nightly local)   - owner-only steps (seal, QVeris)
```

## 3. Six Sigma DMAIC

### Define: critical-to-quality (CTQ) characteristics

| CTQ | Target |
|---|---|
| 1. Point-in-time integrity | 0 leakage defects |
| 2. Forecast quality | log loss below the base rate; direction or ranking skill with a CI above 0 on validation and confirmed forward |
| 3. Forward-test reliability | at least 99.5% of tickers logged per session, 0 late rows, chain intact |
| 4. Forward label fidelity vs training | at least 99% agreement |
| 5. Reproducibility | 0 feature mismatches; max probability difference below 0.001 |
| 6. Security | 0 secrets in public history |

### Measure: baseline and after (defects per million opportunities and sigma level, 1.5-sigma shift)

| CTQ | Measure | Before | After |
|---|---|---|---|
| 1 | Planted-leak tests, seal gate tests | pass | pass (gate tests added) |
| 2 | Pooled direction AUROC (v2, 2020-22) | 0.48-0.52 | unchanged (frozen) |
| 2 | Daily rank IC (F0 LightGBM, validation) | not measured | **5D 0.019, 10D 0.020, 20D 0.032**, CIs above 0 |
| 3 | Tickers failing per session | v1.1 6,000 DPMO (4.0 sigma); v2 2,000 (4.4 sigma) | SPC p-chart now monitors it nightly |
| 3 | Late rows | 0 | 0 |
| 4 | Label agreement, forward vs training (Jan-Mar 2022) | 98.4% (16,070 DPMO, 3.6 sigma) | **99.2% (7,750 DPMO, 3.9 sigma)** after A-28 |
| 5 | Rehearsal feature cells mismatching | 0 of 3.5 million (6+ sigma) | 0 (re-run) |
| 5 | Lean sign agreement vs stored predictions | v1.1 99.966% (4.9 sigma), v2 99.997% (5.5 sigma) | unchanged |
| 6 | Secret patterns in full git history | 0 | 0 (re-scanned) |

### Analyze: root causes

1. **The skill is in relative performance, not direction.** The same models whose pooled up/down AUROC is about 0.50
   rank stocks on the same day with an IC of 0.02-0.03 (validation). A common market move is unpredictable for single
   stocks (E11, E19), but it cancels in a cross-sectional ranking.
2. **The edge is thin against costs.** The within-date top-minus-bottom quintile spread after 20 bp is not significantly
   positive at any horizon, and it is negative at 1D (-10.3 bp, CI [-17.4, -3.4]).
3. **Validation windows are too short for long horizons** (2017-2018 holds fewer than 4 independent 6M windows and about
   one 1Y window), so 3M-1Y skill claims cannot be supported, and calibration on one year (2019) hurt 10D-3M in 2020-22.
4. **Forward-test conditions differ from training.** No news or SEC text live, and Yahoo prices differ from the vendor
   (median 2.6 bp).

### Improve: actions taken in this audit

| Action | Effect |
|---|---|
| A-28 forward scorer: 3-year label warm-up | label disagreement halved (1.6% to 0.8%) |
| A-29 v2 frame `ret_` columns repaired, builder fixed | data integrity; labels byte-identical |
| A-30 v1 live decision time beyond the calendar | tz-correct cut-off |
| A-31 6M/1Y no longer shown as validated | honest dashboard |
| A-32 market calendar to 2030 | no 2028 breakage |
| A-33 E15 deflated Sharpe reported (0.0001) | honest economics |
| E21 pre-registered cross-sectional ranker | no candidate beat the baseline; baseline B0 kept |
| D-038 ranker in the forward test | rank IC measured nightly on all ~500 stocks |

### Control: what keeps it fixed

* `scripts/shadow/spc_monitor.py` runs nightly:
  * p-chart of ticker failures (3-sigma);
  * zero-late-rows rule;
  * X-mR charts of daily log-loss gain with Western Electric rules 1-2;
  * report in `shadow/spc_report.json`.
* The full test suite (94 tests) runs nightly (`artifacts/logs/nightly_tests.log`), alongside the existing frozen-model
  golden test and the seal-gate tests.
* Hash-chained forecasts are pushed publicly every night.
* Every post-result decision goes in the decision log with disclosure.

## 4. Audit 1 register (section by section)

| Section | Status | Findings / evidence |
|---|---|---|
| Data sources | OK, with structural limits | Survivor-only universe; news on 46% and SEC on 53% of samples; a 2024 news gap (258 articles) |
| Point-in-time rules | OK | Seal gate, purge and embargo tests pass; 0 leakage defects |
| Labels | Fixed | A-28 forward warm-up; A-29 v2 `ret_` columns |
| Features: numeric/indicators | OK | Strongest branch (E03: numeric alone 1.0605 vs all fused 1.0592) |
| Features: computer vision | OK | Embeddings reproduce TriFrame (cosine at least 0.99997); small gain (H1 +0.0010, n.s.) |
| Features: NLP | OK, weak | News alone barely beats the prior; FinBERT helps only on top of B at 1D (E16) |
| Market/macro | OK (negative result) | No stock-level gain (E11, D-033); SPY 20D read only (E19) |
| Models | Finding | Wrong objective for the available skill: E21 pre-registered and run |
| Calibration | Finding | 2019-only calibration hurt 10D-3M on 2020-22 (frozen; v3 should use a longer window) |
| Statistics | Fixed | A-31 skill claims at 6M/1Y; D-032 blocks |
| Backtests | Fixed (reporting) | A-33: DSR about 0; no edge demonstrated |
| Forward test | Fixed and controlled | A-28; SPC; ranker logging |
| Sealed evaluation | OK | Rehearsal and dry run pass; owner-gated |
| Inference/dashboard | Fixed | A-30, A-31 |
| Security | OK | 0 secrets in history; keys only in owner files; config key procedural |
| Operations | Fixed | A-32 calendar; nightly tests |
| Documentation | Fixed | V2_RESULTS, V3_RESEARCH, CHIEF_AUDIT_REPORT updated |

## 5. Audit 2 register (re-verification after fixes)

| Check | Result |
|---|---|
| Full test suite | **94 passed**, 0 failed |
| Sealed scorer dry run (2022, end to end) | passed again (234 s) |
| Frozen v1.1 golden test | passed (outputs unchanged) |
| Label parity after A-28 | 99.2% (from 98.4%) |
| Trade Desk headless run | no exceptions; 6M/1Y display fixed |
| Secrets rescan of public history | 0 |
| SPC first run | no alerts; coverage, late rows and chains in control |
| v2 forecaster with `rank_score` | dry run passes; scorer reports rank IC once data matures |

**Residual risks that cannot be fixed inside the project today** (reasons stated):

| Risk | Why it cannot be fixed today |
|---|---|
| Survivorship bias | needs delisted-stock history; no free source |
| No live news/SEC in the forward test | needs a paid or authenticated feed (QVeris sign-in or the owner's `SEC_USER_AGENT`); adding it is a new pre-registered variant |
| Direction skill is absent and ranking skill is thin against costs | only forward evidence and the sealed period can settle it; more fitting on reused data would overfit |
| Test period reused many times | by nature; the forward test and the sealed period are the remedy |
| Gemini single free model; no hosted CI | external; mitigated by fallback logic and nightly local tests |

## 6. SWOT #2 (after both audits)

| Strengths | Weaknesses |
|---|---|
| Everything in SWOT #1, plus: measured cross-sectional skill (IC 0.02-0.03); SPC control of the forward test; label fidelity 99.2%; 94 tests nightly | Edge too thin to pay costs on development data; long horizons unsupported; survivor universe |

| Opportunities | Threats |
|---|---|
| The nightly universe-wide rank IC will confirm or reject the ranking skill within weeks; a market-neutral, longer-horizon (10D-20D) book if confirmed; a live news feed as a pre-registered variant; longer-window calibration in v3 | Over-reading short forward samples (SPC needs at least 20 sessions); data-source drift; costs |

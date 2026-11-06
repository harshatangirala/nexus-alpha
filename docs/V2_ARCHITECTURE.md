# NEXUS Alpha v2: market, macro and trade-type architecture (pre-registered 2026-11-06)

Status: **design and pre-registration**. Written before any v2 model was fitted on the new features. Changes after the first
v2 result must be logged in `docs/DECISION_LOG.md` with the reason, and count as new trials.

## 1. What changes from v1.1

| Area | v1.1 (frozen) | v2 |
|---|---|---|
| Horizons | 1D, 2D, 5D, 10D, 20D, 1M | **1D-20D** (1D = the intraday open-to-close trade; every swing length 2-20), **3M, 6M, 1Y** (`targets_v2.py`) |
| Market context | none (stock-only features) | S&P 500 via **SPY** 1H/1D/1W bars from TriFrame's own minute source, sector/size/rates/credit/commodity ETFs, Yahoo VIX, Treasury yields, dollar, oil, gold, copper (`macro/features.py`) |
| Macro | none | **GMD 2026_09** annual (publication-lagged), **FRED/ALFRED** real-time vintages (when `FRED_API_KEY` is set) |
| Text | TF-IDF/SVD news and SEC, Loughran-McDonald tone | adds local transformer encoders (FinBERT-class, Colab GPU) as a pre-registered comparison; **Gemini** reads user-supplied text live (dashboard only, logged for forward testing, never in backtests) |
| Screenshots | displayed only | **ticker-anchored**: the legend's date and OHLC are checked against real bars; the model always uses bars rendered by the training renderer |
| Dashboard | ticker + horizon table | **Trade Desk**: trade type (intraday / swing N days / long-term 4W, 3M, 6M, 1Y), 3 screenshots, 1-10 texts |

v1.1 stays frozen (registry/frozen_models/v1.1.json) and keeps running in the forward shadow test. v2 code is additive: v1
modules are not edited in ways that change v1.1 outputs; `tests/test_frozen_v11_golden.py` fails if they do.

## 2. Point-in-time rules (pre-registered)

| Source | Available for decision at 09:00 NY on t+1 if ... | Look-ahead risk left |
|---|---|---|
| ETF bars (SPY etc.) | bar closes at or before 16:00 NY on t | none (prices are final) |
| Yahoo market series | close of session t (futures settle before 16:00) | none |
| FRED/ALFRED | vintage released on day R <= t (>= 1 day after any release time) | none; dates before a series' first vintage are missing |
| GMD annual | year Y from 1 May of Y+1 | revisions: only the 2026_09 vintage is held (stated limitation); forecasts and crisis dummies excluded |
| Stock-relative | trailing windows ending at t; sector re-inferred each session from trailing correlation | none |

Universe and survivorship are unchanged from v1 (survivor-only, one vendor).

## 3. Sources considered and how each is used

| Source (from the owner's list) | Access found (2026-11-06) | Use |
|---|---|---|
| GMD zip (2026_09) | local, 80 CSVs, annual | model features (`macro/gmd.py`); Research Use Terms v1.1: academic, non-commercial, no re-hosting, cite NBER WP 33714 |
| IMF data portal | SDMX API behind a portal sign-in; old JSON API offline | via GMD (IMF WEO/IFS are GMD sources) and via DBnomics for display |
| World Bank | keyless API, annual | via GMD (WDI is a GMD source); display |
| DBnomics | keyless API, mirrors BLS/BEA/Fed/IMF/OECD/ECB; revisions archived but no documented vintage query | dashboard context only (no reliable vintages, so not in the model) |
| FRED/ALFRED | free key required (keyless CSV refuses automated clients) | model features with exact vintages once the owner sets `FRED_API_KEY` |
| Trading Economics, CEIC | paid / enterprise | not used (no subscription) |
| QVeris hosted MCP | registered in Claude Code (needs the owner's browser sign-in); pay-per-call marketplace, providers not documented | interactive research only; not a reproducible training source |
| awesome-ai-in-finance, gitnux list, quant.SE thread | link lists / marketing list / unreachable | background reading; no data |
| Owner's framework (`institutional_quant_trading_framework.md`) | local | architecture menu for E13 (TCN/GRU/Transformer sequence branch, HMM regimes, purged CV, PBO/DSR) |

## 4. Pre-registered experiments

Blocks as v1 (train 2003-15, inner 2016, valid 2017-18, calib 2019, test 2020-22, sealed 2023+). Per-horizon purge for
3M/6M/1Y. **Selection uses validation only.** Test is a reused diagnostic (it has been scored by TriFrame and six NEXUS
experiments) and never drives a choice.

* **E11 macro ladder (tabular).** Multinomial logistic (L2 1e-3, standardised) and LightGBM (E07 parameters, early stopping
  on inner 2016 only) on nested feature sets: F0 v1 numeric; F1 = F0 + `rel_*`; F2 = F1 + `mkt_*`; F3 = F2 + `gmd_*`;
  F4 = F3 + `fred_*` (run when the key is available). Nine representative horizons (D-026: 1D; 2D, 5D, 10D, 15D, 20D;
  3M, 6M, 1Y); the chosen set is then fitted on all 23 horizons in **E18** for the blend. Report valid and test log loss, gain vs F0 with
  month-block bootstrap 95% CIs, up-vs-down AUROC. **Keep rule:** a block is kept if its mean validation gain over the
  previous rung is > 0 with a bootstrap lower bound > 0 in at least one of the three horizon groups (1D intraday, 2D-20D swing,
  3M-1Y long-term) and is not negative (upper bound < 0) in any group. Controls: date-permutation placebo of the `mkt_*`,
  `gmd_*`, `fred_*` blocks (shuffle whole dates) must remove the gain.
  *Clarification (2026-11-06, before E11 ran):* the rule is applied per model family. A block enters the carried-forward
  feature set if it passes for at least one family whose placebo also passes (placebo vs F1 does not meet the keep rule).
  Gains are measured on the nested ladder as run (each block on top of all earlier blocks), whether or not an earlier
  block was kept. If FRED vintages arrive after E11, F4 is run as a separate rung on top of F3 and counted as a new trial.
  *1Y labels (2026-11-06, after the v2 frame was built and before any E11 result):* the per-horizon purge leaves no 1Y
  labels in the one-year blocks inner (2016) and calib (2019), because every 1-year window there reaches the next block.
  Therefore LightGBM's 1Y tree count is inherited from the previous horizon (6M, same feature set) instead of early
  stopping, and 1Y is left uncalibrated in E14. 1Y validation (11,708 labels) and test (24,575) are unaffected. A first E11
  launch was stopped at its first horizon to apply this rule; no result was read.
* **E12 v2 fusion.** The E06 network with one head per v2 horizon and the kept numeric blocks added to the numeric
  modality; image, news, SEC modalities as v1.1; 5 seeds; E06 settings (no new tuning).
* **E13 sequence branch.** A small causal TCN and a GRU over the last 63 daily bars of the stock and SPY (scale-free
  inputs), trained per E06 settings; enters only through the blend.
* **E14 v2 blend and freeze.** Simplex blend of E11/E12/E13 components per horizon group on validation, refit on
  2003-2018 with fixed epochs/trees, calibration chosen on 2019 halves, quantile coverage check, then **freeze v2**.
* **E15 v2 backtest** with the E04 v1.1 engine (orders, fills, costs, kill-switch, gross cap unchanged) on the deployed
  v2 bundle's calibrated probabilities, one horizon per trade type (1D; 5D, 10D; 20D, 3M). Policy thresholds are frozen
  on **2019** with >= 30 trades (D-025, stated before E15 ran), then 2020-2022 under base, stressed and severe costs;
  benchmarks: equal-weight survivors and SPY buy-and-hold with dividends.
* **E17 screenshot legend reader** (validates D-019's screenshot check, not a forecast): Gemini reads the O/H/L/C legend
  of TriFrame's 66 real TradingView screenshots (22 stocks x 1W/1D/1H); accuracy against the hand-recorded values.
* **E16 text encoder comparison** (added 2026-11-06, before any E11/E16 result): on the E11-kept feature set, compare
  (a) + v1 news features (TF-IDF/SVD, Loughran-McDonald tone) with (b) + FinBERT features (`fb_*`: cluster-weighted tone,
  min/max, negative share, count, 16-d PCA of the mean embedding fitted on training rows) and (c) both. Same models, same
  bootstrap, same keep rule; reported on all rows and on rows with news. FinBERT weights are fixed (ProsusAI/finbert).
* **E19 market-direction model** (added 2026-11-06, D-029; written after E11's 1D-10D validation log losses had printed and
  before any E11 keep decision). One row per session in the v2 frame (2001-2022; blocks as above), label = SPY's own
  v2 label (same rule, every horizon in D-026's set), features = the date-level values the stock models saw. Nested
  sets: P (training class prior); M1 SPY technicals (`mkt_spy_*`); M2 = M1 + the other `mkt_*` (without `mkt_n`,
  D-030); M3 = M2 + `gmd_*`; M4 = M3 + `fred_*`; placebo = M4 with whole feature rows permuted across dates (seed 11).
  Models: logistic (median-imputed, standardised; per-row L2 chosen on inner 2016 from {1e-3, 1e-2, 0.1, 1, 10}) and
  LightGBM (num_leaves 4, min_data_in_leaf 60, learning rate 0.03, feature/bagging fraction 0.7/0.8, L2 1, early
  stopping on inner). 1Y has no inner labels: it takes 6M's penalty and tree count. Comparisons M1 vs P, M2 vs M1,
  M3 vs M2, M4 vs M3, M4 vs P, placebo vs P. Keep rule as E11, per family, on validation, with two changes fixed now:
  (1) the bootstrap resamples moving blocks of max(1, ceil(h/21)) months, because overlapping labels make one-month
  blocks too narrow for long horizons; (2) a horizon enters the keep decision only if validation holds at least 4
  non-overlapping label windows (labelled sessions / h >= 4); others are reported only. The placebo must not pass vs P.
  Use: if a block passes for a family whose placebo passes, the market model is refitted on 2003-2018 with the selected
  penalty / scaled tree count and shown in the Trade Desk as a separate market-direction read with its validation
  skill. It is not blended into the stock forecast (stock-level stacking would be a new experiment). If nothing passes,
  the Trade Desk shows the market and macro readings as context with "no validated predictive skill (E19)".
* **Trial budget:** E11 <= 10 feature-set x model fits per horizon; E12 1 configuration x 5 seeds; E13 2 architectures;
  E14 one blend search; E16 3 sets x 2 models per horizon; E19 6 sets x 2 models x 9 horizons (logistic: 5 penalties each). Every trial is counted for the deflated Sharpe ratio and PBO.

## 5. Joint sealed evaluation (owner decision 2026-11-06)

The seal is opened **once**, after v2 is frozen, scoring v1.1 and v2 together on 2023-01-03 onward under a plan written
before opening (`docs/SEALED_EVALUATION_PLAN.md`, to be written at v2 freeze): per-horizon log loss vs the training prior,
up-vs-down AUROC, quantile coverage, and the frozen backtest policies against SPY buy-and-hold. The owner runs the unlock
(`seal.unlock`) in their own terminal; code never creates or guesses the key.

## 6. Language models

* **In the model (backtestable):** classical text features (v1) and local encoders run on Colab GPU with weights fixed
  before fitting; their knowledge cut-off is reported. A modern LLM is not used for historical features because it has
  read text about the outcomes of 2003-2022.
* **In the dashboard (live only):** Gemini returns a structured read of the user's texts and screenshots (direction lean,
  catalysts, risks, confidence, and the legend values it can read). It is shown beside the model, never blended into
  it, and every read is logged so it can be scored in the forward test.

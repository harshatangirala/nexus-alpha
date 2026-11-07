# Kronos in NEXUS Alpha

Owner decision D-043; pre-registration E24 in `docs/V3_RESEARCH.md`. Code in `src/nexus_alpha/kronos/`.

## What it is

Kronos is a family of decoder-only foundation models for financial candlesticks (Shi et al., *Kronos: A Foundation
Model for the Language of Financial Markets*, arXiv:2508.02739, AAAI 2026).
* **Tokens:** a tokenizer quantises each OHLCV bar into two hierarchical tokens.
* **Sampling:** a transformer samples the next bar's tokens, and repeating this gives future candle paths.
* **Licence:** code and weights are MIT.

| | Used here |
|---|---|
| Code | `shiyu-coder/Kronos` commit `67b630e`, vendored in `src/nexus_alpha/kronos/vendor/` with one documented import change (`VENDORED.md`; a test checks the files against the recorded hashes) |
| Weights | `NeoQuasar/Kronos-small` and `NeoQuasar/Kronos-Tokenizer-base` (also mini and its 2k tokenizer) at pinned revisions, safetensors only, SHA-256 in `artifacts/kronos/manifest.json` (git-ignored) |
| Model | Kronos-small, 24.7M parameters, context 512 (the smallest size the paper evaluates) |

## The deciding fact: contamination

From the paper (verified in the PDF):
* "The pre-training data for Kronos extends up to June 2024" (p. 14).
* The data table on p. 21 lists the NYSE (7,073 instruments) and Nasdaq (8,725), stocks and ETFs, daily frequency
  included, from 2000-01-01.

So every NEXUS development year (2003-2022), and the first 18 months of the sealed period, are inside Kronos's
training data. Three consequences:
1. Kronos output is **never a feature** of a trained NEXUS model.
2. A historical result is a **necessary condition only**: contamination can inflate skill but cannot hide it (E24).
3. The **forward test** is the only evidence.

## Engine (`engine.py`)

* **Inputs:** exactly as `KronosPredictor.predict`. The last 400 daily bars up to and including the as-of session
  (point in time; tested) give open, high, low, close, volume and amount = volume x mean OHLC. Each column is z-scored
  over the window and clipped to +-5. Time features come from the market calendar.
* **Every path is kept.** The vendored `auto_regressive_inference` returns the mean of its samples; probabilities need
  the whole distribution.
* **Seeding.** Token draws are inverse-CDF draws from per-stock random streams seeded by (ticker, as-of date). They are
  deterministic and do not depend on which stocks share a batch. The distribution is the same as `torch.multinomial`.
* **Exact speed-ups:**
  * a key/value cache;
  * the shared context prefilled once per stock and copied to its 32 paths;
  * cached cross-attention projections. They are exact because the vendored cross-attention rotates at position 0,
    where cos = 1 and sin = 0.
* **Checks** (`scripts/kronos/parity_and_speed.py`, `reports/verification/kronos_parity_speed.json`, and tests):
  * under deterministic top-1 decoding, the NEXUS sampler equals the vendored predictor, maximum relative difference
    0.0;
  * the cached sampler equals the uncached reference, 8 of 8 seeded paths identical;
  * Kronos-small takes 2.9 s per stock for 32 paths and 20 sessions (about 24 minutes for 500 stocks), 17-19x faster
    than the reference loop.
* **Probabilities** (`label_probabilities`): the v2 label rule applied to each path. R_h = close[t+h] / open[t+1] - 1,
  with sideways within max(0.5 sigma_t sqrt(h), 10 bp), counts smoothed by 0.5 per class. `rank_score` is the mean path
  return.

## E24 result (2026-11-07)

On the contaminated 2017-18 diagnostic, Kronos-small zero-shot did **not** meet the pre-registered necessary condition:
swing log-loss gain over the class prior -0.678 [-0.768, -0.581] (the LightGBM reference: +0.011), swing rank IC +0.005
[-0.028, +0.039]. It is therefore **not** in the forward test (D-044). The forecaster and freeze script are kept,
inactive. Details are in `docs/V3_RESEARCH.md`.

## Behaviour to know

Kronos normalises **price levels** over its window. A stock at the top of its 400-bar range is therefore forecast to
revert. On 2026-11-05, AAPL sat at 344, near its 400-day high of 350; the median 20-session path ended at 290.88 (-15%),
and 97% of paths finished below the sideways band.
* This is the model, not the integration: the vendored predictor with its README settings (T = 1.0, top-p = 0.9, 32
  samples averaged) gives the same 290.88.
* Expect probabilities far from calibrated, and a ranking that behaves partly like a reversal factor. E24 and the
  forward test measure both.

## Where it is used

* **Trade Desk, section 4d** (on demand). It shows:
  * a fan chart of 32 sampled paths (median, 10-90% band, a few raw paths);
  * the label-rule probabilities at 1D-20D;
  * the E24 and forward-test status.

  Historical as-of dates are all before July 2024 and carry a warning that they are in-sample for Kronos.
* **E24** (`scripts/experiments/e24_kronos_zero_shot.py`): the contaminated 2017-18 diagnostic, a necessary condition
  only.
* **Forward test** (`scripts/shadow/kronos_forecast.py`, version `kronos_zs`): only if E24's condition is met.
  * **Freeze:** `scripts/kronos/freeze_kronos.py`. The flat bundle has its four weight and config files at the top
    level, every one hashed by the frozen record.
  * **Inputs:** tonight's cached bars only.
  * **Promotion:** the forward criterion in `docs/V3_RESEARCH.md`.

## Limits

* Zero-shot only. Fine-tuning on 2003-2018 would still be contaminated and is not planned.
* Kronos-base (102M) is about 4x slower on this CPU and does not fit the nightly budget. A Colab GPU could run it for
  research.
* Kronos ignores news, filings, fundamentals and the market context: it sees one stock's candles.

# E31 pre-registration (2026-11-14): does refitting v1.1's exact recipe on recent data help live?

**Status: REGISTERED** on 2026-11-14 (IST, before 07:00), before the refit is run and before any forward outcome exists.
Decision D-062. The owner gave full authority and offered Colab. The refit takes about 11 minutes on this PC's CPU (the E08
run did), so Colab is not needed.

## Why

- **Every frozen stock model was fitted on data up to 2018.** v1.1 was refit on 2003-2018 and calibrated on 2019.
- **v1.1's gain over the class prior shrank every test year** (A-111): +0.020, +0.009, +0.0045 in 2020-22. Two
  explanations fit: the model's knowledge decays with age, or 2020 was an unusually easy regime.
- **Refitting on everything available before going live is standard practice.** No experiment has checked whether it
  helps here.

The historical years cannot answer this: 2020-22 would be in the refit. The forward test can, cleanly.

## The model: v1.2 = v1.1's recipe, three years later

Nothing is chosen. Every setting is copied from the v1.1 bundle and its source runs (E06, E07, E08):

| Part | Setting |
|---|---|
| Blend | 0.1 LightGBM + 0.6 gated fusion (seeds 7, 11, 23, 42, 101) + 0.3 convex logistic (lambda 1e-3), the recipe `blend_l0.1_f0.6_g0.3` |
| LightGBM | E07 parameters; rounds = E07's inner-stopped best iteration x (refit rows / train rows), as in E08 |
| Fusion | E06 settings (lr 3e-4, dropout 0.3, d_model 64, modality dropout 0.15, cosine schedule, batch 2048); epochs per seed = E06's best epoch + 1, as in E08 |
| Inputs | the same six modalities; the same preprocessor and TF-IDF/SVD (fitted on 2003-2015); frozen ImageNet ConvNeXt-Tiny chart embeddings |
| Refit rows | every labelled stock-date from 2003 with an as-of date on or before **2021-11-30**, so no 1-month label window reaches 2022 (E08's walk-forward convention) |
| Calibration | E08's `fit_best` per horizon (none, temperature or vector) on the stock-dates of **2022** (the frame ends 2022-12-01; labels that would need bars after the seal are missing) |
| Direction skill | **not validated at any horizon.** No out-of-sample period is left, so the D-057 gate stays closed and the desks plan no directional trade from v1.2 |

The sealed 2023+ period is not read. v1.2 is not added to the sealed scorer, whose files and hash stay unchanged.

## Forward test

- **Logging.** v1.2 is frozen with `scripts/shadow/freeze_model.py` (a new name; v1.1 is never touched). It is logged
  nightly from the same cached bars as v1.1, under `shadow/v1.2` with its own hash chain. It runs after the main push,
  so it cannot delay or endanger the other logs.
- **Endpoint.** For each decision session where both models logged on time and the outcome has matured: the mean, over
  matched stock-horizon rows (horizons 1D-1M), of log loss(v1.1) minus log loss(v1.2).
- **Test.** One-sided (H1: v1.2 is better); moving-block bootstrap over sessions with blocks of 20
  (`evaluation/forward_criteria.block_p_value`).
- **Decision.** After **120** such sessions. "Refit helps" if p < 0.05 and the mean gain is positive. This is a single
  test in its own family, so no Holm.
- **Consequence if it helps.** Future model generations are refit on all development data before freezing, and the
  owner may switch the desks from v1.1 to v1.2. **Consequence if not:** v1.1 stays the reference.

## Reported, deciding nothing

- Gain by horizon.
- Daily rank IC of p_up - p_down for both models.
- Calibration (reliability) of both.
- The class shares of 2022 against 2019 (the two calibration years).

## Trial count

1 forward test. 0 selection trials on 2017-2022: the recipe is copied, not chosen.

# E29 pre-registration (2026-11-13): recalibration and the text branch, judged forward only

**Status: REGISTERED** on 2026-11-13 (IST evening), before any forward outcome it is judged on exists. Decision D-058.
The owner gave full authority over E29 in this session.

**Why this design and not another.** Audits 9-11 (`docs/AUDIT_9_10_11_2026-11-13.md`) showed four things:
* direction skill does not replicate;
* the vision branch adds nothing once indicators are present;
* calibration fitted on 2019 does not transfer (A-85);
* the forward test has never seen the text branch, because live forecasts have no news.

The historical years cannot judge anything new honestly: 2017-18 has carried about 681 selection trials, and 2020-22
has been read many times. So E29 fits **no new model**. It registers two forward-only questions about the frozen v1.1,
with every rule fixed here.

## E29a: prior-shift recalibration of v1.1

* **Hypothesis.** Reweighting v1.1's probabilities toward the class shares recently realised in the forward test
  improves their log loss (A-85: class shares move between regimes).
* **Transform.** `src/nexus_alpha/calibration/label_shift.py`, constants fixed here:
  * `p'_k ∝ p_k · π_new,k / π_old,k`, renormalised;
  * the ratio is clipped to [0.25, 4];
  * π_new = realised shares of v1.1's own on-time forward rows of the same horizon whose exit session closed before
    the forecast's decision time, over the trailing 60 decision dates;
  * π_old = v1.1's mean predicted probabilities on those same rows;
  * no correction with fewer than 2,000 such rows.
* **Point in time.** Every input is either logged before the decision time or matured before it. The corrected
  probabilities can therefore be recomputed from the public, hash-chained logs at any later time, with no look-ahead.
* **Endpoint.** Per decision session, the mean over corrected rows of log loss(v1.1) minus log loss(E29a).
  * Test: one-sided moving-block bootstrap over sessions, blocks of 20 (`evaluation/forward_criteria.block_p_value`).
  * Scorer: `scripts/shadow/e29_score.py`.
* **Decision.** After **120** sessions with corrected rows. E29a is kept if its Holm-adjusted p-value (across E29a and
  E29b) is below 0.05 and the mean gain is positive. "Kept" means a candidate calibration for a future frozen model.
  The frozen v1.1 and its logs never change.

## E29b: does the text branch help live?

* **Hypothesis.** v1.1 with the company news archived before the decision time has lower log loss than v1.1 without
  news (what the forward test logs today), on the stocks that have news.
* **Inputs.**
  * Headlines from `scripts/shadow/news_snapshot.py`, archived nightly since 2026-11-13. Each row carries
    `fetched_utc`, the time this machine saw it.
  * A headline may be used only if `fetched_utc` is at or before the decision time, and it is available under the
    project's rule inside the 5-day window, at most 7 articles. This is the same path as the dashboard
    (`inference/pipeline._articles_frame`).
* **Forecasts.** A second v1.1 pass with news, for stocks with at least one usable headline. It is logged with its own
  hash chain and pushed before the decision time, exactly like the existing forecasters.
* **Start condition.** E29b's first scored session is the first one after both of these hold:
  * the news archive has run on at least 10 nightly runs without failing;
  * the news-on pass has been added to the nightly run and has logged one complete session on time.

  Until then E29b has no data.
* **Endpoint.** Per decision session, the mean over stocks with news of log loss(v1.1 without news) minus log loss(v1.1
  with news), at 1D-1M. Same bootstrap as E29a.
* **Decision.** After 120 scored sessions. Holm across E29a and E29b. "Kept" means the text branch is shown on the
  desks as validated forward. Otherwise it stays labelled as unvalidated context, as now.

## What E29 does not do

* **No new architecture, no ranking objective, no new seeds.** The forward test already logs three rankers (v2's
  `rank_score`, v3_fund, v3_alpha), and E21 found no dedicated ranker better than that baseline.
* **No use of 2017-2022 for any choice.** A historical replay of E29a on 2020-22 may be reported after this file is
  committed, as a diagnostic only. It cannot change a constant above.
* **No change to the frozen models, their logs, the risk engine or the sealed scorer.**

## Trial count

2 forward tests (E29a, E29b), Holm-corrected together. 0 selection trials on 2017-2022.

# Label engineering (D-005)

## Definition (frozen before any NEXUS model was fitted)

For a sample with as-of session t (decision 09:00 New York on t+1, entry at the open of t+1) and horizon h sessions:

```
R_h   = close[t+h] / open[t+1] - 1                      price only, split-adjusted
sigma = sqrt( EWMA_{alpha=0.06}( r_d^2 ) ) at t          daily close-to-close returns, lambda = 0.94, uses bars <= t
tau_h = max( 0.5 * sigma * sqrt(h), 0.0010 )
y_h   = UP if R_h > tau_h, DOWN if R_h < -tau_h, else SIDEWAYS        class index DOWN 0, SIDEWAYS 1, UP 2
z_h   = R_h / (sigma * sqrt(h))                          volatility-scaled return for the quantile heads
end_h = session t+h from the exchange calendar           label window [open t+1, close t+h]
```

Horizons: 1D = 1, 2D = 2, 5D = 5, 10D = 10, 20D = 20 and **1M = 21 sessions** (a documented operational month; it differs from 20D by one session, so the two heads are near-duplicates and are reported as such).

Why this rule: it is TriFrame's frozen 5-session label (`tf_data.labels`, threshold max(0.5σ√5, 10 bps)) generalised to all horizons, so it was chosen before seeing any NEXUS result and is directly comparable with earlier work. Scaling by the stock's own volatility keeps the class mix stable across calm and turbulent years (section 3), and the 10 bps floor equals the base round-trip cost, so a "move" smaller than costs is never called UP or DOWN. It was not tuned for balance.

## Verification

| Check | Result |
|---|---|
| Hand-computed returns, thresholds, classes and window ends on synthetic data | equal (`test_labels_match_hand_computation`) |
| σ is causal (changing later prices leaves σ at t unchanged) | yes (`test_sigma_is_causal`) |
| Windows that would need sealed bars | NaN, per horizon (`test_labels_are_nan_where_the_window_is_incomplete`) |
| Against TriFrame's frozen labels on all 236,520 s500 rows | σ identical (max difference 0.0); window ends identical (100%); 5-day return identical on 95.8% of rows, the rest differ only by dividends (TriFrame adds dividends with ex-dates t+2..t+5); 5-day class agreement **99.32%** (`reports/verification/label_parity.json`) |

## Class distributions (s500 frame)

| Horizon | train DOWN / SIDE / UP | test DOWN / SIDE / UP |
|---|---|---|
| 1D | 25.6 / 47.8 / 26.6 % | 25.5 / 48.7 / 25.9 % |
| 2D | 26.3 / 45.0 / 28.7 % | 26.6 / 45.4 / 28.0 % |
| 5D | 26.4 / 42.7 / 31.0 % | 26.6 / 43.6 / 29.9 % |
| 10D | 25.5 / 42.2 / 32.3 % | 26.9 / 42.5 / 30.6 % |
| 20D | 24.9 / 40.9 / 34.2 % | 26.9 / 41.9 / 31.2 % |
| 1M | 24.9 / 40.5 / 34.6 % | 27.0 / 41.3 / 31.8 % |

The upward tilt at long horizons reflects a rising market in a survivor universe. Imbalance is mild, so no resampling is used; the losses are unweighted cross-entropy and the base-rate (prior) model is always reported.

## Sensitivity (not used for training; `reports/labels/class_distributions.csv`)

| Rule | SIDEWAYS share train → test, 1D | 1M |
|---|---|---|
| registered: max(0.5σ√h, 10 bps) | 47.8 → 48.7 % | 40.5 → 41.3 % |
| fixed: 0.5%·√h | 29.5 → 24.0 % | 23.9 → 19.1 % |
| ATR-scaled: 0.25·ATR%·√h | 35.9 → 35.1 % | 29.9 → 29.6 % |
| cost only: 10 bps | 6.2 → 5.1 % | 1.1 → 1.0 % |

A fixed threshold drifts with the volatility regime (fewer SIDEWAYS labels in the turbulent 2020-2022 test years); the registered rule does not.

By volatility tercile (training cut points), the registered rule labels more SIDEWAYS outcomes in the most volatile third (47-51%) than in the calmest third (36-45%): trailing EWMA volatility over-estimates future volatility in turbulent states (volatility mean-reverts). This is the main source of the "will it move" predictability found in E01, and it is a volatility effect, not a directional one.

## Limitations

* Price-only labels slightly favour DOWN around ex-dividend dates (0.68% of 5-day labels differ from TriFrame's dividend-inclusive labels).
* The overnight move from close t to open t+1 is excluded by construction: forecasts concern what is still tradable at the decision time.
* Labels overlap across horizons and across a stock's consecutive samples (every ~10 sessions vs a 21-session window), so inference uses month blocks, never row counts.

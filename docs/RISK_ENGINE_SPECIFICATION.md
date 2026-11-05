# Risk and decision engine specification

Implemented in `nexus_alpha/trading/plan.py` (planner), `costs.py` and `backtest.py`; tested in `tests/test_trading.py`. Research and paper trading only: no broker endpoint exists anywhere in the code.

## Principles

1. Forecasting and trading are separate. A high UP probability is never by itself a LONG.
2. Hard gates come first and a model score cannot waive them (decision-engine spec 8.4, rule E150, X114).
3. Exact levels come only from structured OHLCV. Screenshot-only input yields NO_SIGNAL for numeric plans (spec 13.4).
4. A stop is never widened to fit a size (X010, X019); a missing level makes the plan incomplete, not invented.

## Gate order (first failing gate decides; all failing reasons are reported)

| # | Gate | Source rules | Failure |
|---|---|---|---|
| 1 | Data sufficiency: >= 252 daily bars, structured OHLCV for the as-of session | E001, E003, E006, E007 | NO_SIGNAL |
| 2 | Calibrated forecast for the horizon | E138 | NO_TRADE |
| 3 | **Directional skill**: validation up-vs-down AUROC 95% month-block CI lower bound > 0.5 (stored in the bundle) | D-015 | NO_TRADE |
| 4 | Edge: abs(P(UP) − P(DOWN)) >= min_edge (0.10; the backtest chooses it on validation) | | NO_TRADE |
| 5 | Short selling allowed (if SHORT) | E010 | NO_TRADE |
| 6 | Weekly permission compatible (BULL → long preferred; BEAR → short preferred; RANGE → edges; TRANSITION → with confirmation; HIGH_VOL / UNKNOWN → blocked) | E011-E018 | NO_TRADE |
| 7 | Daily setup on the same side with a structural invalidation and first obstacle | E023/E024, E026/E027, E031/E032, E066/E067 | NO_TRADE |
| 8 | Hourly trigger on the as-of session (structure break, engulfing, rejection) | E121/E122, E041-E044 | NO_TRADE (wait) |
| 9 | Stop = invalidation ∓ 0.25 ATR; stop on the losing side of the entry reference | X001, X011 | NO_TRADE |
| 10 | Target 1 = first opposing obstacle; net reward-to-risk after round-trip costs (and borrow for shorts) >= 1.5 | X021, E039, E149 | NO_TRADE |
| 11 | Size = floor(capital × risk% / (risk per share + cost per share)), capped by max position %; < 1 share → reject | X010 | NO_TRADE |

Plan fields: entry reference (last close; executed at the next open if within the max-chase price = 1 ATR beyond, else cancelled, E111/E139), stop, target 1, target 2 (2 × first-target distance), risk and reward per share, gross and net R:R, quantity, position value, planned loss in cash, holding sessions (= horizon, X071), invalidation text, cost assumptions, rule ids and the level source.

## User configuration

Account capital, risk per trade (default 0.5%), max position (10%), max portfolio gross (100%), max daily loss (2%, enforced by the portfolio simulator and documented for the live engine), short allowed, cost scenario. No personal capital is hard-coded.

## Costs and execution (D-010)

Round trip 10 / 15 / 20 bps (base / stressed / severe) of entry value; shorts pay 0.5%/yr borrow. Entry at the next open; cancel if the open is beyond the chase limit or the stop; from the second session an open beyond the stop exits at that open (gap-through, X083), an open beyond the target exits at the open; inside a bar the stop is assumed hit first when both levels are touched; otherwise exit at the close of the horizon's last session. There are no quote data, so spreads are not observed; the scenarios bound them.

## What the evidence says about the engine

On validation (2017-2018) the selected model's up-vs-down AUROC was 0.517-0.538 with intervals above 0.5, so gate 3 is **open at every horizon**. Most decisions are still NO TRADE (the edge, setup, trigger and reward-to-risk gates rarely all pass), but some LONG/SHORT plans are issued. The backtest of exactly this gated policy on 2020-2022 (E04) did not beat equal-weight buy-and-hold: Sharpe 0.20 at 5D (88 trades) and −0.64 at 20D (254 trades, expectancy −0.15R), intervals including zero, deflated Sharpe ≈ 0. A rule-only version lost money at both horizons. The gate was registered before these results and is **not** retuned on them (that would be selection on test data); instead the app attaches this evidence as a warning to every LONG/SHORT plan. A stricter, economically grounded gate (for example positive net expectancy on validation and calibration) is a candidate for a pre-registered change tested on fresh data.

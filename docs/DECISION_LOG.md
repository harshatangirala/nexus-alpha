# Decision log

Architecture and methodology decisions, with the alternatives considered. Conflicts between source documents are resolved here explicitly, never silently in code.

| ID | Date | Decision | Why | Alternatives considered |
|---|---|---|---|---|
| D-001 | 2026-11-05 | New repository at `C:\Users\harsh\nexus-alpha\`, local git, no remote until the owner approves | Separate from both source projects; outside `Downloads`, whose unrelated `CLAUDE.md` once leaked a wrong affiliation into a paper | `Downloads\TriFrame_Multimodal_Research` (master prompt's suggestion) |
| D-002 | 2026-11-05 | Seal from 2023-01-03, enforced at read time; NEXUS's own single-use owner unlock | Inherit TriFrame's boundary so the same years stay protected; TriFrame's unlock would write into its source tree | Filter after loading (rows would exist in memory and in code paths) |
| D-003 | 2026-11-05 | Decision 09:00 New York on t+1; bars through close t; entry at the open of t+1 | Conflict: the master prompt forecasts at a timestamp, the decision-engine spec triggers on hourly bars. Resolution: one daily decision before the open, with the hourly trigger evaluated on the as-of session's completed hourly bars. After-close filings and overnight news count, and the label starts at a price that was still executable | Decision at the close (loses after-hours information); intrabar hourly decisions (separate product per E141, needs intraday labels) |
| D-004 | 2026-11-05 | News availability: minute + 15 min; day-level and midnight-stamped rows at the next day 08:00 UTC; explicit ticker tags only | Conservative and testable; 2.92M midnight-stamped "minute" rows are day-level in disguise | The corpus `date_trading` anchor (next open; excludes day-level news dated t from a 09:00 decision on t+1 although it was public); company-name matching (forbidden by the spec: ambiguous) |
| D-005 | 2026-11-05 | Labels: TriFrame's vol-scaled 3-class rule for all six horizons; 1M = 21 sessions; price-only | Frozen before modelling; comparable with TriFrame; stable class mix across regimes | Fixed, ATR-scaled and cost-only thresholds (reported as sensitivity only) |
| D-006 | 2026-11-05 | Splits: train 2003-15 / inner 2016 / valid 2017-18 / calib 2019 / test 2020-22; 21-session purge + 5-session embargo; walk-forward 2008-22 for tabular models | Separate early-stopping, selection and calibration data; start 2003 because early weekly windows are padded (TriFrame review P4) | TriFrame's train 2001-16 / val 2017-19 / test 2020-22 (no calibration block) |
| D-007 | 2026-11-05 | SEC availability: the SEC study's decision time (acceptance + 35 min); quality C bounded by the filing date 18:05 | Reuse a verified point-in-time contract; never earlier than provable | Filing date only (loses intraday timing for 96% of events) |
| D-008 | 2026-11-05 | Reuse TriFrame's s500 chart embeddings instead of re-embedding | 236,520 sets × 3 views is infeasible on this CPU; byte-exact renderer parity and cosine ≥ 0.99998 embedding parity verified | Re-render and re-embed on Colab (planned for the native-resolution variant) |
| D-009 | 2026-11-05 | Fine-tuned ConvNeXt embeddings only evaluated from 2017 | Their weights saw TriFrame targets through 2014 and were early-stopped on 2015-16 | Exclude them (loses the strongest TriFrame backbone) |
| D-010 | 2026-11-05 | Execution: entry at next open; cancel on a gap beyond the chase limit or the stop; gap-through stops fill at the open; stop-first on ambiguous bars; time exit at the close of t+h; costs 10/15/20 bps round trip, borrow 0.5%/yr on shorts | TriFrame conventions plus the labelling spec's conservative rules | Exit at the next open after the time stop (TriFrame data decision #8); the label convention (close t+h) is used for consistency |
| D-011 | 2026-11-05 | Local text features are classical (TF-IDF→SVD, Loughran-McDonald tone); transformer encoders are registered GPU adapters | Transformer passes over ~650k articles are many CPU-hours; the SEC study found embeddings add nothing over classical text | FinBERT on headlines locally (est. 5-7 h) |
| D-012 | 2026-11-05 | Rule catalogue: all 270 rules catalogued with provenance; a tested subset executable; no rule is a hard-coded trading signal | The rulebooks call themselves hypotheses; 270 free rules would be a multiple-testing trap (rulebook safeguard 5) | Implement every rule |
| D-013 | 2026-11-05 | Weekly regime: one confirmed swing of each kind → TRANSITION, two of each needed for BULL/BEAR | Strict pivots confirm few lows in smooth trends; UNKNOWN there was too coarse; the rulebook forbids calling a trend on one swing | UNKNOWN (blocks everything in strong trends) |
| D-014 | 2026-11-05 | Calibration: none / temperature / vector scaling per horizon, chosen on chronological halves of the calib block; no isotonic | TriFrame's audit: isotonic worsened log loss (0.590 → 0.604) by producing exact 0/1 | Isotonic; Platt one-vs-rest |
| D-015 | 2026-11-05 | The trading engine requires validated directional skill per horizon (validation up-vs-down AUROC 95% CI above 0.5) before LONG or SHORT | A probability without demonstrated skill is not a trading signal; E01 found no direction edge | Trade on any probability edge |
| D-016 | 2026-11-05 | Market-context features kept only as an ablation group | They hurt in E01 (as in TriFrame's audit) | Always include |

## Conflicts between source documents

| Topic | Sources | Resolution |
|---|---|---|
| Decision timing | master prompt (forecast timestamp) vs decision-engine spec (hourly triggers) | D-003 |
| Time-stop exit price | TriFrame data decision #8 (observe at close, exit next open) vs playbook tables (at the close) | NEXUS uses the close of t+h, matching its labels (D-010) |
| Chart style | master prompt ("candlestick, indicator-free") vs TriFrame (binary OHLC bars) | Reuse OHLC bars (verified embeddings exist); a candlestick renderer is a registered, unrun variant |
| Partial bars | E141 (separate experimental family) vs completed-bar baseline | Completed bars only |
| "Exactly six indicator groups" (playbook) vs the 33-feature numeric branch (owner-approved) | TriFrame `CLAUDE.md` | Both used: NEXUS indicators (spec list) and TriFrame's 33, compared by ablation |

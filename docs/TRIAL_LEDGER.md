# Trial ledger

Every experiment's trial budget as its pre-registration states it, with the data each one read for selection. A
reader of the results needs this to judge how much a "keep" on a reused validation block can mean: the more
comparisons, the more one passes by chance. The deflated Sharpe ratios and PBO in the backtests (E04, E15) were
computed over the policy configurations those experiments tried. The model-selection trials below are the context
for every keep decision on 2017-18.

Sources: `docs/PROJECT_SPECIFICATION.md` and `reports/RESULTS.md` (v1), `docs/V2_ARCHITECTURE.md` section 4 (v2),
`docs/V3_RESEARCH.md` (v3). A "trial" is one (feature set or configuration) x model x horizon x endpoint evaluation on
the selection data. Placebo sets are counted, because they are evaluated the same way.

| Experiment | Question | Selection data | Pre-registered budget | Trials | Outcome |
|---|---|---|---|---|---|
| E07 | LightGBM hyperparameters | 2017-18 | TPE search | 25 | E07 parameters, used ever since |
| E04 | Trading policies (v1/v1.1) | thresholds on 2017-18 | policy grid | 20 configurations | no policy beats buy-and-hold; PBO 0.127 (v1), 0.119 (v1.1) |
| E11 | Market / macro blocks | 2017-18 | <= 10 feature-set x model fits per horizon, 9 horizons | <= 90 | not kept at stock level (D-033) |
| E12 | v2 fusion | 2017-18 | 1 configuration x 5 seeds | 5 | component of v2 |
| E13 | Sequence branch | 2017-18 | 2 architectures | 2 | component of v2 |
| E14 | v2 blend | 2017-18 | one blend search | 1 | v2 frozen |
| E16 | Text encoders | 2017-18 | 3 sets x 2 models per horizon, 9 horizons | 54 | see `docs/V2_RESULTS.md` |
| E19 | Market-direction model | 2017-18 | 6 sets x 2 models x 9 horizons (logistic: 5 penalties each) | 108 (plus penalty choice) | market_e19 kept and frozen |
| E20 | Regime rules | 2017-18 | 3 sets x 2 models x 9 horizons | 54 | not kept |
| E21 | Cross-sectional rankers | 2017-18 | 3 candidates x 5 horizons | 15 | none beat B0 |
| E22 | Fundamentals, universe ranks | 2017-18 | 3 sets x 2 models x 9 horizons x 2 endpoints | 108 | not kept |
| E23 | Fundamentals, industry-neutral | 2017-18 | 2 sets x 2 models x 9 horizons x 2 endpoints | 72 | kept, v3_fund frozen |
| E24 | Kronos zero-shot | 2017-18 (contaminated) | 1 configuration x 6 horizons x 2 endpoints | 12 | necessary condition not met |
| E25 | Qlib Alpha158 | 2017-18 | 4 sets x 2 models x 9 horizons x 2 endpoints | 144 | kept on rank IC, v3_alpha frozen (5D-1Y) |
| **Selection trials since v2 (E11-E25)** | | | | **<= 665** | |
| E26 | NEXUS vs implied volatility | forward only | 2 models x 2 horizons x 2 endpoints, Holm across endpoint 2 (D-051) | 8 | running from 2026-11-10 |
| E27 | LLM committee | forward only | descriptive | - | running, on demand |
| Forward criteria | v3_fund, v3_alpha vs ranker_v3 | forward only | 2 candidates x 2 conditions, Holm across candidates (D-051) | 2 | after 120 sessions |

**Reading.** The 2017-18 block has carried up to about 665 model-selection trials since v2, besides v1's selection
work. Keeps on it (market_e19, v3_fund, v3_alpha) are hypotheses for the forward test, not findings. The forward
test (E26, E27 and the forward criteria) selects nothing, and its multiple comparisons are Holm-corrected (D-051).
The sealed 2023+ period has never been read.

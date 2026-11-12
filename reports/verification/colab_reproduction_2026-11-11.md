# Colab reproduction, 2026-11-11 (owner's run)

Copied from `COLAB_RESULTS.md`, which `notebooks/NEXUS_Alpha_Colab_RunAll.ipynb` wrote to the owner's Google Drive
(`antigravity_workspace/colab_results/`). It was read on 2026-11-12 and recorded here because the Drive folder is not
part of the repository.

Run: 2026-11-11T21:53:30+00:00 | commit bfca98f | verdict: **REPRODUCED**

## Machine
Python 3.13.18, 8 CPUs, RAM 54.8 GB, GPU: none.
Packages: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, pydantic 2.14.3,
pytest 9.2.0, torch 2.14.1+cpu.

## Steps
| step | result | seconds |
|---|---|---|
| copy Drive to local disk | ok | 573.3 |
| layout check | ok | 0.1 |
| tests | ok | 10.8 |
| ratio build | ok | 87.8 |
| audit (7 hard checks) | ok | 107.6 |
| E22 | ok | 676.4 |
| E23 | ok | 665.1 |
| Alpha158 build | ok | 85.1 |
| E25 | ok | 2490.8 |
| compare with committed | ok | 0.1 |
| frozen v3_fund verify | ok | 0.9 |

Tests: 27 passed in 6.59 s.

## Reproduction against the committed runs
| experiment | committed run | re-run on Colab | keep decisions identical | max log-loss diff | max mean daily IC diff | verdict |
|---|---|---|---|---|---|---|
| E22 fundamentals | `20261107T001413_e22_fundamentals_a9900a` | `20261111T220725_e22_fundamentals_a9900a` | yes | 0.00000 | 0.00006 | REPRODUCED |
| E23 industry fundamentals | `20261107T002423_e23_industry_fundamentals_2d062c` | `20261111T221842_e23_industry_fundamentals_2d062c` | yes | 0.00000 | 0.00007 | REPRODUCED |
| E25 Alpha158 | `20261110T144028_e25_alpha158_951bbf` | `20261111T223024_e25_alpha158_951bbf` | yes | 0.00001 | 0.00009 | REPRODUCED |

Frozen v3_fund record verified: `20261107T003327_v3_fund_refit_053064`.

A re-run is a reproduction only: nothing here replaces a committed record or touches the sealed period.

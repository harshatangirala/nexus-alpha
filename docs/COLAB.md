# Running NEXUS Alpha on Google Colab

Colab is optional: everything in this repository ran on the owner's CPU. It is useful for re-running the heavy stages away
from that PC, or for the GPU-only extensions registered in the notebook (section 13).

## What can and cannot run on Colab

| Part | On Colab? | Why |
|---|---|---|
| Fundamentals build, audit, E22/E23 reproduction (notebook sections 14-15) | **Yes, CPU** | Pure pandas/LightGBM; needs the 1.8 GB staged data set |
| v1 pipeline (embeddings, samples, fusion, walk-forward, RAG: sections 1-12) | Yes, with a much larger data set (about 15 GB: TriFrame bars and caches, SEC study outputs, embeddings) | Needs the private TriFrame store and an HF token |
| Sealed 2023+ evaluation | Not planned | D-035: CPU, 3-5 hours, and the sealed bars stay on one machine |
| Nightly forward test (`run_daily.cmd`) | **No** | Windows Task Scheduler, one machine, hash-chained logs pushed from it; Colab sessions stop when idle |

## Steps for the fundamentals path

1. On the owner's PC: `python scripts/colab/stage_for_drive.py --go` copies the minimal data set (about 1.4 GB, 6,400 files)
   into `G:\My Drive\antigravity_workspace\` (read-only on the source projects; no secrets, no `.env`, no sealed data).
   Wait until Google Drive for desktop shows it as synced.
2. In Colab, open `notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb` (from GitHub or Drive). Run sections 1-4. Section 2
   clones the public repository to Drive if it is not there. Optional Colab Secret: `SEC_USER_AGENT` (only to refresh SEC
   facts).
3. Run sections 14-15 in order. Each step prints what it did; the layout check lists anything missing.
4. `compare_with_committed.py` must print `REPRODUCED` for E22 and E23 (same keep decisions, validation log losses within
   0.002). A re-run is a reproduction only: never commit it and never replace the committed record.

## Time (from this PC's timings; free Colab has 2 CPU cores, Pro+ high-RAM more)

| Step | This PC (12 cores) | Free Colab (2 cores), estimate |
|---|---|---|
| Staging copy and Drive upload | 10-60 min, depending on the upload speed | same |
| Sync to local disk, installs, tests | - | 10-15 min |
| Ratio build (snapshots reused) | 1.5 min | about 8 min |
| Audit (7 hard checks) | 3 min | about 5 min |
| E22 + E23 | 8 + 8 min | about 30-45 min each |
| Snapshot rebuild (optional) | 55 min | several hours |

The numbers for Colab are estimates, not measurements: this session had no access to Colab.

## Rehearsal against the staged Drive copy (2026-11-07, on the owner's PC)

The notebook's steps were run with `NEXUS_ARTIFACTS`, `NEXUS_OHLCV_ROOT` and `NEXUS_SEC_FULL500_RUN` pointing at the staged
copy in `G:\My Drive\antigravity_workspace` (exactly what Colab sees under `/content/drive/MyDrive/antigravity_workspace`):

| Step | Result |
|---|---|
| Layout check | 16 of 16 present |
| Fundamentals tests and runner tests | pass (`test_seal_gate.py` needs hourly bars and the news index, outside the minimal set) |
| Ratio build from the staged snapshots | panel bit-identical to the original (2,758,286 x 112; max difference 0.0; same missing pattern) |
| Audit | 7 of 7 hard checks pass |
| E22 and E23 re-run + `compare_with_committed.py` | **REPRODUCED**: identical keep decisions; log-loss and rank-IC differences 0.00000 |

Still untested on Colab itself (this session had no Colab access): the Linux runtime, `pip install -r requirements-colab.txt`
against Colab's preinstalled packages, and `rsync` on the Drive mount. The reproduction cell catches any numerical drift.

## Not done on Colab, on purpose

* `v3_fund` is never refitted there: its bundle is frozen (`registry/frozen_models/v3_fund.json`) and the notebook only
  verifies it file by file.
* The forward test, the sealed evaluation and anything that writes to `shadow/`.

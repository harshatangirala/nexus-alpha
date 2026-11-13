# Running NEXUS Alpha on Google Colab

Colab is optional: everything in this repository ran on the owner's CPU. It is useful for re-running the heavy stages away
from that PC, or for the GPU-only extensions registered in the notebook (section 13).

There are two notebooks:
* **`NEXUS_Alpha_End_to_End.ipynb`**: the presentation copy, with the code of every step inline (below).
* **`NEXUS_Alpha_Colab_RunAll.ipynb`**: the reproduction run, which checks the committed E22/E23/E25 numbers.

## The presentation notebook: `notebooks/NEXUS_Alpha_End_to_End.ipynb` (audit 6)

1. On the PC (once): `python -I scripts/colab/stage_notebook_inputs.py`. It writes about 1 GB to
   `MyDrive/antigravity_workspace/nexus_artifacts/notebook/`:
   * the sample frame;
   * the chart-embedding components, fitted on training rows;
   * the news, SEC and rule features;
   * the language inputs of sections 8A-8C (E28): the cleaned text and ChronoBERT vectors of 68,853 SEC filings, the
     SEC price-impact expert's table (labels whose exit is before 2019 only), and the FinBERT, ChronoBERT and expert
     blocks per sample;
   * the saved predictions of the documented runs;
   * the knowledge base.
   * for notebook v5 (section 13B, E30), also run `python -I scripts/colab/stage_e30_inputs.py` once: it writes
     `fundamentals/e30_stock_months.parquet` (23 MB, built from the sealed development stores) next to the other
     inputs and copies it to Drive. Without it section 13B stops with a clear message; every other section is
     unaffected.

   It is private (book text). Wait until Google Drive for desktop has synced it.
2. In Colab: File > Open notebook > GitHub, then enter `harshatangirala/nexus-alpha` and pick the notebook.
   Runtime > Change runtime type: A100 GPU (or L4), High-RAM. Then Runtime > Run all and allow the Drive mount.
3. Run time:
   * **A100:** about 50-65 minutes (41 minutes before audits 9-11 added five seeds to 13A).
   * **L4:** works unchanged, roughly 1.5-2 times slower. bfloat16 is supported, so section 8A's ChronoBERT
     re-encoding of all 68,853 filings takes minutes.
   * **T4:** works; it uses float16 instead of bfloat16.
   * **CPU only:** runs, but sections 8, 8A, 8C and 10 shrink. 8A re-encodes a random 400 filings to check parity, and
     10 fine-tunes on a few thousand charts. Expect a few hours.

   The first code cell has `QUICK = False`; `True` is a short smoke test.

It trains the logistic regression (with its global-minimum certificate), LightGBM, the fusion network and a fine-tuned
ConvNeXt itself, with their loss curves, then evaluates every model, including the frozen models' saved predictions.
Daily bars are read with a date filter, so nothing from the sealed period is loaded. The same cells run on this PC
with `python -I scripts/colab/run_notebook_locally.py --quick`, which saves every figure for inspection.

## One pass: `notebooks/NEXUS_Alpha_Colab_RunAll.ipynb` (reproduction)

1. On the PC: `python scripts/colab/stage_for_drive.py --go`, then wait until Google Drive for desktop shows it as synced.
2. In Colab: open the run-all notebook (from GitHub), **Runtime -> Run all**. A CPU runtime is enough; no secret is needed.
3. At the end it prints, and saves to `MyDrive/antigravity_workspace/colab_results/<time>/COLAB_RESULTS.md`, one small
   summary: versions, hardware, timings, test counts, the audit result, the E22/E23/E25 comparison (**REPRODUCED** or
   **MISMATCH**) and the frozen `v3_fund` check. It contains no data, no secrets and no email address.

Everything runs through `scripts/colab/reproduce.py`, so the same command works on any machine:
`python scripts/colab/reproduce.py --drive-root <staged folder> [--work <local copy>] [--skip-experiments]`.

Rehearsed on 2026-11-07 on the owner's PC: a fresh clone, a clean environment built from `requirements-colab-repro.txt`
plus torch (preinstalled on Colab; the script never reinstalls it and stops with a clear message if it is missing), the
staged Drive copy as the data. Verdict **REPRODUCED** in about 21 minutes on 12 cores: copy 95 s, tests 19 passed, ratio
build 93 s, audit 7 of 7, E22 467 s and E23 472 s with identical keep decisions and differences of 0.00000, frozen
`v3_fund` verified. The rehearsal caught two hidden torch dependencies (one test, and E22/E23 through the v1.1 bundle
loader). Not yet run on Colab itself; on a 2-core runtime expect about 1-2 hours. The orchestrator notebook below is the full v1 pipeline (about 15 GB of private data) and is not meant for Run
all.

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

## E25 added to the one pass (2026-11-10)

After E22 and E23, `reproduce.py` rebuilds the Alpha158 features from the staged daily bars
(`scripts/data/build_alpha158.py`), re-runs E25 and compares it with the committed record. The verdict is
**REPRODUCED** only if all three match. E25 adds about 1-3 hours on a small runtime; set `WITH_E25 = False` in the
notebook to skip it.
* **Rehearsal status.** The E25 step was not rehearsed through a Drive copy: the PC's Google Drive mount was not
  available that evening. Its comparison logic was checked against a copy of the committed run. The inputs it needs
  (daily bars, the v2 frame, the v1.1 bundle) are the ones the E22/E23 rehearsal already used.

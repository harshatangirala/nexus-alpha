# Training methodology

## Order of work (master spec 17)

1. Data contracts and leakage tests pass (`tests/`), then
2. branch baselines: class prior, momentum logistic, LightGBM on indicators (E01); chart embeddings with linear heads and with trees (E02);
3. fusion variants and ablations (E03), calibration, registered hypotheses, inference bundle;
4. walk-forward re-evaluation of the tabular model across 2008-2022 (E05);
5. economic backtest of pre-specified policies (E04);
6. retrieval evaluation (RAG);
7. audit phase: fusion optimisation (E06), LightGBM tuning (E07), convexity certificate for the logistic head (E09), final assembly with refit (E08), controls (E10), backtest v1.1 on the fixed engine (E04 v1.1).

## Fitting rules

* Everything learned from data (scalers, imputation medians, PCA, TF-IDF vocabularies and SVD axes, tree models, fusion weights) is fitted on the `train` block (2003-2015) only.
* Early stopping uses `inner` (2016), never the selection block.
* Model and configuration selection uses `valid` (2017-2018).
* Calibration and decision thresholds use `calib` (2019).
* `test` (2020-2022) is scored after the configuration is frozen; it is a historical diagnostic because TriFrame already read those years.

## Fine-tuning policy (master spec 10.1)

| Branch | Decision | Reason |
|---|---|---|
| Chart backbones | frozen ImageNet features, plus TriFrame's already chart-fine-tuned ConvNeXt-T as a comparison | TriFrame measured full fine-tuning (+0.032 validation R² for ConvNeXt on volatility) and partial unfreezing (≈ 0); re-fine-tuning on six-horizon direction labels needs a GPU and starts from a branch with no direction signal (E02) |
| Text encoders | not fine-tuned; classical features locally | CPU budget; prior evidence that embeddings add nothing over classical text |
| Fusion heads | trained from scratch on cached features | small, fast, resumable |
| LightGBM | v1: fixed settings, early-stopped. v1.1: 25-trial TPE search on validation (E07) | the search is counted in the trial log; the tuned settings did not beat the defaults on the test years |

LoRA and discriminative learning rates are available through the GPU profile but were not run. The fusion networks got a small, logged sweep (E06), not an open-ended search: each extra configuration is another look at validation.

## Optimisation (audit phase)

| Step | What was done | Result |
|---|---|---|
| Learning rate | 3e-4, 1e-3, 3e-3 with linear warm-up (1 epoch) and cosine decay to 5% | 3e-4 best on validation (1.05883); 3e-3 clearly worse (1.06360) |
| Early stopping | patience 6 (v1: 3), best epoch restored; the curve records train CE in eval mode, 2016 CE and validation CE every epoch | most v1 models had stopped after 5-9 epochs; curves in the audit report |
| Capacity and regularisation | width 128, weight decay 1e-3, dropout 0.3 | dropout 0.3 best (1.05867); weight decay had no effect |
| Seeds | 5 seeds (7, 11, 23, 42, 101) | validation 1.05908 ± 0.00100; the seed ensemble 1.05804 |
| Loss landscape | 1-D path between two seeds; 2-D filter-normalised slice around the selected model | trained point within 0.0003 of the slice minimum (smaller than the seed spread); the seed-to-seed path crosses a barrier of +0.009, so seeds land in different basins of similar height |
| LightGBM | TPE over learning rate, leaves, leaf size, feature and bagging fractions, L2; tree count by early stopping on 2016 only (A-21); per-iteration curves from a separate fixed-rounds fit | validation 1.05625 vs 1.05699 default (1D/5D/20D); no gain on the test years |
| Convex head | multinomial logistic, reference-class parameterisation, L2 1e-3, L-BFGS from 5 starts | certified unique global minimum (E09) |
| Assembly | simplex blend of the three families (step 0.1), calibration choice on 2019 halves, refit on 2003-2018 with fixed epochs and trees (`restore_best=False`) | E08; deployed bundle |

"Global minimum" applies only to the convex head. Gradient-boosted trees are greedy. The networks are non-convex, so what can be shown is convergence, stability across seeds and a smooth basin. See `docs/CHIEF_AUDIT_REPORT.md` section 4.

## Hardware and reproducibility

* `training/hardware.py` logs the device, GPU, VRAM, CUDA, torch version and free disk, and chooses bf16 / fp16 / fp32.
* Seeds fixed (7; E06/E08 use 7, 11, 23, 42, 101); deterministic algorithms on CPU; LightGBM `deterministic=True`, `force_row_wise=True`.
* Fusion checkpoints every epoch (`last.pt` with optimiser state; `best.pt`); a rerun resumes.
* Every run writes a manifest (config, config hash, git commit, dirty flag, the list of dirty files and a SHA-256 of the working-tree diff, input digests, environment) and an experiment-log line, including failed runs.

## Compute actually used

Laptop CPU (Ryzen 5 5500U, 6 cores, 15 GB RAM, no NVIDIA GPU). No GPU time was spent. Full-universe re-embedding and transformer text encoders are deferred to Colab and need the owner's go-ahead with the cost estimate printed by `scripts/run_pipeline.py`.

# Training methodology

## Order of work (master spec 17)

1. Data contracts and leakage tests pass (`tests/`), then
2. branch baselines: class prior, momentum logistic, LightGBM on indicators (E01); chart embeddings with linear heads and with trees (E02);
3. fusion variants and ablations (E03), calibration, registered hypotheses, inference bundle;
4. walk-forward re-evaluation of the tabular model across 2008-2022 (E05);
5. economic backtest of pre-specified policies (E04);
6. retrieval evaluation (RAG).

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
| LightGBM | fixed settings, early-stopped | small search keeps the multiple-testing count honest |

LoRA and discriminative learning rates are available through the GPU profile but were not run; no hyperparameter search (Optuna) was run on the fusion models, by choice: one configuration per variant keeps the number of looks small.

## Hardware and reproducibility

* `training/hardware.py` logs the device, GPU, VRAM, CUDA, torch version and free disk, and chooses bf16 / fp16 / fp32.
* Seeds fixed (7); deterministic algorithms on CPU; LightGBM `deterministic=True`, `force_row_wise=True`.
* Fusion checkpoints every epoch (`last.pt` with optimiser state; `best.pt`); a rerun resumes.
* Every run writes a manifest (config, config hash, git commit and dirty flag, input digests, environment) and an experiment-log line, including failed runs.

## Compute actually used

Laptop CPU (Ryzen 5 5500U, 6 cores, 15 GB RAM, no NVIDIA GPU). No GPU time was spent. Full-universe re-embedding and transformer text encoders are deferred to Colab and need the owner's go-ahead with the cost estimate printed by `scripts/run_pipeline.py`.

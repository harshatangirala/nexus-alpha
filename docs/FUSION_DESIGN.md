# Fusion design

Implemented in `nexus_alpha/multimodal/` (`fusion.py`, `dataset.py`, `trainer.py`); tested in `tests/test_fusion_and_calibration.py`.

## Inputs

| Modality | Dimension | Presence mask |
|---|---|---|
| `img_W`, `img_D`, `img_H` | 64 each (PCA of 768-d ConvNeXt-T embeddings, per view) | 1 if the view has >= 40% of its bars (data contract), else 0 |
| `num` | 80 (47 NEXUS multi-timeframe indicators + 33 TriFrame features) | always 1 (NaN warm-up values → 0 after standardisation) |
| `news` | 75 (64 SVD + tone + counts, recency, match specificity, precision) | 1 if at least one article was available in the 5-day window |
| `sec` | 91 (17 structured + 2 × (32 SVD + 4 tone + present flag)) | 1 if the sample is inside SEC coverage |
| `rules` | 21 one-hot rule features (regime, setups, candles, triggers) | always 1; used only in ablation 17 |

All scalers and PCA are fitted on training rows; masked blocks are zero-filled and cannot influence the output (tested for every variant by feeding garbage into a masked block).

## Variants

* **late**: one head per modality; fused logits are a softmax-weighted average over present modalities (learned weights). The simplest prediction-level fusion.
* **concat**: projected vectors (64-d each, Linear-GELU-Linear-LayerNorm) plus mask bits → MLP (128) → shared 64-d representation.
* **attention**: modality tokens + learned modality/timeframe type embeddings + a CLS token → 2-layer pre-norm Transformer encoder (4 heads), padding mask for missing modalities; the CLS output feeds the heads. This is the cross-modal attention model: every modality token attends to the others.
* **gated**: softmax gate over present modalities (tanh scoring network) weights their projections; gates are reported per sample as diagnostics.
* **hybrid**: attention-contextualised tokens pooled by the gating network.

Heads: six horizon classifiers (3 logits each, no forced agreement across horizons) and quantile heads (0.1, 0.5, 0.9) of the volatility-scaled return z_h. Loss: masked cross-entropy averaged over available labels + 0.2 × pinball loss. Modality dropout 0.15 during training (never all modalities). AdamW (lr 1e-3, wd 1e-4), batch 2048, gradient clipping 1.0, early stopping on the 2016 inner block (patience 3, at most 20 epochs), checkpoints every epoch for resumption, seed 7.

## Compute strategy

The chart backbones are not fine-tuned inside the fusion run: their embeddings are cached (TriFrame s500, verified), so fusion trains on 236k × ~450 inputs on a laptop CPU in minutes per model. "All models included" does not mean all loaded at once: branches are cached, the fusion head is trained on top, and end-to-end fine-tuning is a GPU-profile option.

## Selection and tests

Variant chosen by mean validation log loss over the six horizons (ties within 0.0005 → simpler). Ablations, missing-modality stress tests and the registered hypotheses are in `docs/PROJECT_SPECIFICATION.md` and the E03 results (`docs/EXPERIMENT_LOG.md`, `reports/`).

## Not done (registered)

Horizon embeddings with a joint head, return-regression point heads beyond the quantiles, end-to-end fine-tuning of the chart backbone through the fusion loss, shared-vs-separate backbones per timeframe, Optuna search of fusion hyperparameters (one fixed configuration was used to keep the search count small and honest).

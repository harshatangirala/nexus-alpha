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
| `news_tx` (E28) | 20 (FinBERT tone mean/min/max, negative share, 16 PCA components of the headline embeddings) | 1 if at least one headline was available |
| `sec_tx` (E28) | 66 (ChronoBERT-2010 vectors of the latest 8-K and 10-Q/10-K, 32 PCA components each, plus two flags) | 1 if either filing exists |
| `sec_impact` (E28) | 9 (the SEC price-impact expert's P(move) and P(up) - P(down) at 1 and 5 sessions for the latest filing, its age and type; the 30-day maximum and count) | 1 if a filing became public within 14 days |

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
## The language branch (E28, D-053)

The three `*_tx`/`sec_impact` blocks turn the fusion into one vision + language network. They are built by
`scripts/data/build_sec_tx_features.py` and `scripts/data/build_sec_impact_features.py` (FinBERT features from E16).
`multimodal.dataset.build(..., modalities)` assembles them like every other block: standardised on training rows where
present, zero-filled and flagged where absent. `load_tables(nlp=True)` joins them.

* **Encoder choice.** ChronoBERT with a 2010-12-31 cutoff (`nexus_alpha.nlp.transformer_text`) is chronologically
  consistent: it never read text from the inner, validation or test years. A filing encoder pretrained on later
  filings (SEC-BERT, to 2019) would carry look-ahead.
* **SEC study labels.** The price-impact expert (`nexus_alpha.nlp.sec_impact`) learns the SEC study's `T_trade`
  labels from its event features. Events before 2016 are predicted out of fold, so a NEXUS training row never sees a
  prediction fitted on its own event.
* **Test.** Pre-registered keep test against the six-block network and a within-date placebo: `docs/V3_RESEARCH.md`
  (E28).

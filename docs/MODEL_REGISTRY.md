# Model registry

Status vocabulary (master spec 6.4): **included-finetuned**, **included-adapter** (used through a tested adapter), **evaluated-not-retained**, **excluded** (with reason), **registered** (adapter or entry exists, not run in this profile).

## Chart vision (contract verified: renderer byte-exact, embeddings cosine >= 0.99998; `reports/verification/cv_parity.json`)

| Backbone | Source / checkpoint | Embedding | Status | Evidence |
|---|---|---|---|---|
| ConvNeXt-Tiny (ImageNet-12k → 1k), frozen | timm `convnext_tiny.in12k_ft_in1k` | 768 | **included-adapter**: the image branch of every E03 fusion model | TriFrame s500 winner; E02 |
| ConvNeXt-Tiny fine-tuned on TriFrame charts | private `triframe-cv-model-s500/ft_convnext_tiny.safetensors` | 768 | **included-adapter**, evaluated from 2017 only (weights saw TriFrame targets through 2014, early-stopped on 2015-16) | E02, E03 `b01` |
| DINOv2 ViT-S/14, frozen | timm `vit_small_patch14_dinov2.lvd142m` | 384 | **evaluated-not-retained** for fusion (best frozen backbone images-only on validation in E02, by a small margin over ConvNeXt) | E02 |
| ResNet-18, frozen | timm `resnet18.a1_in1k` | 512 | evaluated-not-retained | E02 |
| ResNet-18 random weights (seed 0) | timm `resnet18`, no pretraining | 512 | control | E02 |
| EfficientNet-B0, ViT-B/16, Swin-T, MobileNetV3 (frozen and fine-tuned) | TriFrame runs repo | various | **registered**: embeddings exist on Hugging Face but were not fetched (bandwidth and disk); TriFrame found them no better | TriFrame s500 |
| ConvNeXt-Tiny "p25/p50" partial fine-tunes, risk-only fine-tune, Swin seeds | TriFrame runs repo | | registered | TriFrame study 5F |

Preprocessing for all: binary r1 chart (128 × 3·bars) repeated to 3 channels, bilinear antialiased resize to 224², backbone mean/std, timm `num_classes=0` pooled features. Per-view PCA to 64 components fitted on training rows. Timeframe identity is explicit: each view is its own modality token with a learned type embedding; the backbone is shared across views (TriFrame's design), and shared-versus-separate backbones was not re-tested.

## Text

| Model | Status | Note |
|---|---|---|
| TF-IDF (1-2 grams, 50k terms) → SVD-64 on news headline + lead | **included** (local profile) | fitted on articles available before 2016 |
| TF-IDF → SVD-32 on SEC event text (first 3,000 characters) | **included** | fitted on events available before 2016 |
| Loughran-McDonald 1993-2025 tone | **included** | read in place; 2025 vintage is a mild look-ahead for early samples |
| ProsusAI/finbert, yiyanghkust/finbert-tone, BAAI/bge-base-en-v1.5, nlpaueb/sec-bert-base, manelalab/chrono-bert-v1-20101231 | **registered** (cached locally; GPU profile) | not run: ~650k articles would take many CPU-hours; the SEC study found transformer embeddings add nothing over classical text |
| BAAI/bge-small-en-v1.5 | **included** for RAG retrieval only | |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | **included** for RAG reranking only | |

## Tabular and fusion heads

| Model | Status | Evidence |
|---|---|---|
| Class prior, momentum logistic | baselines | E01 |
| LightGBM on NEXUS indicators / TriFrame-33 / both / + market | E01; **both** selected on validation; market context hurt (D-016) | E01, E05 walk-forward |
| Logistic on chart PCA features (per view and tri-frame) | E02 | E02 |
| LightGBM on numbers + chart PCA | E02 | E02 |
| Fusion: late, concat, attention, gated, hybrid | E03 (selection by validation) | `docs/EXPERIMENT_LOG.md` |
| Selected fusion model + per-horizon calibrators | v1 inference bundle `artifacts/bundles/20261106T000935_e03_fusion_8ddffb` (superseded, kept) | E03 |
| Fusion, gated, E06 settings (lr 3e-4, cosine, dropout 0.3), 5 seeds | **included** in the v1.1 blend (weight 0.6) | E06, E08 |
| Multinomial logistic on numeric features (L2 1e-3) | **included** in the v1.1 blend (weight 0.3); certified global minimum | E09, E08 |
| LightGBM per horizon, E07 tuned parameters | **included** in the v1.1 blend (weight 0.1) | E07, E08 |
| **Deployed: blend v1.1, refit 2003-2018, temperature at 10D/20D/1M** | inference bundle `artifacts/bundles/20261106T020411_e08_final_model_5c4327` (first build `20261106T013154_e08_final_model_8bbb54` superseded by the A-21 rebuild) | E08; `docs/MODEL_CARD.md` |

Checkpoints and bundles are never committed (data licences). Provenance: `model_registry/checkpoint_inventory.json`, `model_registry/triframe_backbones_s500.json`, `artifacts/cv_embeddings/s500/manifest.json` (source revision and sha256 of every array).

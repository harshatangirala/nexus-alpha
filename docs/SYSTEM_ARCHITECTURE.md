# System architecture

```
             SEALED POINT-IN-TIME SAMPLE  (as-of session t, decision 09:00 New York on t+1)
                                   |
   +-----------+-----------+-------+--------+-----------------+------------------+
   |           |           |                |                 |                  |
 WEEKLY      DAILY      HOURLY        INDICATORS            NEWS              SEC FILINGS
 chart r1    chart r1   chart r1   1D/1W/1H + TriFrame-33   <=7 articles,      latest 8-K, 10-Q/10-K,
 (104 bars)  (126)      (140)       (completed bars only)   5-day window,      structured filing
   |           |           |                |                dedup/clusters     features
   +-----------+-----------+                |                     |                  |
   CV REGISTRY: ConvNeXt-T frozen (fusion),  |               NLP REGISTRY (CPU profile: TF-IDF->SVD + Loughran-
   DINOv2-S, ResNet-18, random control,      |               McDonald tone; GPU adapters: FinBERT, BGE, SEC-BERT,
   fine-tuned ConvNeXt-T (eval from 2017);   |               ChronoBERT)
   per-view PCA(64) fitted on train          |                     |                  |
   +-----------------------------------------+---------------------+------------------+
                                   |
            MODALITY PROJECTIONS + modality/timeframe identity embeddings + presence MASKS
                                   |
      FUSION VARIANTS: late | concat | cross-modal attention | gated | hybrid (attention + gating)
                                   |
             SIX HORIZON HEADS (1D 2D 5D 10D 20D 1M) x (DOWN, SIDEWAYS, UP)  +  quantile heads on z_h
                                   |
          CALIBRATION (none / temperature / vector, per horizon, fitted on 2019 only)
                                   |
   EVIDENCE: indicator values | rule events (weekly regime, daily setups, hourly triggers) | news & filings with
   timestamps | leave-one-modality-out and chart-occlusion DIAGNOSTICS | RAG citations (books p./rulebook line)
                                   |
   SEPARATE DECISION ENGINE (deterministic): data gates -> directional-skill gate -> edge -> weekly permission ->
   daily setup -> hourly trigger -> structural stop (+0.25 ATR) -> first-obstacle target -> net R:R after costs ->
   risk-based size -> LONG / SHORT / NO TRADE / NO SIGNAL
                                   |
                        STREAMLIT APP (paper / research only)
```

## Module map (`src/nexus_alpha/`)

| Module | Responsibility |
|---|---|
| `seal.py` | sealed period, dev masks, owner unlock with hash-chained log |
| `data/calendar.py`, `data/bars.py` | XNYS sessions (verified, extended), sealed readers, causal multi-timeframe windows |
| `data/news.py`, `data/news_select.py` | ticker mapping, availability policy, dedup and clustering, per-sample selection |
| `data/sec.py` | SEC event availability and per-sample filing features |
| `indicators/core.py`, `indicators/market.py`, `indicators/triframe_visible.py` | causal indicators (TriFrame conventions), market context, vendored 33 features |
| `targets.py`, `splits.py` | six-horizon labels, purged/embargoed splits, walk-forward folds |
| `cv/render.py`, `cv/backbones.py`, `cv/store.py`, `cv/features.py` | vendored renderer r1, backbone registry and embedder, stored-embedding access with the fine-tuning rule, train-fitted PCA |
| `nlp/text_features.py`, `nlp/aggregate.py` | lexicon tone, TF-IDF->SVD, aggregation shared by training and inference |
| `nlp/transformer_text.py` | ChronoBERT-2010 (pinned revision): filing cleaning (safe harbour dropped), resumable length-sorted encoding, 256 tokens, masked mean pooling (E28 `sec_tx`) |
| `nlp/sec_impact.py` | the SEC price-impact expert: LightGBM on the SEC study's event features and `T_trade` labels (exits before 2019 only, numeric non-time features only), out-of-fold before 2016, point-in-time per-sample evidence (E28 `sec_impact`) |
| `multimodal/dataset.py` | the fusion's blocks, incl. the E28 language blocks (`NLP_BLOCKS`), presence masks and the input type guard (`check_feature_types`) |
| `language_view.py` | data layer of dashboard tab O: the pinned E28 run, the expert per filing, point-in-time similar filings and ChronoBERT analogues (centred vectors) |
| `rules/catalog.py`, `rules/detectors.py`, `rules/batch.py` | 270-rule catalogue, executable causal detectors, batch evaluation |
| `multimodal/dataset.py`, `fusion.py`, `trainer.py` | masked modality blocks, five fusion variants, resumable trainer |
| `calibration/scalers.py` | temperature and vector scaling with method choice on the calibration block |
| `evaluation/metrics.py`, `evaluation/stats.py` | all metrics; month-block bootstrap, Holm, Newey-West, PSR/DSR, PBO |
| `trading/costs.py`, `plan.py`, `backtest.py` | cost scenarios, deterministic planner, event-driven simulator and portfolio |
| `rag/ingest.py`, `retrieval.py`, `explain.py` | page-provenance ingestion, six retrieval methods, citations |
| `explainability/attribution.py` | leave-one-modality-out and chart-occlusion diagnostics |
| `inference/schema.py`, `inference/pipeline.py` | typed output contract; end-to-end prediction (historical and live) |
| `experiments.py`, `training/hardware.py` | run manifests and experiment log; device and precision selection |

## Data flow and artifacts

Source projects are read-only. All large outputs live under `artifacts/` (git-ignored; data licences forbid publishing data, embeddings, article or book text, or weights): `cv_embeddings/`, `samples/`, `news/`, `features/`, `rag/`, `predictions/<run>/`, `models/<run>/`, `bundles/<run>/`. Every experiment writes `experiments/<run_id>/manifest.json` (config hash, git commit, inputs, environment) and appends to `docs/EXPERIMENT_LOG.md`.

## Deviations from the suggested layout

* Labels and splits are top-level modules (`targets.py`, `splits.py`) rather than inside `data/`, because they are used by every stage.
* One Colab orchestrator notebook calls `scripts/run_pipeline.py` instead of five stage notebooks; the stages are the scripts it calls, which avoids notebook logic drifting from the package.
* The app is a single Streamlit file with eleven tabs (A-K) rather than a pages directory.

# Model card: NEXUS Alpha bundle `20261106T000935_e03_fusion_8ddffb`

## Intended use

Research and paper-trading decision support for US-listed stocks in the 500-stock TriFrame sample. It produces calibrated UP / SIDEWAYS / DOWN probabilities for six horizons (1, 2, 5, 10, 20, 21 sessions) and hands them to a separate deterministic risk engine. **Not** for live trading, not investment advice, not for stocks outside the training universe without new validation.

## Model

* **Architecture:** gated multimodal fusion (selected among late, concat, attention, gated and hybrid by mean validation log loss): six modality projections (weekly, daily and hourly chart embeddings; 80 numeric indicator features; news; SEC filings) → softmax gate over present modalities → six 3-class heads + quantile heads. 64-d modality space, modality dropout 0.15.
* **Inputs:** frozen ConvNeXt-Tiny embeddings of TriFrame r1 charts (PCA 64 per view), NEXUS multi-timeframe indicators + TriFrame's 33 visible-window features, TF-IDF/SVD and Loughran-McDonald tone of up to 7 de-duplicated articles from the 5 days before the decision, SEC filing features and text.
* **Training data:** s500 sample, as-of dates 2003-2015 (143,695 samples); early stopping on 2016; selection on 2017-2018; calibration on 2019 (temperature scaling at 10D, none elsewhere, chosen on chronological halves).
* **Decision timing:** 09:00 New York on the session after the as-of close; entry at that session's open.

## Performance (test block 2020-2022, historical diagnostic; all numbers from `reports/RESULTS.md`)

| Horizon | Log loss (class prior) | AUROC up vs down | ECE |
|---|---|---|---|
| 1D | 1.0317 (1.0487) | 0.508 | 0.014 |
| 2D | 1.0573 (1.0674) | 0.505 | 0.009 |
| 5D | 1.0674 (1.0753) | 0.511 | 0.018 |
| 10D | 1.0800 (1.0798) | 0.503 | 0.036 |
| 20D | 1.0793 (1.0835) | 0.514 | 0.038 |
| 1M | 1.0815 (1.0851) | 0.511 | 0.044 |

Most of the gain over the prior is in predicting SIDEWAYS versus a move (sideways AUROC about 0.58), a volatility effect; direction is close to chance on these years. On validation (2017-2018) the up-vs-down AUROC was 0.517-0.538 with 95% intervals above 0.5, so the risk engine's direction-skill gate is open at every horizon; the backtest (E04) and the walk-forward study (E05: up-vs-down AUROC 0.517 at 1D and 5D over 2008-2022, 0.514 at 20D with large year-to-year swings) are the evidence for whether that matters economically.

**None of the registered multimodal hypotheses is supported** (tri-frame, NLP, indicators, trainable fusion vs late, calibration). A numbers-only LightGBM (E01) and a numbers-only fusion head match or beat the full multimodal model on the test years.

**Economics (E04, 2020-2022, thresholds frozen on 2017-2018):** the gated decision policy built on this model had Sharpe 0.20 at 5D (88 trades) and −0.64 at 20D (254 trades); a model-only ATR policy 0.29 (5D) and 0.04 (20D); a rule-only policy −0.96 and −0.98; equal-weight buy-and-hold of the same survivors 0.53. All model-policy intervals include zero and deflated Sharpe ratios are about 0 (PBO 0.13 over 20 configurations). The model is not shown to support profitable trading.

## Limitations

See `docs/LIMITATIONS.md`. In brief: survivor-only universe from one vendor; test years already read by TriFrame; sealed 2023+ period not virgin; one seed; charts scale-free OHLC bars; classical text features; calibration error grows with horizon (ECE 0.036-0.044 at 10D and longer); trained on 2003-2015, so live 2026 forecasts are extrapolations.

## Ethical and legal

Data licences forbid publishing data, embeddings, article or book text and weights; the bundle stays private. Article text is never shown in full; book snippets are capped at 200 characters. No broker connection exists.

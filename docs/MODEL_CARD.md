# Model card: NEXUS Alpha bundle `20261106T020411_e08_final_model_5c4327` (v1.1)

Supersedes the v1 bundle `20261106T000935_e03_fusion_8ddffb` and the first v1.1 build `20261106T013154_e08_final_model_8bbb54` (rebuilt after audit fix A-21; same recipe, test log loss unchanged to 5 decimals). Both are kept for reproducibility. Built in the audit phase; see `docs/CHIEF_AUDIT_REPORT.md`.

## Intended use

Research and paper-trading decision support for US-listed stocks in the 500-stock TriFrame sample. It produces UP / SIDEWAYS / DOWN probabilities and a 10-50-90% return range for six horizons (1, 2, 5, 10, 20, 21 sessions) and hands them to a separate deterministic risk engine. **Not** for live trading, not investment advice, not for stocks outside the training universe without new validation.

## Model

* **Recipe (E08, chosen on 2017-2018 validation among 68 candidates):** probability blend 0.6 × fusion ensemble + 0.3 × convex logistic + 0.1 × LightGBM, then per-horizon calibration.
  * *Fusion ensemble:* five seeds (7, 11, 23, 42, 101) of the gated multimodal fusion network (weekly, daily and hourly chart embeddings; 80 numeric features; news; SEC). Settings from E06: learning rate 3e-4, linear warm-up then cosine decay, dropout 0.3, patience 6.
  * *Convex logistic:* multinomial logistic regression on the standardised numeric features, L2 1e-3. Its optimum is certified as the unique global minimum (E09).
  * *LightGBM:* one model per horizon on the raw numeric features, with E07's tuned parameters (25-trial TPE search on validation) and tree counts from early stopping on 2016 only.
* **Refit:** every component was refit on 2003-2018, with the epoch and tree counts fixed from the selection stage (no early stopping on held-out data during the refit).
* **Calibration:** temperature scaling at 10D, 20D and 1M; none at 1D-5D. Chosen on chronological halves of 2019 and fitted on 2019.
* **Return range:** the fusion seeds' quantile heads (0.1, 0.5, 0.9 of the volatility-scaled return), averaged and sorted, then multiplied by σ_t·√h. A horizon's range is shown only if its 10-90% band covered 80% ± 8 points on validation. All six pass, at 0.83-0.84.
* **Decision timing:** 09:00 New York on the session after the as-of close; entry at that session's open.

## Performance (test block 2020-2022, historical diagnostic)

| Horizon | Log loss (class prior) | Gain vs prior, 95% CI | Gain vs v1 (E03) | AUROC up vs down | AUROC sideways | ECE | 10-90% coverage |
|---|---|---|---|---|---|---|---|
| 1D | 1.0287 (1.0487) | +0.0200 [+0.015, +0.026] | +0.0030 | 0.511 | 0.613 | 0.023 | 0.841 |
| 2D | 1.0539 (1.0674) | +0.0134 [+0.009, +0.019] | +0.0034 | 0.514 | 0.594 | 0.020 | 0.829 |
| 5D | 1.0639 (1.0753) | +0.0114 [+0.007, +0.017] | +0.0035 | 0.508 | 0.587 | 0.015 | 0.825 |
| 10D | 1.0742 (1.0798) | +0.0057 [−0.003, +0.016] | +0.0058 | 0.497 | 0.577 | 0.038 | 0.825 |
| 20D | 1.0742 (1.0835) | +0.0092 [−0.004, +0.025] | +0.0050 | 0.495 | 0.593 | 0.038 | 0.825 |
| 1M | 1.0769 (1.0851) | +0.0082 [−0.006, +0.025] | +0.0046 | 0.494 | 0.594 | 0.042 | 0.821 |

Mean test log loss is 1.0620 (v1: 1.0662). The gain over v1 is significant at 1D-10D, using month-block bootstrap CIs over 36 blocks. Accuracy is 0.42-0.49. It is dominated by SIDEWAYS, and balanced accuracy is 0.34-0.37. Models are selected on log loss, a proper scoring rule, not on accuracy.

**What the model knows.** It knows whether a stock will move: SIDEWAYS AUROC is 0.58-0.61. Direction is close to chance on these years. Up-versus-down AUROC is 0.49-0.52 on the full block, and 20D swings from 0.41 in 2020 to 0.56 in 2021. Placebo controls (E10) show that about a third of the 1D gain is date-level, market-wide timing. A within-date label shuffle keeps that part; a global shuffle removes everything.

**Direction-skill gate.** The gate opens a horizon when its validation up-vs-down AUROC has a 95% interval above 0.5. It is closed at 1D and 2D and open at 5D-1M (validation AUROC 0.52-0.54). On 2020-2022, direction skill at 10D-1M was not present. The gate is statistical, not economic (audit finding A-12).

**Economics (E04 v1.1: fixed engine, new model, thresholds frozen on 2017-2018 with ≥ 30 validation trades):** the gated live planner had Sharpe 0.31 at 5D (41 trades, CI [−0.64, 2.08]) and −0.14 at 20D (322 trades, CI [−1.19, 0.77]). The model-ATR policy had 0.02 and 0.16, and the rule-only policy −0.87 and −0.34. Equal-weight buy-and-hold of the same survivors had 0.53. Deflated Sharpe ≈ 0 and PBO 0.12. **The model is not shown to support profitable trading.** Details: `docs/CHIEF_AUDIT_REPORT.md` section 5.7.

**Calibration finding.** On the test years, no calibration (1.0613) beat static 2019 calibration (1.0625) and walk-forward annual recalibration (1.0617) at the selection stage. The pre-registered procedure is kept, because switching now would be a test-set choice. It is flagged for the next study.

## Limitations

See `docs/LIMITATIONS.md`. In brief:
* Survivor-only universe from one vendor.
* The test years were already read by TriFrame and have been re-scored by six NEXUS experiments.
* The sealed 2023+ period is not virgin.
* Charts are scale-free OHLC bars, and the text features are classical. Transformer text was tested inside the
  fusion in E28 (FinBERT news, ChronoBERT-2010 filings, an SEC price-impact expert) and not kept: it beat a
  placebo but not the classical blocks (`docs/V3_RESEARCH.md`, E28 results).
* Calibration error grows with horizon (ECE 0.038-0.042 at 10D and longer).
* The model was trained on 2003-2018, so live 2026 forecasts are extrapolations.

## Ethical and legal

Data licences forbid publishing data, embeddings, article or book text and weights, so the bundle stays private. Article text is never shown in full, and book snippets are capped at 200 characters. No broker connection exists.

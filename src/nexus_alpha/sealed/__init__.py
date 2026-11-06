"""The one-time joint sealed evaluation (docs/SEALED_EVALUATION_PLAN.md) and its rehearsal on a development year.

Every evaluation row is rebuilt from raw bars, the news corpus and SEC events with the same functions that built the
development frames and that inference uses (structured indicators, TriFrame's visible-window features, the chart
renderer and frozen backbone, rule detectors, label rules, news and SEC aggregation with the frozen TF-IDF models).
Reading anything on or after 2023-01-03 requires `seal.open_for_evaluation()` (the owner's unlock) in this process.

The rehearsal runs the identical code on development rows (2022, the last development year) and must reproduce the
stored development features, embeddings and predictions; that is the acceptance test for the sealed path.
"""

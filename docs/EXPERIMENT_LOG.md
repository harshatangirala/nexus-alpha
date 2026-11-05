# Experiment log

One line per run, appended automatically by `nexus_alpha.experiments.Run`. Failed and abandoned runs stay in the log so the full search is counted.

| run_id | name | profile | status | elapsed | summary |
|---|---|---|---|---|---|
| `20261105T194117_e01_structured_baselines_098364` | e01_structured_baselines | local-cpu | COMPLETED | 290.7 s | valid-selected best structured model: lgbm_struct_x33; test log loss by horizon 1D 1.0312 vs prior 1.0487, 2D 1.0565 vs prior 1.0674, 5D 1.0662 vs prior 1.0753, 10D 1.0740 vs prior 1.0798, 20D 1.0744 vs prior 1.0835, 1M 1.0768 vs prior 1.0851 |
| `20261105T194747_e02_cv_branch_93f5ac` | e02_cv_branch | local-cpu | COMPLETED | 4610.3 s | best frozen backbone (valid, WDH, linear): vit_small_patch14_dinov2; images-only valid log loss WDH vs prior-free 1D 1.0415, 2D 1.0580, 5D 1.0678, 10D 1.0744, 20D 1.0787, 1M 1.0796 |
| `20261105T235511_rag_eval_088152` | rag_eval | local-cpu | COMPLETED | 248.8 s | best within-document method by MRR: within_document/hybrid_rerank (0.657); rule lookup MRR 0.896 |

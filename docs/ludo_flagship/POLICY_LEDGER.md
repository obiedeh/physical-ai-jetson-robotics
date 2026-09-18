# Policy ledger

One row per trained policy version. Negative results are rows.

| Date (UTC) | Policy id | Data (dataset, episodes, hash) | Recipe and key hyperparameters | Checkpoint hash | Evaluation artifact | Result | Decision |
|---|---|---|---|---|---|---|---|
| 2026-08-20 | ludo_groot17_v1 | ludo_corpus01 (sim, target-blind) | GR00T N1.7 LoRA | see reports/training/ludo_groot17_v1 | reports/ludo_groot17_eval01 | 0/20 corrected | Corpus made target-observable |
| 2026-08-21 | ludo_groot17_v2_tobs | ludo_corpus03_tobs (sim, target-observable) | GR00T N1.7 LoRA | see reports/training/ludo_groot17_v2_tobs | reports/ludo_groot17_eval02_tobs | 0/20, 17 never grasped | Real demonstrations before further sim retrains |

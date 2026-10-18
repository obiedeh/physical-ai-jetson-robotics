# Simulation policy ledger

One row per policy trained in the simulation lane. Negative results are rows.
Nothing here is physical evidence: these policies train on Isaac takes, run in the
Isaac scene, and never count toward D1 to D5. The physical lane keeps its own
[policy ledger](POLICY_LEDGER.md); the trainer refuses to put a simulation row
there, or a physical row here.

Rows are appended by the trainer with the same eight columns as the physical
ledger, plus `lane=simulation` in the recipe cell and the trained episode count as
`N of M episodes trained`, so a data-scaling study reads straight off the table.

| Date (UTC) | Policy id | Data (dataset, episodes, hash) | Recipe and key hyperparameters | Checkpoint hash | Evaluation artifact | Result | Decision |
|---|---|---|---|---|---|---|---|

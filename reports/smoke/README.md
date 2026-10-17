# Console smoke capture reports

Copied from the console workspace (`smoke-reports/`) after a disposable smoke
capture. The console writes them when the smoke context closes; the dataset
itself is disposed and never counts.

- `physical_smoke_<UTC timestamp>.json`: one physical capture through the
  console. Commit it with its activity log entry.
- Simulation and synthetic demo smoke reports stay in the workspace unless a
  simulation result is being cited.

Status: no physical capture recorded yet. **Implemented, unmeasured.**

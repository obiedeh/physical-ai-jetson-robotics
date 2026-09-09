# Evidence Standard

This repo treats proof as committed, reproducible artifacts tied to a specific run or validation step.

## What Counts As Proof

- A command or launch path that another reviewer can run.
- Logs, screenshots, metrics, or reports produced by that run.
- Hardware and software version context for Jetson or robot execution.
- Clear pass/fail notes, limitations, and next validation steps.
- CI, tests, smoke checks, or schema validation where applicable.

## What Does Not Count As Proof

- Planned capabilities without artifacts.
- Synthetic or mock output presented as real hardware validation.
- Jetson or Thor claims without committed device logs, versions, and metrics.
- Screenshots or reports without enough context to reproduce the run.
- Benchmark language without run duration, configuration, latency, throughput, and memory data.

## Minimum Evidence Checklist

- Reproducible command or launch instruction.
- Input source and configuration used.
- Runtime artifact path under `reports/`.
- Observed metrics or validation notes.
- Known limitations.
- Next validation step.


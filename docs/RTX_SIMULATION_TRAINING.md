# RTX 5090 Simulation and Training

Updated 2026-09-07. This page covers the synthetic reaching utility, not the
full Ludo/GR00T pipeline. Device benchmarks and simulation results already exist;
see [README](../README.md). Robot-dependent smoke runs require locally obtained
[vendor assets](VENDOR_INTEGRATION_MAP.md); they have not been rerun after removal.

This track runs high-impact simulation and training work on the Linux RTX 5090
workstation before Jetson Orin/Thor or real robot hardware is needed.

## Synria Synthetic Reaching Policy

Train a compact CUDA-backed reaching policy from synthetic arm state and target
pose data:

```bash
physical-ai-lab train-synria-reach --device cuda --samples 8192 --epochs 40
```

Default outputs:

```text
reports/training/synria_reach_policy.json
runs/rtx_training/synria_reach_policy.pt
```

The checkpoint is generated output and remains outside git under `runs/`.

## Simulation Smoke Runs

Run Synria MoveIt smoke evidence:

```bash
bash scripts/linux_rtx/run_synria_moveit_smoke.sh
```

## One-Command RTX Suite

```bash
bash scripts/linux_rtx/run_rtx_training_suite.sh
```

This trains the Synria synthetic reaching policy.

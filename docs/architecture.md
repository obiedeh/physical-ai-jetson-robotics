# Architecture

## System Purpose

This repository is the flagship Physical AI systems platform in the portfolio. It organizes ROS 2 robot descriptions, OpenUSD / Isaac simulation assets, Jetson-oriented runtime planning, robot telemetry, and evidence reports into a repeatable validation workflow.

## Current Implementation Status

- **Implemented:** Python package and CLI, ROS 2 description packages, RViz configs, OpenUSD asset staging, simulation/report templates, tests, and evidence documentation.
- **Runnable scaffold:** validation scripts, report templates, and ROS 2 / simulation smoke-test structure.
- **Planned Jetson deployment:** Jetson runtime evidence, edge inference metrics, sustained-run logs, and hardware validation artifacts.
- **Future hardware validation:** real robot runs and sim-to-real evidence after safety and readiness gates are met.

## Main Components

- `physical_ai_lab/`: Python CLI, configuration, telemetry, and reporting utilities.
- `ros2_ws/src/`: ROS 2 robot descriptions, display launches, RViz configs, MoveIt/Gazebo scaffolds.
- `isaac/`: OpenUSD scenes, robot asset staging, and Isaac-oriented simulation assets.
- `scripts/`: platform setup and validation helpers for Windows, Linux RTX, and Jetson paths.
- `reports/`: evidence templates and completed validation notes.
- `docs/diagrams/`: Mermaid architecture views for reviewer inspection.

## Runtime Flow

Developer or reviewer workflows start from local commands such as `pytest -q`, ROS 2 description smoke checks, OpenUSD inspection, or report-generation scripts. Those commands validate robot descriptions and simulation assets before any Jetson or hardware evidence is claimed.

## Data / Telemetry Flow

Robot descriptions, simulation assets, run commands, and logs become validation artifacts. Runtime telemetry and Jetson metrics are planned evidence inputs, not claimed benchmark results until committed under `reports/` or `artifacts/`.

## Deployment Modes

- **Local development:** Python tests, ROS 2 description checks, RViz display configs, and OpenUSD asset inspection.
- **Linux RTX workstation:** OpenUSD / Isaac simulation and rendering workflows where supported by the host environment.
- **Planned Jetson deployment:** edge inference, telemetry capture, latency summaries, memory snapshots, and sustained-run notes.
- **Future robot hardware:** physical runs only after evidence gates and limitations are documented.

## Evidence Artifacts

- Existing evidence lives primarily in `reports/`.
- New reviewer-facing placeholders live in `artifacts/sample-inputs/`, `artifacts/sample-outputs/`, `artifacts/logs/`, and `artifacts/reports/`.
- Diagram sources live in `docs/diagrams/`.

## Known Limitations

- Jetson benchmark numbers are not claimed until hardware artifacts are committed.
- Isaac Sim / OpenUSD assets are validation scaffolds unless paired with run logs and screenshots.
- ROS 2 descriptions and RViz configs do not prove safe physical robot operation.

## Next Validation Step

Publish one reproducible end-to-end robotics or Jetson demo with run command, logs, metrics, screenshots, limitations, and next validation step.

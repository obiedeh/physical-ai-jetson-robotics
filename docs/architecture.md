# Architecture

Updated 2026-09-07. Active implementation; physical autonomous manipulation is not established.

## Current Paths

| Path | Implementation | Evidence / limit |
| --- | --- | --- |
| Core utilities | `physical_ai_lab/`, CLI, telemetry, triage; `edge_ai/` ONNX runtime | Unit/static checks and report artifacts; mock outputs are not device measurements |
| Ludo simulation | `isaac/scripts/ludo_turn_executor.py`, Isaac Lab task/MDP, game adapters | Default PPO/scripted executor; optional separate GR00T lane; kinematic attach and release-hold. [Frozen results](../reports/ludo_stats_frozen/) |
| Thor inference | External Isaac-GR00T checkout, checkpoints and TensorRT engines | [Measured eager/TRT benchmark](../reports/thor_trt_benchmark/thor_trt_benchmark.json), not a real-arm deployment result |
| Orin board I/O | `robots/lerobot_robot_rosmaster_m3pro/`: ROS hardware host, ZMQ client, keyboard recorder | [Device probes](../reports/jetson/yahboom_day_one/) and [verified recording-path notes](notes/yahboom_strategy_2026-08-20.md); human demonstrations pending in that record |
| Robot descriptions | First-party ROS xacro wrappers, mock-control and MoveIt configuration | Vendor inputs fetched locally; no geometry redistributed in current tree |

## Hardware Boundary

The Orin host owns ROS 2 Humble board I/O in Python 3.10. The newer LeRobot
recording environment runs on the RTX client over ZMQ. Arm state is derived from
commands because the board provides no joint feedback. Active control uses
host/hardware deadman handling; passive observation does not own board control.

The RTX host also runs Isaac Sim 5.1 / Isaac Lab simulation and policy training.
Thor's GR00T timings are a separate device benchmark, not proof that the Ludo
simulation executor controls a physical Synria arm through TensorRT.

## Evidence Flow

Device probes, simulation attempts and recording workflows emit logs and
provenance into `reports/`. Run conditions and scoring rules travel with results.
The raw GR00T eval01 result is retained alongside its corrected 0/20 sidecar.
Short device benchmarks do not establish sustained thermal or safety validation.

Views: [system](diagrams/system-architecture.mmd), [deployment](diagrams/deployment-view.mmd),
[runtime](diagrams/runtime-flow.mmd), [data flow](diagrams/data-flow.mmd).

## Remaining Work

Autonomous real-arm pick/place, physical object-success ground truth, numeric
TensorRT/PyTorch parity and sustained thermal/safety validation remain unestablished.
See [NOT_CLAIMED](../reports/NOT_CLAIMED.md). ROS mock control and simulation scoring
do not authorize physical operation.

[Vendor setup](VENDOR_INTEGRATION_MAP.md), [RTX environment](SETUP_RTX.md),
[Jetson deployment](JETSON_DEPLOYMENT.md) and the [root README](../README.md)
describe prerequisites and the current evidence.

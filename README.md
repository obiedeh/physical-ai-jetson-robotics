# Physical AI Jetson Robotics

Robot simulation, policy inference, teleoperation and evidence collection on Jetson AGX Thor, Jetson Orin NX and an RTX 5090 host.

**[Visual evidence: device charts, simulation and corrections](docs/evidence/README.md)**

## Status

**Active and incomplete.**

- Thor GR00T inference and Orin CUDA/camera probes run through to recorded benchmark artifacts, linked below.
- The Isaac Sim Ludo executor runs commands through scoring and reports, with scripted components and kinematic attach.
- The Orin ROS/ZMQ host and RTX keyboard recording path are [verified end to end](docs/notes/yahboom_strategy_2026-08-20.md). That record leaves human demonstration collection pending.

**Not established:** autonomous real-arm pick and place; physical object-success ground truth; TensorRT/PyTorch numerical parity; sustained thermal and safety validation. See [NOT_CLAIMED](reports/NOT_CLAIMED.md).

## Status by Layer

What is real, in one table. Labels: **measured** (a committed artifact with device and date), **implemented, unmeasured** (runs, no performance artifact), **scaffold** (code or plan only), **planned** (not started).

| Layer | Status | Date | Hardware | Artifact |
|---|---|---|---|---|
| GR00T inference, eager and TensorRT | **measured** | 2026-08-20 | Jetson AGX Thor 128GB, 120 W | [`reports/thor_trt_benchmark/thor_trt_benchmark.json`](reports/thor_trt_benchmark/thor_trt_benchmark.json) |
| Device probes: CUDA matmul, camera grab, thermal, power | **measured**, 9.9 s run | 2026-08-20 | Jetson Orin NX, MAXN_SUPER | [`reports/jetson/yahboom_day_one/day_one_smoke.json`](reports/jetson/yahboom_day_one/day_one_smoke.json) |
| Ludo expert executor, scripted plus PPO reach, seeds 10-12 | **measured**, simulation | runs 2026-08-19, aggregate 2026-08-20 | Isaac Sim on RTX 5090 | [`reports/ludo_stats_frozen/`](reports/ludo_stats_frozen/) |
| GR00T policy lane evaluation | **measured**, simulation; result and its correction are in [Corrections](#corrections-and-retractions) | 2026-08-20 | Isaac Sim on RTX 5090 | [`reports/ludo_groot17_eval01/`](reports/ludo_groot17_eval01/) |
| M3 Pro recording path: Orin ROS/ZMQ host, RTX keyboard client | **implemented, unmeasured**; verified end to end, no human demonstrations collected | 2026-08-20 | Jetson Orin NX, RTX 5090 host | [`docs/notes/yahboom_strategy_2026-08-20.md`](docs/notes/yahboom_strategy_2026-08-20.md) |
| Physical dice roll and integrated game, simulation demos | **implemented, unmeasured**; videos, not scored runs | 2026-08-24 to 2026-08-27 | Isaac Sim on RTX 5090 | [`docs/notes/ludo_game_milestones_2026-08-21.md`](docs/notes/ludo_game_milestones_2026-08-21.md) |
| Vendor asset staging with receipts and hashes | **implemented**; no vendor assets in this repository | 2026-09-07 | any | [`scripts/fetch_vendor_assets.py`](scripts/fetch_vendor_assets.py), [`docs/VENDOR_ASSET_REMOVAL.md`](docs/VENDOR_ASSET_REMOVAL.md) |
| Rover board contract and first motion tests: /cmd_vel, /arm6_joints, /odom_raw, /scan0, /scan1, /imu/data_raw, /battery | **implemented, unmeasured**; topics verified read-only and three operator-confirmed motion tests, recorded in prose, no artifact | 2026-08-20 | Jetson Orin NX, ROSMASTER M3 Pro on stand | [`docs/notes/yahboom_strategy_2026-08-20.md`](docs/notes/yahboom_strategy_2026-08-20.md) |
| Rover base deadman and safety bridge: LiDAR min-obstacle, battery and deadman rules posted as safety events | **implemented, unmeasured**; the firmware has no /cmd_vel timeout, the host-side deadman exists, detection-to-stop latency never timed | 2026-08-20 | Jetson Orin NX, RTX 5090 host | [`robots/lerobot_robot_rosmaster_m3pro/`](robots/lerobot_robot_rosmaster_m3pro/), [`scripts/jetson/m3pro_watchdog_test.py`](scripts/jetson/m3pro_watchdog_test.py) |
| Rover joystick and scripted teleop: js0 arm and base, m3pro_pub | **implemented, unmeasured**; no timing or accuracy artifact | 2026-08-20 | Jetson Orin NX | [`scripts/jetson/`](scripts/jetson/) |
| Orin NX software inventory capture | **implemented**; read-only command record, not a measurement | 2026-08-20 | Jetson Orin NX | [`reports/inventory/yahboom_orin_nx.json`](reports/inventory/yahboom_orin_nx.json) |
| Rover ROS 2 packages: rosmaster_m3pro description, control, bringup, hardware, perception, simulation, moveit_config | **scaffold**; configuration files only, no executables, the description includes a vendor URDF that is not in this repository | | | [`ros2_ws/src/`](ros2_ws/src/), [`ros2_ws/src/rosmaster_m3pro_hardware/config/hardware_interface.yaml`](ros2_ws/src/rosmaster_m3pro_hardware/config/hardware_interface.yaml) |
| Rover Isaac Sim model, first-party primitive geometry | **scaffold**; chassis, camera and LiDAR dimensions are listed as measurement debt | | | [`isaac/usd/robots/rosmaster_m3pro/`](isaac/usd/robots/rosmaster_m3pro/), [`docs/rosmaster_m3pro/MILESTONE_1_DESCRIPTION.md`](docs/rosmaster_m3pro/MILESTONE_1_DESCRIPTION.md) |
| SLAM helpers: mecanum odometry calibration, map store, navigation plans | **scaffold**; demo data and unit tests only, no hardware calibration session recorded | | | [`slam/`](slam/), [`reports/slam/README.md`](reports/slam/README.md) |
| Rover SLAM mapping run, loop closure and re-localisation | **planned**; the vendor SLAM stack lives on the Orin, not in this repository, and no map has been recorded | | | [`docs/YAHBOOM_ROBOT_CAR_PLAN.md`](docs/YAHBOOM_ROBOT_CAR_PLAN.md) |
| Rover navigation and obstacle avoidance, Nav2 or vendor stack | **planned** | | | [`docs/YAHBOOM_ROBOT_CAR_PLAN.md`](docs/YAHBOOM_ROBOT_CAR_PLAN.md) |
| Rover sensor and odometry calibration: LiDAR mount poses, wheel odometry scale, camera intrinsics | **planned**; no calibration data in this repository | | | [`docs/rosmaster_m3pro/MILESTONE_1_DESCRIPTION.md`](docs/rosmaster_m3pro/MILESTONE_1_DESCRIPTION.md) |
| Video-to-data layer | **scaffold**, feasibility record only | 2026-08-21 | RTX 5090 | [`docs/notes/video_to_data_layer_plan_2026-08-21.md`](docs/notes/video_to_data_layer_plan_2026-08-21.md) |
| Autonomous real-arm pick and place | **planned** | | | [`reports/NOT_CLAIMED.md`](reports/NOT_CLAIMED.md) |
| TensorRT versus PyTorch numerical parity | **planned**, never recorded | | | [`reports/NOT_CLAIMED.md`](reports/NOT_CLAIMED.md) |
| Sustained thermal and safety validation | **planned**; existing runs are 60 s and 9.9 s | | | [`reports/NOT_CLAIMED.md`](reports/NOT_CLAIMED.md) |

## Measured Results

[Charts, distributions, source limits and media](docs/evidence/README.md).

| Measurement | Hardware | Date (UTC) | Result | Evidence |
| --- | --- | --- | --- | --- |
| GR00T PyTorch eager | AGX Thor 128GB, 120W | 2026-08-20 | **126.6 ms median, 7.9 Hz** | [benchmark](reports/thor_trt_benchmark/thor_trt_benchmark.json), [log](reports/thor_trt_benchmark/bench.log) |
| GR00T TensorRT | AGX Thor 128GB, 120W | 2026-08-20 | **101.6 ms median, 9.8 Hz** | [benchmark](reports/thor_trt_benchmark/thor_trt_benchmark.json) |
| FP16 CUDA matmul | Orin NX, MAXN_SUPER | 2026-08-20 | 2048 matrix size; 200 warm samples; **9.72 TFLOPS at p50** | [day-one probe](reports/jetson/yahboom_day_one/day_one_smoke.json), [provenance](reports/jetson/yahboom_day_one/provenance.json) |
| V4L2 camera grab | Orin NX | 2026-08-20 | 640 x 480, 60 frames; **16.7 FPS from p50 interval** | [day-one probe](reports/jetson/yahboom_day_one/day_one_smoke.json), [provenance](reports/jetson/yahboom_day_one/provenance.json) |
| Thermal / input power | Orin NX | 2026-08-20 | Peak **61.781 C**; VDD_IN peak **11616.2 mW**; no throttling suspected by the probe | [day-one probe](reports/jetson/yahboom_day_one/day_one_smoke.json) |
| Ludo, Isaac Sim, post-fix seeds 10-12 | RTX 5090 | Runs 2026-08-19; aggregate 2026-08-20 | **32/36 first tries**; **35/36 turns** after the executor's built-in single retry (3 of 36 turns succeeded on retry, 4 retries ran); successful first-try error p50 **19.2 mm**, p95 **24.6 mm** | [frozen aggregate](reports/ludo_stats_frozen/stats_2026-08-20T050701Z_reports_ludo_soak_s1%5B012%5D.json), [run provenance](reports/ludo_soak_s10/provenance.json) |

Thor conditions: BF16, batch 1, 100 iterations after 10 warmups; E48 checkpoint-35000, simulated dataset inputs. These are measured end-to-end timings, not sums of component medians. Parity against PyTorch was not recorded.

Thor GPU temperature peaked at **42.0 C**, with no throttling observed over the benchmark run.
[Thermal record](reports/thor_trt_benchmark/thor_trt_benchmark.json).

**The thermal figures are short benchmarks, not soak tests.** The Orin probe ran for 9.9 seconds. Camera grab timing is not perception latency.

**Ludo is simulation, not physical manipulation.** The executor includes scripted grasp/endgame/fallback behavior, kinematic attach and release-hold patches. The selected block uses the same recorded scoring rule: final error below 35 mm and tilt below 15 degrees, judged after hold/withdrawal. Earlier configurations and the separately scored GR00T evaluation are not pooled into it. The sentinel-bearing clearance percentile is omitted.

## Corrections and Retractions

- **The 128 ms / 7.8 Hz TensorRT attribution is retracted.** It was a torch.compile baseline. The TensorRT stage failed on a missing `matplotlib` import. The first successful recovered TensorRT benchmark is the 101.6 ms / 9.8 Hz run above. [Correction record](docs/notes/instrumentation_pass_2026-08-19.md)
- **GR00T Ludo eval01 is 0/20, not 1/20.** The apparent success never touched the cup; adjacent targets allowed the untouched cup to satisfy the geometric scoring rule. The raw artifact remains, with a [corrected sidecar](reports/ludo_groot17_eval01/session_summary_corrected.json).
- **That target-blind GR00T run is not a fair goal-conditioned comparison.** Its placement goal was neither visible nor encoded in the constant instruction. It cannot establish goal-conditioned placement ability. [Analysis](docs/notes/groot17_eval01_findings_2026-08-20.md)

[NOT_CLAIMED](reports/NOT_CLAIMED.md) records the limits of these results.

## Architecture

The RTX host runs Isaac simulation, training and the recording client. The Orin host owns ROS board I/O behind a ZMQ boundary. Thor runs the separately measured GR00T inference workload. Logs, provenance, scoring and corrections are retained with their runs.

Views: [system](docs/diagrams/system-architecture.mmd), [deployment](docs/diagrams/deployment-view.mmd), [runtime](docs/diagrams/runtime-flow.mmd), [data flow](docs/diagrams/data-flow.mmd). Details: [architecture](docs/architecture.md).

## Design Decisions

| Decision | Rejected alternative and reason | Record |
| --- | --- | --- |
| ROS on the Orin, LeRobot client over ZMQ | Installing the newer LeRobot stack into the Orin Python 3.10 ROS environment: the required Python versions conflict | [hardware boundary](robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/hardware.py), [host](robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/m3pro_host.py) |
| Recorded arm state is command-derived | Assuming joint feedback: the board does not publish it; commanded pose does not prove achieved pose | [hardware contract](robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/hardware.py) |
| Declare simulator contact patches | Treating a fitting contact approximation as task success: broken contact behavior required attach/release intervention, limiting what the score proves | [limitations](reports/NOT_CLAIMED.md), [executor](isaac/scripts/ludo_turn_executor.py) |
| Host-side deadman and disconnect stop in active control | Relying on base-command expiry: the documented test found motion continued after command publication stopped | [timeout finding](docs/notes/yahboom_strategy_2026-08-20.md), [implementation](robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/hardware.py) |
| Keyboard recording input | Continuing with the bundled joystick: the documented faulty axes/buttons blocked arm input | [bring-up record](docs/notes/yahboom_strategy_2026-08-20.md), [recorder](robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/record_kbd.py) |

## Hardware and Environment

- **Jetson AGX Thor 128GB:** benchmark at 120W, L4T R38.4.0, BF16 GR00T n1d7-family pipeline. [Device/run record](reports/thor_trt_benchmark/thor_trt_benchmark.json)
- **Jetson Orin NX:** Yahboom M3 Pro host, L4T R36.4.4, ROS 2 Humble and Python 3.10.12. Probe records TensorRT 10.7.0.23 and PyTorch 2.5.0a0+872d972e41.nv24.08. [Provenance](reports/jetson/yahboom_day_one/provenance.json)
- **RTX 5090 development host:** Isaac Sim 5.1 / Isaac Lab 2.3.2 simulation environment. [Setup](docs/SETUP_RTX.md), [recorded run environment](reports/ludo_soak_s10/provenance.json)

The **Isaac-GR00T checkout, TensorRT engines and required GR00T checkpoints are external**, not vendored here. Historical report paths can refer to files on the original machines. Isaac Sim/Lab, vendor ROS runtime/message packages, robot geometry and external game-table scenes must also be obtained separately. Root-package installation does not install those environments.

**Data availability.** Raw simulation corpora (`reports/ludo_corpus*`), LeRobot conversions and fine-tuned checkpoints (`reports/training/ludo_*`, `reports/training/synria_e46_lerobot*`), per-run frame dumps, per-frame metadata and per-run videos are not distributed in this repository and are not under Git LFS. Where the experiment ledger or notes name those paths, they describe local artifacts on the original machines, available on request. Each run directory keeps its provenance, session summary and turn records tracked, so every number cited in this README and the [evidence page](docs/evidence/README.md) resolves to a tracked file.

## Setup

Core Python utilities and static checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_ros2_description_yahboom.py tests/test_simulation_foundation.py
python scripts/fetch_vendor_assets.py --help
```

Vendor-dependent checks skip when optional assets are absent. Obtain the vendor sources, stage them locally and regenerate USDs using the [vendor setup path](docs/VENDOR_INTEGRATION_MAP.md#setup). Missing downloads fail explicitly; converted vendor assets are ignored by Git. This does not yet reproduce every historical scene or contact patch.

Use [RTX setup](docs/SETUP_RTX.md) for Isaac and [Jetson deployment](docs/JETSON_DEPLOYMENT.md) for the device paths. The M3 Pro recorder uses a separate newer LeRobot environment; the root `robot-learning` extra is not that environment. Hardware operation requires the operator present and the documented device checks.

## What Is Next

- Record tolerance-checked TensorRT/PyTorch output deltas.
- Validate physical-arm pick/place with observed object-success ground truth.
- Collect human keyboard demonstrations through the verified recording path.
- Run sustained thermal/power and safety validation, including command-loss behavior.

## License and Attribution

First-party work is [MIT](LICENSE). That grant does not cover vendor robot assets, external checkpoints/engines or bundled third-party skills. Users obtain robot assets directly from their vendors under their terms; [NOTICE](NOTICE) and the [removal manifest](docs/VENDOR_ASSET_REMOVAL.md) record provenance and unresolved terms.

Bundled NVIDIA `.claude/skills/` material has a [separate notice](NOTICE#bundled-nvidia-skills) and [Apache-2.0 license](LICENSES/Apache-2.0.txt).

**History note.** This public repository starts from a clean tree on 2026-09-09 with no vendor geometry in any commit. The development history before that date, which contained the removed vendor files, is retained in a private repository and is not published.

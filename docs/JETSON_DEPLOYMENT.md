# Jetson Deployment

Updated 2026-09-07. Device measurements exist; autonomous real-arm manipulation
and sustained validation are not established.

| Device | Recorded work | Target work, not established here |
| --- | --- | --- |
| AGX Thor 128GB | GR00T eager/TRT inference, 120W mode; [2026-08-20 benchmark](../reports/thor_trt_benchmark/thor_trt_benchmark.json) | Synria autonomous pick/place, observed object success, numerical parity and sustained thermal validation |
| Orin NX on Yahboom M3 Pro | CUDA matmul, camera grab, thermal/power [probes](../reports/jetson/yahboom_day_one/); ROS/ZMQ keyboard recording path [verified in notes](notes/yahboom_strategy_2026-08-20.md) | Human demonstration corpus, autonomous manipulation, calibrated navigation/SLAM and sustained safety validation |

## Environment Boundary

The recorded Orin environment uses L4T R36.4.4, Python 3.10.12 and ROS 2 Humble.
The board host depends on ROS messages, NumPy, OpenCV and pyzmq, not LeRobot.
The client recorder uses a separate newer LeRobot/Python environment on the RTX
host. Do not install the root `robot-learning` extra as a substitute for that
client stack. The plugin's dependency metadata and runtime Python requirements
still need reconciliation before claiming a portable installation recipe.

Thor's benchmark uses L4T R38.4.0 and external Isaac-GR00T engines/checkpoint.
See the artifact for precision, batch size, warmups, iterations and recorded
thermal conditions. Engine/checkpoint files are not distributed here.

## Setup Boundary

1. Obtain vendor packages and geometry through [vendor setup](VENDOR_INTEGRATION_MAP.md).
2. Source the robot's installed ROS 2 Humble environment and vendor workspace for
   board/message packages; those runtime packages are external.
3. Configure the host/client addresses, ROS domain and DDS profile for your own
   environment. See `m3pro_host.py` and `client.py` under
   `robots/lerobot_robot_rosmaster_m3pro/lerobot_robot_rosmaster_m3pro/`.
4. Recheck device identity, power mode, camera access and operator stop behavior
   before any physical session. The recorded configuration is not an automatic
   deployment or safety approval.

## Control Semantics

`hardware.py` records commanded arm pose, not joint feedback. Active control
clamps commands and runs a deadman; the host adds a command-loss watchdog.
The documented vendor base test continued moving after command publication
stopped. Passive joystick observation has different control ownership and does
not provide the active host's stop guarantee.

The verified keyboard recording path is not evidence of autonomous object
success. An operator must be present for physical work.

## Evidence

Retain device identity, runtime versions, exact command, source revision, data
kind, timing distribution, power mode and short-run versus sustained scope.
The Thor and Orin artifacts above are short measurements. Sustained thermal,
command-loss and safety validation remain next steps.

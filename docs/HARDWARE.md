# Hardware Plan

## Sim-to-Real Robot/Jetson Pairings

After successful simulation on the RTX 5090 workstation, each robot deploys to its bound Jetson:

| Robot | Edge compute | Primary track |
|---|---|---|
| Synria 6DOF arm (650 mm reach, 1 kg payload) + Synria C10 wrist camera | NVIDIA Jetson AGX Thor | Sim-to-real manipulation — chess and Ludo board-play demos |
| Yahboom ROSMASTER M3 Pro (mecanum base + onboard 6DOF arm + display) | NVIDIA Jetson Orin NX 8GB (ships with the M3 Pro) | SLAM, computer vision, and onboard-arm tasks |

Synria runs on AGX Thor because chess/Ludo play stacks perception (board-state recognition), planning (engine-driven move selection), and manipulation in one runtime — Thor's higher TOPS and memory headroom are the right fit for the combined perception + policy workload.

Yahboom runs on the Orin NX 8GB the platform ships with, which is appropriate for SLAM, navigation, and onboard arm control.

Shared peripherals:

- USB/CSI cameras
- optional IMU, wheel encoders, and LiDAR

## Synria AI Robotic Arm

Primary manipulation hardware (binds to **Jetson AGX Thor**):

- 6 degrees of freedom
- 650 mm reach
- 1 kg payload
- LeRobot / ALOHA-compatible learning workflows
- ROS 1 / ROS 2 support
- MoveIt / MoveIt 2 support
- Synria C10 wrist camera for eye-in-hand perception
- end-goal demos: chess play and Ludo play, trained in Isaac Sim 5 / Isaac Sim 5.1 on the RTX 5090 workstation

Training scenes — arm mounted on the same 1.2 × 0.8 × 0.75 m table with the game pieces staged within reach. Shared floor / table / arm / lighting / camera scaffolding lives in `isaac/scripts/_synria_scene_common.py`; each game owns its board and pieces.

| Game | Scene | Builder | Output USD |
|---|---|---|---|
| Ludo | [scene README](../isaac/usd/scenes/synria_ludo/README.md) | [`build_synria_ludo_scene.py`](../isaac/scripts/build_synria_ludo_scene.py) | `isaac/usd/scenes/synria_ludo/synria_ludo_v0.usda` |
| Chess | [scene README](../isaac/usd/scenes/synria_chess/README.md) | [`build_synria_chess_scene.py`](../isaac/scripts/build_synria_chess_scene.py) | `isaac/usd/scenes/synria_chess/synria_chess_v0.usda` |
| Checkers | [scene README](../isaac/usd/scenes/synria_checkers/README.md) | [`build_synria_checkers_scene.py`](../isaac/scripts/build_synria_checkers_scene.py) | `isaac/usd/scenes/synria_checkers/synria_checkers_v0.usda` |

All three scenes are built on the RTX workstation via `bash scripts/linux_rtx/build_synria_<game>_scene.sh`.

Initial integration goals:

1. Verify vendor SDK, ROS 2 packages, URDF, meshes, and MoveIt configuration.
2. Bring up the C10 camera on Windows and Linux as a plain USB camera.
3. Bring up the C10 camera in ROS 2 on Linux.
4. Validate mock MoveIt planning before enabling real hardware motion.
5. Record safe joint limits, payload limits, workspace bounds, and emergency-stop procedure.
6. Build an OpenUSD/Isaac Sim representation for sim-first task development.

## Yahboom ROSMASTER M3 Pro

Primary mobile manipulation hardware (binds to **Jetson Orin NX 8GB**):

- Jetson Orin NX 8GB compute target (ships with the platform)
- ROS 2 robot platform
- mecanum wheel omnidirectional chassis
- onboard 6DOF arm configuration
- display-equipped version
- multimodal AI large model support
- AI voice interaction
- vision recognition

Initial integration goals:

1. Confirm exact model line, firmware image, ROS 2 distribution, and Yahboom tutorial package.
2. Record JetPack, Ubuntu, ROS 2, CUDA, and TensorRT versions from the robot.
3. Validate base teleoperation before autonomous navigation.
4. Validate camera topics, display, microphone, speaker, and arm topics.
5. Bring up SLAM in a controlled indoor test area.
6. Add mecanum-specific odometry, calibration, and drift notes.
7. Build an OpenUSD/Isaac digital twin of the mobile base and arm.
8. Benchmark on-device vision and voice interaction workloads.

## Jetson Setup Checklist

1. Install the supported JetPack version for the target board.
2. Verify CUDA, TensorRT, and Python runtime availability.
3. Confirm camera access through CSI, USB, or GStreamer.
4. Install ROS 2 distribution compatible with the Jetson OS image.
5. Configure a dedicated ROS domain ID for lab experiments.
6. Run thermal, power, and inference smoke tests before robot integration.

## Real Robot Bring-Up Checklist

1. Validate manual control with wheels or arm motors disabled where possible.
2. Confirm sensor timestamps and coordinate frames.
3. Record baseline telemetry at idle and under load.
4. Run simulation-only tests before real-world autonomous actions.
5. Enable emergency stop and operator override.

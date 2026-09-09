# Roadmap

Updated 2026-09-07. Completed code or device probes do not establish autonomous
physical manipulation. Current evidence and exclusions: [README](../README.md).

## Milestone 1: Software Foundation

- Cross-platform Python package
- CLI smoke demos
- Windows/Linux CI
- architecture and hardware docs
- synthetic robot telemetry generator

## Milestone 2: RoboCar SLAM

- ROS 2 workspace bootstrap ✅
- sensor calibration notes ✅ (`slam/calibration.py`, `physical-ai-lab slam-calibration`)
- map generation workflow (hardware-later)
- navigation launch files (hardware-later)
- Jetson SLAM benchmark report (hardware-later)
- Yahboom Orin NX manual base/arm and ROS/ZMQ recording-path bring-up recorded
  (`docs/notes/yahboom_strategy_2026-08-20.md`); sustained safety validation pending
- mecanum odometry calibration and drift report ✅ (`physical-ai-lab mecanum-calibration`)
- multimodal voice/vision interaction demo (hardware-later)
- mobile manipulation task using the onboard 6DOF arm (hardware-later)

## Milestone 3: Robotic Arm Sim-to-Real

- First-party OpenUSD structure retained; vendor robot geometry now fetched locally
  ([vendor setup](VENDOR_INTEGRATION_MAP.md))
- Isaac Lab reaching environment ✅ (`isaac/isaaclab_tasks/synria_pickplace/`)
- domain randomization plan ✅ (env config in `env_cfg.py`)
- real-arm calibration workflow (hardware-later)
- Simulation training/evaluation reports and Thor inference benchmark recorded;
  autonomous physical-arm policy deployment remains unestablished
- Synria 6DOF arm ROS 2 / MoveIt 2 bring-up ✅
  (`ros2_ws/src/synria_arm_description/`, `synria_arm_moveit_config/`, `synria_arm_gazebo/`)
- Synria C10 wrist-camera calibration and eye-in-hand perception ✅ (simulation contract
  and pipeline defined; real calibration hardware-later)
- LeRobot / ALOHA-style demonstration dataset workflow ✅
  (`lerobot/schema.py`, `dataset.py`, `recorder.py`, `policy_eval.py`)
- Synria upstream parity TODOs ✅ (all 8 RTX-Now items complete — see
  `docs/SYNRIA_UPSTREAM_TODO.md`)
- vendor robot-description audit ✅ (`docs/reports/synria_vendor_description_parity.md`)
- ROS 2 Humble-to-Jazzy compatibility pass ✅ (`docs/reports/synria_ros2_jazzy_port.md`)
- simulated cube sorting and hand-eye calibration ✅
  (`lerobot/cube_sort.py`, `reports/synria/hand_eye_calibration_sim.md`,
  `reports/demo/synria_cube_sort_sim.md`)

## Milestone 4: OpenUSD Digital Twin

- simulated cameras and markers ✅ (USD scene structure, C10 camera xacro)
- synthetic dataset generation ✅ (`lerobot/dataset.py`, GR00T dataset adapter)
- validation scenes for navigation and manipulation ✅ (Isaac USD scene builders)
- Yahboom mobile base and 6DOF arm digital twin (RTX-gated — USD re-import from vendor
  URDF needs Isaac Sim)

## Milestone 5: Edge Vision Runtime

- ONNX model-loading/inference paths (`edge_ai/onnx_runner.py`, explicit mock + real path);
  this module does not export ONNX models
- TensorRT deployment notes ✅ (`docs/JETSON_DEPLOYMENT.md`, `edge_ai/README.md`)
- live camera inference demo ✅ (`edge_ai/camera_inference.py`, `physical-ai-lab edge-ai-benchmark`)
- Separate Thor GR00T and Orin matmul/camera probes recorded on 2026-08-20;
  these are different workloads, not an Orin-versus-Thor comparison

## Milestone 6: Robot Operations Copilot ✅

- document ingestion ✅ (telemetry → triage pipeline)
- telemetry summarization ✅ (`physical_ai_lab/ops_copilot.py`, sliding window replay)
- safety-bounded recommendations ✅ (`AnthropicClassifier` + `DeterministicClassifier`,
  `escalate_to_human` flag on every report)
- operator CLI/API ✅ (`physical-ai-lab ops-copilot`, `ops-copilot-health`, `--replay` mode)
- ROS 2 node ✅ (`ros2_ws/src/physical_ai_ops_copilot/`)
- GR00T fine-tune scaffold ✅ (`isaac/isaaclab_tasks/synria_pickplace/gr00t/`)

## Next Up

- TensorRT/PyTorch numerical parity with recorded tolerances and deltas
- Sustained Jetson thermal/power and safety validation
- Yahboom SLAM/navigation evidence (Yahboom Orin NX)
- Real Synria arm bring-up (physical arm powered)
- LeRobot real demonstration recordings (arm + C10 live)

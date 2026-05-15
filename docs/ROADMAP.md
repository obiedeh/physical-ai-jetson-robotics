# Roadmap

## Milestone 1: Production Foundation

- Cross-platform Python package
- CLI smoke demos
- Windows/Linux CI
- architecture and hardware docs
- synthetic robot telemetry generator

## Milestone 2: RoboCar SLAM

- ROS 2 workspace bootstrap
- sensor calibration notes
- map generation workflow
- navigation launch files
- Jetson SLAM benchmark report
- Yahboom Orin NX mecanum base bring-up
- mecanum odometry calibration and drift report
- multimodal voice/vision interaction demo
- mobile manipulation task using the onboard 6DOF arm

## Milestone 3: Robotic Arm Sim-to-Real

- OpenUSD robot/workcell asset structure
- Isaac Lab reaching environment
- domain randomization plan
- real-arm calibration workflow
- policy deployment report
- Synria 6DOF arm ROS 2 / MoveIt 2 bring-up
- Synria C10 wrist-camera calibration and eye-in-hand perception
- LeRobot / ALOHA-style demonstration dataset workflow
- Synria upstream parity TODOs from `docs/SYNRIA_UPSTREAM_TODO.md`
- vendor robot-description audit before replacing the approximate arm model
- ROS 2 Humble-to-Jazzy compatibility pass for Synria-style packages
- simulated cube sorting and hand-eye calibration before real robot motion

## Milestone 4: OpenUSD Digital Twin

- factory-cell USD scene
- simulated cameras and markers
- synthetic dataset generation
- validation scenes for navigation and manipulation
- Yahboom mobile base and 6DOF arm digital twin

## Milestone 5: Edge Vision Runtime

- ONNX export path
- TensorRT deployment notes
- live camera inference demo
- Jetson Orin vs Thor benchmark table

## Milestone 6: Robot Operations Copilot

- document ingestion
- telemetry summarization
- safety-bounded recommendations
- operator CLI/API

## Milestone 7: AI-RAN Robotics Readiness

- private 5G telemetry simulator
- latency/congestion forecasting
- edge workload placement recommendations
- smart factory business-case report

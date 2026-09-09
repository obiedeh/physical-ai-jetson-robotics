# Environment Strategy

This lab is designed to move cleanly across three execution environments.

## Windows Development Machine

Primary role:

- portfolio authoring
- Python package development
- tests and CI parity
- agent/RAG workflows
- telemetry demos
- GitHub publishing

Use Windows for the parts of the project that should be easy for recruiters and collaborators to clone, inspect, and run without robotics hardware.

## Linux RTX Workstation

Primary role:

- OpenUSD scene creation
- Isaac Sim rendering and robotics simulation
- Isaac Lab robot training
- synthetic data generation
- ROS 2 simulation
- CUDA/RTX validation

Use the Linux RTX machine for the visually impressive and technically deep simulation work. This is where the OpenUSD digital twin, robotic arm training, and simulated robo car environments should mature.

## Jetson Thor / Orin

Primary role:

- real robot deployment
- edge AI inference
- camera streams
- SLAM/navigation runtime
- robotic arm control
- TensorRT and JetPack validation
- power, latency, thermal, and FPS benchmarking

Use Jetson hardware for proof that the system leaves simulation and runs on real edge robotics compute.

## Repo Rule

Keep the core Python package cross-platform. Put hardware-specific behavior behind scripts, docs, launch files, and optional extras so Windows development never blocks Linux/Jetson progress.

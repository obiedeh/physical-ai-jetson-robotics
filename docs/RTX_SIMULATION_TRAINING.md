# RTX 5090 Simulation and Training

This track runs high-impact simulation and training work on the Linux RTX 5090
workstation before Jetson Orin/Thor or real robot hardware is needed.

## Project List

```bash
physical-ai-lab rtx-projects
physical-ai-lab rtx-projects synria-reach-policy --json
physical-ai-lab synria-upstream-todos --phase rtx-now
```

The Synria upstream TODO list keeps vendor-alignment work explicit while we stay
on RTX simulation: description parity, Jazzy porting, mock `ros2_control`, C10
camera contracts, hand-eye calibration simulation, cube sorting, LeRobot schema,
and RoboCore-style kinematics benchmarks.

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

Render the factory-cell scene:

```bash
bash scripts/linux_rtx/render_factory_cell.sh
```

Use the Blender fallback renderer:

```bash
bash scripts/linux_rtx/render_factory_cell_blender.sh
```

Run ROS/Gazebo factory-cell smoke evidence:

```bash
bash scripts/linux_rtx/run_ros_gazebo_factory_cell.sh
```

Run Synria MoveIt smoke evidence:

```bash
bash scripts/linux_rtx/run_synria_moveit_smoke.sh
```

## One-Command RTX Suite

```bash
bash scripts/linux_rtx/run_rtx_training_suite.sh
```

This lists the RTX projects, trains the Synria synthetic reaching policy, and
captures a Blender factory-cell render.

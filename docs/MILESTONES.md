# Milestones

This file tracks the visible engineering milestones for the Physical AI Jetson Robotics Lab. Each milestone should leave behind code, repeatable commands, and evidence in `reports/`.

## M1: Repository Foundation

Status: working scaffold

- Python package and CLI are present.
- Cross-platform setup docs exist.
- Hardware inventory and report folders are in place.
- Initial tests and smoke commands are available.

Evidence:

- `physical_ai_lab/`
- `tests/`
- `reports/inventory/linux_rtx.json`

## M2: ROS 2 Robot Descriptions

Status: in progress

- Synria arm ROS 2 description and MoveIt scaffold are present.
- Yahboom ROSMASTER M3 Pro description package is being tuned.
- Generated Yahboom URDF parses with `check_urdf`.
- Canonical editable xacros live under `ros2_ws/src/*_description/urdf/`.

Evidence:

- `ros2_ws/src/synria_arm_description/`
- `ros2_ws/src/rosmaster_m3pro_description/`
- `docs/rosmaster_m3pro/MILESTONE_1_DESCRIPTION.md`

## M3: Isaac Sim / OpenUSD Assets

Status: working scaffold

- Synria Isaac USD asset folder is present.
- Yahboom generated Isaac URDF is present.
- Robot asset folders now share the same structure.
- Factory-cell USD scene exists as the first digital-twin scene.

Evidence:

- `isaac/usd/robots/synria_6dof_arm/`
- `isaac/usd/robots/yahboom-rosmaster-m3-pro/`
- `isaac/usd/factory_cell_v0.usda`
- `reports/isaac/2026-05-14-linux-rtx-openusd-bringup.md`

## M4: Simulation Smoke Tests

Status: working scaffold

- MoveIt smoke-test script exists.
- ROS/Gazebo factory-cell script exists.
- Blender render script exists for visual evidence.

Evidence:

- `scripts/linux_rtx/run_synria_moveit_smoke.sh`
- `scripts/linux_rtx/run_ros_gazebo_factory_cell.sh`
- `scripts/linux_rtx/render_factory_cell_blender.sh`
- `reports/simulation/`

## M5: Robot Learning Track

Status: working scaffold

- Synria reaching policy training command is available.
- Training report output path exists.
- LeRobot / ALOHA workflow docs are present.

Evidence:

- `docs/LEROBOT_ALOHA.md`
- `scripts/linux_rtx/run_rtx_training_suite.sh`
- `reports/training/synria_reach_policy.json`

## M6: Jetson Hardware Bring-Up

Status: planned / partial docs

- Jetson inventory scripts are present.
- Yahboom and Synria bring-up report templates exist.
- Real hardware evidence still needs to be collected and committed.

Evidence targets:

- `reports/yahboom/`
- `reports/synria/`
- `reports/inventory/`

## M7: Edge AI and Operations Copilot

Status: planned / scaffold

- Edge AI module and operations-copilot module are present.
- Next step is to connect real telemetry/logs and produce operator recommendations with safety gates.

Evidence targets:

- `edge_ai/`
- `physical_ai_lab/edge_ai.py`
- `physical_ai_lab/ops_copilot.py`
- `agents/`

## Next Public-Facing Push

- Add screenshots or renders under `docs/assets/`.
- Add a short GIF or image from Isaac Sim / Blender to the root README.
- Open focused GitHub issues for Yahboom URDF tuning, Isaac import validation, and Jetson bring-up evidence.
- Keep generated robot files documented and separate from editable ROS 2 xacro source.

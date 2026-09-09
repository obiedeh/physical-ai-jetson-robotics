# Simulation Bring-Up Report

Date:

Operator:

Host:

Repo commit:

## Scope

- Simulation target:
- Robot descriptions included:
- OpenUSD assets included:
- ROS 2 packages included:
- Explicitly excluded hardware:

## Environment

- OS:
- GPU:
- NVIDIA driver:
- CUDA:
- ROS 2 distro:
- Gazebo:
- MoveIt 2:
- Isaac Sim:
- Blender:

## Static Validation

Commands:

```bash
pytest -q tests/test_simulation_foundation.py
bash scripts/linux_rtx/smoke_robot_descriptions.sh
```

Results:

- URDF/Xacro XML valid:
- Xacro includes resolved:
- Mesh references resolved:
- Launch/config files present:
- Robot structure checks passed:

Evidence:

- Log path:
- Failure summary:

## ROS 2 Build

Commands:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Results:

- Packages built:
- Warnings:
- Failed packages:
- Log path:

## Gazebo Smoke Test

Commands:

```bash
bash scripts/linux_rtx/smoke_ros2_simulation.sh
```

Results:

- Launch started:
- Timeout exit expected:
- Gazebo logs:
- Screenshot or screen recording:

## Synria Mock Control / MoveIt

Commands:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
timeout 8s ros2 launch synria_arm_gazebo control.launch.py
timeout 8s ros2 launch synria_arm_moveit_config demo.launch.py
```

Results:

- Mock hardware loaded:
- Controllers configured:
- MoveIt planning scene loaded:
- Known issues:

## Readiness Decision

- Simulation evidence sufficient for next step:
- Hardware still blocked by:
- Required fixes before powered hardware:

## Next Actions

- [ ]
- [ ]
- [ ]

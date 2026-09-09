# ROS 2 Workspace

ROS 2 Jazzy packages for the Synria 6DOF arm and Yahboom ROSMASTER M3 Pro
mobile manipulator. First-party static checks run without vendor geometry.
Robot-dependent builds/launches require [local vendor setup](../docs/VENDOR_INTEGRATION_MAP.md).
The Yahboom board runtime separately uses ROS 2 Humble on the Orin.

## Packages

### Yahboom ROSMASTER M3 Pro

| Package | Status | Purpose |
|---|---|---|
| `yahboom_M3Pro_description` | External, fetch locally | URDF + STL meshes (SolidWorks export) |
| `rosmaster_m3pro_description` | ✅ Implemented | xacro wrapper, world link, ros2_control mock, display launch, RViz config |
| `rosmaster_m3pro_bringup` | ✅ Launch | Top-level bringup (includes description) |
| `M3Pro_config` | External, fetch locally | MoveIt Setup Assistant config |
| `rosmaster_m3pro_control` | 🗂 Config | ros2_control fake + real controller configs |
| `rosmaster_m3pro_moveit_config` | 🗂 Config | MoveIt 2 planning config (planned) |
| `rosmaster_m3pro_hardware` | 🗂 Config | Hardware interface config (planned) |
| `rosmaster_m3pro_simulation` | 🗂 Config | Gazebo world (planned) |
| `rosmaster_m3pro_perception` | 🗂 Config | Sensor frame config (planned) |

### Synria 6DOF Arm

| Package | Status | Purpose |
|---|---|---|
| `synria_arm_description` | ✅ Scaffold | xacro model, C10 wrist camera, display launch |
| `synria_arm_gazebo` | ✅ Scaffold | ros2_control + joint_trajectory_controller |
| `synria_arm_moveit_config` | ✅ Scaffold | SRDF, OMPL planning, kinematics |

Build from a sourced ROS 2 environment:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

ROS 2 Jazzy desktop and ROS-Gazebo integration have been installed on the Linux
RTX workstation. New shells need `source /opt/ros/jazzy/setup.bash` before `ros2`
or `gz` are on `PATH`.

## Yahboom ROSMASTER M3 Pro

Display the M3 Pro in RViz (arm sliders via joint_state_publisher_gui):

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch rosmaster_m3pro_description display.launch.py
```

Or via the top-level bringup package:

```bash
ros2 launch rosmaster_m3pro_bringup display.launch.py
```

---

## Synria 6DOF Arm

The first-party Synria xacro wraps the vendor Alicia-D URDF with Joint1..Joint6
and gripper joints. That URDF is now a locally obtained input, not a tracked file.
The older approximate model remains only in the Isaac source archive.

Installed arm simulation dependencies:

- MoveIt 2
- ROS 2 control
- ROS 2 controllers
- joint state publisher GUI

Display the arm model:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch synria_arm_description display.launch.py
```

Start mock ROS 2 control:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch synria_arm_gazebo control.launch.py
```

Start the MoveIt 2 demo:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch synria_arm_moveit_config demo.launch.py
```

Historical mock-validation status (requires the local vendor inputs; not rerun
as a complete ROS launch after the 2026-09-07 removal):

- The recorded workspace build passed before vendor removal
- `synria_arm_gazebo control.launch.py` starts mock hardware and activates the arm trajectory controller
- `synria_arm_moveit_config demo.launch.py` loads the robot model, KDL kinematics, and OMPL planning pipeline

# Synria ROS 2 Humble → Jazzy Porting Checklist

**Date:** 2026-05-28
**Target:** Ubuntu 24.04 + ROS 2 Jazzy (this workstation)
**Source:** Synria/Alicia-D upstream packages written for ROS 2 Humble

---

## Motivation

The upstream `Synria-Robotics/Alicia-D-ROS2` packages target ROS 2 Humble
(Ubuntu 22.04). This workstation runs ROS 2 Jazzy (Ubuntu 24.04). This
checklist tracks what changes are required before running Synria ROS 2
packages natively in simulation or on real hardware.

---

## Package Manifest Changes

| Item | Humble | Jazzy | Status |
|---|---|---|---|
| `<buildtool_depend>ament_cmake</buildtool_depend>` | ✅ same | ✅ same | ✅ |
| `<depend>rclpy</depend>` | ✅ | ✅ | ✅ |
| `<depend>std_msgs</depend>` | ✅ | ✅ | ✅ |
| `<depend>sensor_msgs</depend>` | ✅ | ✅ | ✅ |
| `<depend>geometry_msgs</depend>` | ✅ | ✅ | ✅ |
| `<depend>moveit_ros_planning_interface</depend>` | `moveit2` Humble | Jazzy MoveIt 2 | 🔲 Verify ABI |
| `<depend>ros2_control</depend>` | Humble API | Jazzy API | 🔲 Verify YAML schema |
| `format="3"` package.xml | optional | recommended | ✅ using format 3 |

---

## CMakeLists.txt / setup.py

| Item | Status |
|---|---|
| `cmake_minimum_required(VERSION 3.8)` | ✅ no change needed |
| `ament_python_install_package` for Python packages | ✅ unchanged |
| `install(DIRECTORY launch urdf meshes ...)` | ✅ unchanged |
| `ament_auto_find_build_dependencies` | ✅ if used, unchanged |

---

## Launch File API

| Item | Humble | Jazzy | Status |
|---|---|---|---|
| `LaunchDescription` import | `launch` | `launch` | ✅ same |
| `Node` launch action | `launch_ros.actions.Node` | same | ✅ same |
| `DeclareLaunchArgument` | same | same | ✅ same |
| `ExecuteProcess` | same | same | ✅ same |
| `robot_state_publisher` parameters | `robot_description` string | same | ✅ |
| `use_sim_time` parameter type | `bool` | `bool` | ✅ same |

---

## Controller YAML Schema

| Item | Status |
|---|---|
| `controller_manager` top-level key | ✅ unchanged in Jazzy |
| `ros__parameters` nesting | ✅ unchanged |
| `joint_trajectory_controller/JointTrajectoryController` | ✅ same plugin |
| `gripper_action_controller/GripperActionController` | ✅ same plugin |
| `update_rate: 100` | ✅ unchanged |

---

## Python API Changes (rclpy)

| Item | Humble | Jazzy | Status |
|---|---|---|---|
| `node.get_logger().warn(...)` | ✅ | deprecated → `warning` | 🔲 Update call sites |
| `rclpy.spin_once` | ✅ | ✅ | ✅ same |
| `Node.create_subscription` QoS | unchanged | unchanged | ✅ |
| `std_msgs.msg.String` | ✅ | ✅ | ✅ |

**Note:** `get_logger().warn()` is deprecated in Jazzy; replace with
`get_logger().warning()` in all `ops_copilot_node.py` and Synria driver files.

---

## MoveIt 2 API

| Item | Status |
|---|---|
| `moveit_simple_controller_manager` YAML key | ✅ present — see `moveit_controllers.yaml` |
| `FollowJointTrajectory` action interface | ✅ unchanged |
| `move_group` node startup | 🔲 Test with `demo.launch.py` after sourcing workspace |
| SRDF chain link names | ✅ match vendor URDF |

---

## Known Incompatibilities

1. **`get_logger().warn()` → `.warning()`** — affects `ops_copilot_node.py` and
   any ported Synria driver files. Low-risk: runtime warning only, not a crash.

2. **`ament_cmake` vs `ament_python`** — upstream description package uses
   `ament_cmake`; our `physical_ai_ops_copilot` uses `ament_python`. Both work in
   Jazzy. No cross-dependency.

3. **Gazebo Classic → Gazebo Harmonic** — Humble used Gazebo Classic (11);
   Jazzy uses Gazebo Harmonic. Any upstream `gazebo_ros` plugins require a
   porting step to `gz_ros2_control`. **Evidence-gated**: deferred until
   Gazebo simulation is needed.

---

## Action Items

| Item | Priority | Status |
|---|---|---|
| Replace `.warn()` with `.warning()` in node files | Low | 🔲 Pre-hardware |
| Verify `moveit_ros_planning_interface` ABI on Jazzy | Medium | 🔲 Pre-bringup |
| Port Gazebo Classic plugins to Harmonic | Low | 🔲 Evidence-gated |
| Source workspace and run `demo.launch.py` smoke test | High | 🔲 Hardware-later |

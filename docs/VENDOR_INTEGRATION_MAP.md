# Vendor Integration Map

Reviewed: 2026-05-15

This map records how upstream vendor material feeds the Physical AI platform without
turning this repository into a vendor mirror. Upstream code and CAD are treated as
reference inputs. Repo-owned descriptions, simulations, tests, and evidence reports are
the integration surface.

## Source Summary

| Vendor / platform | Upstream source | What it provides | Platform use |
| --- | --- | --- | --- |
| Yahboom ROSMASTER-M3PRO | Yahboom product and course pages: `https://category.yahboom.net/collections/ros-series/products/rosmaster-m3-pro`, `https://www.yahboom.com/study/ROSMASTER-M3PRO` | ROS 2 Humble-oriented mobile manipulator course material, mecanum base, 6-DOF arm, depth camera, dual lidar, MoveIt2, SLAM, Navigation2, and AI demos | Model and simulate a mobile manipulation target under `ros2_ws/src/rosmaster_m3pro_description`, stage robot assets under `isaac/usd/robots/yahboom-rosmaster-m3-pro`, record bring-up in `reports/yahboom` |
| Synria Alicia-D ROS 2 | Synria docs: `https://docs.sparklingrobo.com/docs/alicia-d-series/follower/doc_04_ROS2_Humble`; ROS 2 clone target documented as `https://github.com/Synria-Robotics/Alicia-D-ROS2.git -b v6.1.0rc1` | ROS 2 Humble driver/control package for Alicia-D six-axis arm with gripper, ros2_control, MoveIt2, serial communication, calibration, and state feedback | Keep hardware driver out of the default path; model the arm in `ros2_ws/src/synria_arm_description`, validate mock control in `ros2_ws/src/synria_arm_gazebo`, plan with `ros2_ws/src/synria_arm_moveit_config`, record evidence in `reports/synria` |
| Synria Alicia-D SDK | `https://github.com/Synria-Robotics/Alicia-D-SDK`; PyPI `alicia-d-sdk` | Python SDK for Alicia-D control and state access | Future optional hardware adapter only after simulation and operator safety gates |
| Synria robot descriptions | PyPI `synriard`, project page `https://pypi.org/project/synriard/` | URDF, MJCF, and meshes for Synria platforms including Alicia-D variants | Candidate source for replacing the approximate repo-owned Synria URDF once license, version, and geometry are reviewed |
| Synria Alicia-D Leader ROS | `https://github.com/Synria-Robotics/Alicia-D-Leader-ROS` | ROS 1/catkin leader-arm serial nodes and topics for teach/teleop workflows | Reference only. Do not import into the ROS 2 workspace unless a leader-arm track is explicitly added |

## Platform Mapping

| Platform area | Yahboom ROSMASTER-M3PRO | Synria Alicia-D |
| --- | --- | --- |
| Robot description | `ros2_ws/src/rosmaster_m3pro_description` owns the current mecanum base, arm, sensor, control, mesh, RViz, and display-launch model | `ros2_ws/src/synria_arm_description` owns the approximate six-axis arm and C10 camera description |
| Simulation control | Planned Gazebo/mock-control package should use fake or mock hardware first | `ros2_ws/src/synria_arm_gazebo` uses `mock_components/GenericSystem` through ros2_control |
| Planning | Future MoveIt config should be derived from the repo-owned description and validated against vendor limits | `ros2_ws/src/synria_arm_moveit_config` provides starter SRDF, kinematics, limits, OMPL, and demo launch |
| OpenUSD / Isaac | `isaac/usd/robots/yahboom-rosmaster-m3-pro` stages CAD and mesh extraction assets; `isaac/usd/factory_cell_v0.usda` is the current scene anchor | Future Alicia-D USD assets should live under `isaac/usd/robots/synria-alicia-d` and stay linked to source version metadata |
| Evidence | `reports/yahboom/BRINGUP_TEMPLATE.md` captures sim, inventory, ROS topics, teleop, calibration, navigation, and mobile manipulation evidence | `reports/synria/BRINGUP_TEMPLATE.md` captures sim, MoveIt/mock control, camera, calibration, demonstration, and later hardware evidence |
| Tests | Static tests validate Xacro includes, mesh references, key links/joints/sensors, launch files, and asset presence | Static tests validate the arm model, camera mount, mock ros2_control structure, MoveIt config, launch files, and evidence templates |

## Integration Rules

1. Preserve vendor source URLs, versions, branches, and dates in docs or report evidence.
2. Copy only the files needed for a platform-owned validation step.
3. Prefer approximate repo-owned models for early simulation over hardware drivers that
   assume serial ports, motors, cameras, or powered robots.
4. Keep hardware drivers optional and gated by scripts or launch arguments that make the
   hardware dependency obvious.
5. Translate vendor naming into stable platform package names rather than importing
   upstream workspace layouts wholesale.
6. Track known approximations, missing CAD, missing limits, and unverified transforms in
   reports before claiming hardware readiness.

## Current Gaps

- Yahboom hardware driver packages are not part of the default repo path.
- Yahboom MoveIt and navigation configs still need repo-owned simulation validation.
- Synria URDF is approximate until `synriard` or vendor CAD is reviewed and imported.
- Synria hardware serial control is intentionally absent from default launch paths.
- OpenUSD robot assets need metadata files that record source version, scale, units, and
  conversion commands.
- Simulation smoke scripts are bounded developer checks, not proof of physical robot
  readiness.

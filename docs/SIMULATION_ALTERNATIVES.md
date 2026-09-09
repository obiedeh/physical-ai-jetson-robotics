# Simulation Alternatives

When Isaac Sim RTX rendering is parked, OpenUSD stays as the source of truth and these tools cover screenshots, ROS simulation, and focused control experiments until Isaac Sim is stable on the host workstation.

## Recommended Path

1. Author robot and scene assets in OpenUSD.
2. Use Blender as the first alternate screenshot and visual-review path.
3. Use ROS 2 + Gazebo Harmonic for robot motion, sensors, and Nav2-style workflows.
4. Use MuJoCo only for focused control experiments where its USD import coverage is enough.
5. Return to Isaac Sim / Isaac Lab when the RTX renderer issue is fixed without rolling back the NVIDIA driver branch.

## Option A: OpenUSD-First Validation

Use OpenUSD tooling to validate the scene structure without requiring Isaac RTX rendering.

Best for:

- checking that the stage opens
- verifying prim paths, cameras, lights, units, and references
- keeping the digital twin format portable

Limits:

- not a full robot simulator by itself
- screenshot quality depends on the available viewer or renderer

## Option B: Blender Screenshot Path

Blender supports USD import and export in builds compiled with USD support. The Ubuntu 24.04 package currently installed on this workstation does not expose the USD import operator, so fallback scripts may need to reconstruct the scene directly in Blender for screenshots.

Best for:

- producing screenshots without Isaac Sim
- visual inspection of layout, cameras, tables, bins, zones, and proxy robots
- converting or cleaning visual assets before adding them to the USD track

Limits:

- Blender USD support depends on the installed Blender build
- physics, sensors, and Isaac-specific schemas will not carry over as runnable simulation behavior
- materials and advanced USD composition may need cleanup

Reference:

- Blender USD manual: https://docs.blender.org/manual/en/latest/files/import_export/usd.html

## Option C: ROS 2 + Gazebo Harmonic

Gazebo Harmonic is the strongest alternate robotics simulation path for this project because the repo already targets ROS 2, mobile-base workflows, sensors, SLAM, Nav2, and Jetson deployment.

Best for:

- mobile-base simulation
- camera, lidar, IMU, and joint-state workflows
- ROS 2 topic integration through `ros_gz`
- Nav2 and teleoperation demos
- sim-first checks before touching real Yahboom hardware

Limits:

- Gazebo uses SDF/URDF workflows rather than USD as the native source format
- visual quality and synthetic-data tooling are not the same as Isaac Sim
- OpenUSD scenes need a parallel Gazebo world or conversion layer

Status:

- ROS 2 Jazzy desktop and `ros-jazzy-ros-gz` installed
- MoveIt 2, ROS 2 control, ROS 2 controllers, and joint state publisher GUI installed
- `ros2_ws` builds with `colcon build --symlink-install`
- approximate Synria 6DOF arm packages build and launch
- Synria mock ROS 2 control loads `mock_components/GenericSystem`, activates `joint_state_broadcaster`, and activates `arm_controller`
- Synria MoveIt 2 demo loads KDL kinematics and OMPL; `move_group` reaches "You can start planning now"
- new shells need `source /opt/ros/jazzy/setup.bash` before using ROS commands

References:

- Gazebo ROS 2 overview: https://gazebosim.org/docs/harmonic/ros2_overview/
- Gazebo ROS 2 bridge: https://gazebosim.org/docs/harmonic/ros2_integration/

## Option D: MuJoCo Control Sandbox

MuJoCo can load USD assets, but its OpenUSD support is currently marked experimental. Treat it as a focused physics/control sandbox rather than the main digital twin.

Best for:

- arm or gripper control experiments
- small scenes with simple assets
- fast local iteration on dynamics

Limits:

- USD support is experimental
- not the best fit for ROS-first mobile manipulation demos
- not a replacement for Gazebo or Isaac when sensor/ROS integration is the target

Reference:

- MuJoCo USD import docs: https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html

## Decision

For this repo, use this split:

- OpenUSD remains the digital-twin source format.
- Blender handles immediate visual screenshots.
- Gazebo Harmonic handles ROS 2 robotics simulation.
- MuJoCo stays optional for narrow control tests.
- Isaac Sim remains parked until the renderer crash can be fixed without changing the chosen NVIDIA driver strategy.

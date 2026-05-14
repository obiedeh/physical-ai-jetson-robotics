# ROS 2 Workspace

This folder is reserved for ROS 2 packages and launch files.

Current packages:

- `physical_ai_description`: placeholder robot descriptions for the mobile manipulator track
- `physical_ai_gazebo`: Gazebo Harmonic factory-cell world and launch scaffold
- `synria_arm_description`: approximate 6DOF Synria arm with C10 camera mount
- `synria_arm_gazebo`: ROS 2 control scaffold for the Synria arm
- `synria_arm_moveit_config`: starter MoveIt 2 planning config for the Synria arm

Planned packages:

- robo car bring-up
- sensor calibration
- SLAM and navigation
- robotic arm control
- telemetry publisher
- safety gate

## Gazebo Harmonic Factory Cell

The initial Gazebo track mirrors the repo-local OpenUSD factory-cell layout enough to begin ROS 2 simulation work while Isaac Sim rendering is parked.

Build from a sourced ROS 2 environment:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch physical_ai_gazebo factory_cell.launch.py
```

ROS 2 Jazzy desktop and ROS-Gazebo integration have been installed on the Linux RTX workstation. New shells need `source /opt/ros/jazzy/setup.bash` before `ros2` or `gz` are on `PATH`.

## Synria 6DOF Arm

The Synria arm packages are approximate bringup scaffolds until vendor URDF/CAD and exact joint limits are added.

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

Validated status:

- `colcon build --symlink-install` builds all current workspace packages
- `synria_arm_gazebo control.launch.py` starts mock hardware and activates the arm trajectory controller
- `synria_arm_moveit_config demo.launch.py` loads the robot model, KDL kinematics, and OMPL planning pipeline

# lerobot_robot_rosmaster_m3pro

LeRobot (>= 0.6.2) plugin for the **Yahboom ROSMASTER M3 Pro** (Jetson Orin NX
8GB): 6-DOF bus-servo arm + mecanum base, driven through the vendor's
micro-ROS bridge (`/arm6_joints`, `/cmd_vel`; `/odom_raw`, `/battery`).
Installing the package is enough — LeRobot discovers `lerobot_robot_*`
distributions and the types register themselves:

    --robot.type=rosmaster_m3pro
    --teleop.type=rosmaster_m3pro_keyboard

## Facts this class is built on (measured 2026-08-20, see docs/notes)
- Arm command = absolute servo degrees x6 + `run_time` ms; the board
  interpolates. Servo 6 is the gripper. Vendor home: 90,150,12,20,90,0.
- The board publishes **no joint feedback**. `observation.state` for the arm
  is the last *commanded* pose (open loop, same as every vendor demo) — so the
  robot moves to a known pose on connect (`go_home_on_connect=True`).
- The base has **no cmd_vel watchdog**: motion persists until an explicit zero.
  This class runs a deadman thread (zeros after `deadman_timeout_s` without a
  fresh action) and always zeros on disconnect/exception.
- DDS: vendor agent runs as user `jetson` on ROS_DOMAIN_ID=30; another user
  needs the UDP-only FastDDS profile (`fastdds_profile`). Run from OUTSIDE the
  repo root (the repo-local `lerobot/` folder shadows the library).

## Run (on the robot)
    source /opt/ros/humble/setup.bash
    source /home/jetson/yahboomcar_ws/install/setup.bash      # arm_msgs
    cd /tmp && ~/.venv-lerobot/bin/m3pro-smoke --no-base      # connect, home, +5deg joint1
    lerobot-teleoperate --robot.type=rosmaster_m3pro --teleop.type=rosmaster_m3pro_keyboard \
        --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 15}}'

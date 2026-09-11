# Orin discovery record, 2026-09-10 (read-only)

Protocol step 1 of [SLAM_FIRST_RUN_PROTOCOL.md](SLAM_FIRST_RUN_PROTOCOL.md),
run over SSH as `jetson@yahboom` (192.168.1.251) after the operator restored
key access. Nothing on the robot was launched, moved or configured. The one
change made to the device is a clone of this repository at
`/home/jetson/github/physical-ai-jetson-robotics` (commit `7bc20fb`).

## Host

| Item | Value |
|---|---|
| Image | reflashed by the operator; L4T R36 REVISION 4.7 (GCID 42132812, 2025-09-18); the 2026-08-20 day-one run was on R36 4.4 |
| Power mode | `pmode:0000` (MAXN_SUPER per the earlier record; not re-verified with `nvpmodel -q`) |
| Memory | 7619 MB total, 5351 MB available with the micro-ROS agent running |
| Disk | 117 GB NVMe, 101 GB used, 11 GB free (91 %) |
| Accounts | `jetson` only; the `oedeh` account and its venv from August no longer exist |
| Wi-Fi | `wlP1p1s0`, power save **on** (reset by the reflash); 20/20 pings to the host at 192.168.1.235, 0 % loss, 6.9 ms mean |
| Sampling | `/usr/bin/tegrastats` present; INA3221 rails at `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1` |
| ROS tooling | `ros2bag`, `rosbag2_storage_default_plugins` (sqlite3), `tf2_tools`, `nav2_map_server` 1.1.20 |
| Home permissions | `/home/jetson` was mode 777 after the reflash, which made sshd reject `authorized_keys`; the operator set it to 755 |

## Board contract, verified live (no motion sent)

micro-ROS agent running on `/dev/myserial` (autostart), `ROS_DOMAIN_ID=30`.
Topics: `/arm6_joints /arm_joint /battery /beep /cmd_vel /imu/data_raw /odom_raw /rgb /scan0 /scan1`.

| Topic | Rate observed over 8 s |
|---|---|
| `/scan0` | 7.15 Hz |
| `/scan1` | 7.16 Hz |
| `/odom_raw` | 11.3 Hz |
| `/imu/data_raw` | 25.2 Hz |
| `/battery` | 12.54 V |

Matches the 2026-08-20 contract in `docs/notes/yahboom_strategy_2026-08-20.md`.

## Vendor SLAM stack (M3Pro_ws, built)

Installed: `slam_toolbox` 2.6.10, `cartographer_ros` 2.0.9002, `slam_gmapping`,
`nav2_amcl`, `ira_laser_tools`, `yahboom_laser_filter`, `ekf_bringup`
(`robot_localization`), `imu_filter_madgwick`.

Mapping launch to use for Run A: `ros2 launch slam_mapping slam_toolbox.launch.py`. It starts:

1. `ira_laser_tools laserscan_multi_merger` with `config/laserscan_merge.yaml`:
   inputs `/scan0 /scan1`, output `/scan_multi` in `base_link`, range 0.05 to 4.0 m,
   1 degree increment. This include also pulls `M3Pro display.launch.py`
   (`robot_state_publisher` from `M3Pro/urdf/M3Pro.urdf` plus a joint state publisher).
2. `yahboom_laser_filter laser_filter_node` (angle window -180 to 180). Its input and
   output topic names are not visible in the launch file; confirm at launch with
   `ros2 topic list`. SLAM Toolbox expects `/scan`.
3. `imu_filter_madgwick_node`.
4. `ekf_bringup ekf.launch.py`, remapping `/odometry/filtered` to `/odom`.
5. `slam_mapping online_async_launch.py` with `config/mapper_params_online_async.yaml`:
   `scan_topic: /scan`, `base_frame: base_footprint`, `odom_frame: odom`,
   `map_frame: map`, `mode: mapping`, resolution 0.05 m, `max_laser_range` 4.0 m,
   loop closing on, `minimum_travel_distance` 0.5 m, `minimum_travel_heading` 0.5 rad,
   `map_update_interval` 3.0 s, Ceres solver.
6. `rviz2` with the vendor config. Headless over SSH this node will fail; the rest
   of the launch continues. Prefer launching from the robot's own session or
   accept the rviz error.

Localisation for Run B: `slam_mapping/config/mapper_params_localization.yaml`
(`mode: localization`, `map_start_at_dock: true`, `map_file_name` empty, to be set
to the saved posegraph), and `M3Pro_navigation/launch/localization.launch.py`
(`nav2_amcl`, note it sets `use_sim_time: True`, which must be false on the robot).

Map saving: `slam_mapping/launch/save_slamtoolbox_map.launch.py` and
`save_map.launch.py`; `nav2_map_server map_saver_cli` is available as in the protocol.

## TF tree as declared by the vendor (not measured)

`m3pro_bringup/launch/static_tf.launch.py` publishes, all in metres and with zero rotation:

| Parent | Child | x | y | z |
|---|---|---|---|---|
| `base_footprint` | `base_link` | 0 | 0 | 0.02 |
| `base_link` | `laser0_frame` | -0.11617 | 0.09156 | 0.1253 |
| `base_link` | `laser1_frame` | 0.10766 | -0.09078 | 0.1253 |
| `base_link` | `imu_frame` | 0.06 | 0 | 0.08 |

These are the vendor's numbers, not a calibration. They remove the "mount
position unknown" blocker for a first run and leave "mount pose measured" as
planned. Whether `car_base.launch.py` or `static_tf.launch.py` is what runs at
boot, and whether the merged scan frame `base_link` is consistent with the URDF,
is confirmed by `ros2 run tf2_tools view_frames` once the stack is up.

## Existing artefacts on the device

`/home/jetson/yahboom_map.pbstream` (a cartographer map of an unknown room,
from the factory image). No slam_toolbox posegraph, no bags.

## Harness readiness

`python3 scripts/jetson/slam_session.py --help` runs on the Orin with the system
Python 3.10 and imports `physical_ai_lab.jetson_provenance` from the clone.
`slam_pose_log.py` needs `rclpy`, `nav_msgs` and `tf2_ros`, all present in
Humble; frames to pass are `--map-frame map --base-frame base_footprint`.

## Open before Run A

- Safety gates in protocol step 0 need the operator at the robot: e-stop test
  on the stand, joystick teleop only, room clear.
- Wi-Fi power save is on. Link quality was fine in this session; if it degrades,
  the August fix was `iw dev wlP1p1s0 set power_save off` (needs sudo, operator).
- The laser filter's output topic and the boot-time TF publisher are the two
  facts still to be read off the live system at launch.
- 11 GB free is enough for LiDAR bags at these rates (well under 100 MB per
  5 minutes without the camera).

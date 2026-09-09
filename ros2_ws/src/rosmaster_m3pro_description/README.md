# rosmaster_m3pro_description

ROS 2 description package for the **Yahboom ROSMASTER M3 Pro** mobile
manipulator. Wraps the vendor URDF with a `world` link, xacro composition,
and `ros2_control` mock-hardware setup for simulation and development
bringup.

---

## Robot overview

The ROSMASTER M3 Pro is a mecanum-wheel mobile base with an onboard 5-DOF
serial arm and parallel-jaw gripper. Paired hardware: **NVIDIA Jetson Orin NX
8 GB** (ships with the kit).

| Subsystem | Description |
|---|---|
| Mobile base | Mecanum drive — 4 independently driven wheels |
| Arm | 5 revolute joints (`arm1_Joint`…`arm5_Joint`), ±π/2 range |
| Gripper | `rlink1_Joint` (jaw open/close) + parallel linkage (`rlink2`, `llink1-3`) |
| Cameras | Front RGB (`Camera_Joint`, fixed on base) |
| Compute | Jetson Orin NX 8 GB |

---

## Package structure

```
rosmaster_m3pro_description/
├── CMakeLists.txt
├── package.xml
├── urdf/
│   └── rosmaster_m3pro.urdf.xacro   ← world link + vendor include + ros2_control
├── launch/
│   └── display.launch.py            ← RViz display launch
├── rviz/
│   └── rosmaster_m3pro.rviz         ← RViz config (base_link fixed frame)
└── README.md
```

The vendor URDF and STL meshes live in the sibling package
[`yahboom_M3Pro_description`](../yahboom_M3Pro_description/). This package
references them via `$(find yahboom_M3Pro_description)/…` and does not
duplicate them.

---

## Building

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select rosmaster_m3pro_description yahboom_M3Pro_description
source install/setup.bash
```

---

## Display in RViz

```bash
ros2 launch rosmaster_m3pro_description display.launch.py
```

Opens:
- `robot_state_publisher` — publishes `/robot_description` and the TF tree
- `joint_state_publisher_gui` — sliders for arm1..arm5 and rlink1 joints
- `rviz2` — RobotModel + TF displays, `base_link` as fixed frame

---

## Joint reference

| Joint | Type | Range | Notes |
|---|---|---|---|
| `arm1_Joint` | revolute | ±π/2 | Base rotation (vertical axis) |
| `arm2_Joint` | revolute | ±π/2 | Shoulder pitch |
| `arm3_Joint` | revolute | ±π/2 | Elbow pitch |
| `arm4_Joiint` | revolute | ±π/2 | Wrist pitch (vendor typo — double 'i', preserved) |
| `arm5_Joint` | revolute | ±π/2 | Wrist roll (vertical axis) |
| `rlink1_Joint` | revolute | [−0.95, 0] | Jaw open/close |
| `rlink2_Joint`, `llink1/2/3_Joint` | continuous | — | Parallel linkage (passive) |
| `lwheel1/2_Joint`, `rwheel1/2_Joint` | revolute | [0, 0] | Mecanum wheels (fixed in URDF; add diff-drive/mecanum controller for odometry) |
| `Camera_Joint` | revolute | [0, 0] | Front camera (fixed position) |

**Note on `arm4_Joiint`:** the vendor SolidWorks URDF export has a double-'i'
typo in this joint name. `M3Pro_config`, `joint_limits.yaml`, and this package
all preserve the typo for consistency. Do not correct it without updating all
downstream references.

---

## ros2_control

The `ros2_control` block in `rosmaster_m3pro.urdf.xacro` uses
`mock_components/GenericSystem` — the standard fake hardware plugin for
development and CI. It exposes `position` command + state interfaces for the
5 arm joints and `rlink1_Joint`.

For real-hardware operation, replace `mock_components/GenericSystem` with the
Yahboom SDK hardware interface or a custom `ros2_control` plugin.

---

## Related packages

| Package | Purpose |
|---|---|
| `yahboom_M3Pro_description` | Vendor URDF + STL meshes (ament_python) |
| `M3Pro_config` | MoveIt Setup Assistant config (vendor-generated) |
| `rosmaster_m3pro_bringup` | Top-level bringup launches (includes this package) |
| `rosmaster_m3pro_moveit_config` | MoveIt 2 planning config (planned) |
| `rosmaster_m3pro_control` | ros2_control fake + real controller configs |
| `rosmaster_m3pro_simulation` | Gazebo world (planned) |

---

## Remaining work (hardware-gated)

- **Verify joint names** against loaded Jetson firmware — `arm4_Joiint` may
  need renaming if the Yahboom SDK or hardware interface uses a different name
- **Add mecanum wheel joints** to ros2_control with a `MecanumDrive` controller
  for odometry-based Nav2 operation
- **Wire camera** — add `Camera` sensor frame to the TF tree with calibrated
  extrinsics once the Orin NX is commissioned
- **Nav2 integration** — SLAM and navigation bringup (Milestone 2)

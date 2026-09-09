# ROSMASTER M3 Pro Milestone 1: Description Foundation

Goal: one Xacro source that displays correctly in RViz with a TF tree for the mecanum base, 6DOF arm, depth camera, and dual TOF LiDAR.

Authoritative source links:

- Tutorial/course: https://www.yahboom.net/study/ROSMASTER-M3PRO
- GitHub: https://github.com/YahboomTechnology/ROSMASTER-M3PRO
- Product page: https://category.yahboom.net/products/rosmaster-m3-pro
- Official 3D model folder: https://drive.google.com/drive/folders/1h3XF3F35dB8ZMyHR0PDMXYXPyI2LbfiS
- Official CAD file observed in that folder: `ROSMASTER-M3Pro.x_t`, Google Drive file ID `1FF0BK2RmOKguLVwzlIU-Gv7O8d1jm36m`

Known public hardware facts encoded now:

- ROS 2 Humble target.
- 80 mm Mecanum wheels, modeled as 0.040 m radius.
- 6DOF arm.
- Binocular structured-light depth camera.
- Dual TOF LiDAR on front-left and rear-right.

Measurement debt before simulation/control tuning:

- Chassis length, width, height, and mass.
- Wheel center offsets and roller handedness.
- Arm link lengths, joint zero poses, soft limits, and servo sign conventions.
- Camera body dimensions, optical-frame origin, and baseline.
- Front/rear LiDAR exact mounting positions and yaw.

Visual model status:

- Current visuals are improved primitive proxies, not vendor CAD meshes.
- The official model is a Parasolid `.x_t` CAD file, not directly loadable by RViz.
- Next CAD step is to convert the `.x_t` source to STL/DAE/OBJ visual meshes, then keep primitive collision geometry in Xacro.

Run:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rosmaster_m3pro_description rosmaster_m3pro_bringup
source install/setup.bash
ros2 launch rosmaster_m3pro_bringup display.launch.py
```

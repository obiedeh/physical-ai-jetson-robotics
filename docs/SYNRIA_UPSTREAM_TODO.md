# Synria Upstream TODO Activities

This file tracks work derived from public Synria Robotics repositories. The goal
is to align our RTX simulation, ROS 2, MoveIt, C10 camera, and LeRobot paths with
the vendor ecosystem while deferring real Jetson/physical-arm execution until the
hardware phase.

List the structured activity catalog:

```bash
physical-ai-lab synria-upstream-todos
physical-ai-lab synria-upstream-todos --phase rtx-now
physical-ai-lab synria-upstream-todos --phase hardware-later
```

## Upstream Sources

- `Synria-Robotics/Alicia-D-ROS2`: ROS 2 Humble packages for descriptions,
  driver, MoveIt, calibration, cube sorting, and 6D grasping.
- `Synria-Robotics/Alicia-D-SDK`: Python SDK with serial control, gripper
  control, state reads, drag teaching, trajectories, and RoboCore kinematics.
- `Synria-Robotics/Synria-C10-SDK`: USB C10 wrist-camera SDK and OpenCV examples.
- `Synria-Robotics/lerobot`: Synria/Alicia-D robot learning and recording path.
- `Synria-Robotics/Synria-Robot-Descriptions`: vendor robot description source.

## RTX-Now Activities

1. Vendor robot description parity audit
   - Compare our URDF, SRDF, joint limits, frame names, camera link, and gripper
     assumptions against Synria upstream descriptions.
   - Output: `docs/reports/synria_vendor_description_parity.md`

2. ROS 2 Humble-to-Jazzy porting checklist
   - Track package manifest, controller, MoveIt, launch, and Python dependency
     differences needed to run Synria-style packages on this Ubuntu 24.04/Jazzy
     workstation.
   - Output: `docs/reports/synria_ros2_jazzy_port.md`

3. Mock `ros2_control` hardware parity
   - Model vendor-equivalent joint state, joint command, gripper command,
     torque enable, zeroing, and speed-limit surfaces without connecting
     physical hardware yet.
   - Output: `ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml`

4. C10 camera simulation contract
   - Define OpenCV capture assumptions, frame size, camera index discovery,
     ROS image topic mapping, and calibration metadata.
   - Output: `docs/reports/synria_c10_camera_contract.md`

5. Simulated hand-eye calibration
   - Recreate the vendor ArUco / calibration-pose workflow in simulation before
     using the real C10 camera.
   - Output: `reports/synria/hand_eye_calibration_sim.md`

6. Cube sorting simulation demo
   - Build a colored-cube detection, planning, and pick/place simulation aligned
     with Synria's cube-sort package.
   - Output: `reports/demo/synria_cube_sort_sim.md`

7. LeRobot recording schema contract
   - Align synthetic and future real demonstration data with Synria/Alicia-D
     record fields: camera, joint state, actions, FPS, episodes, and timing.
   - Output: `lerobot/configs/synria_aloha_act_notes.yaml`

8. RoboCore-style kinematics benchmark
   - Benchmark FK, IK, Jacobian, and trajectory targets against our RTX synthetic
     reaching policy.
   - Output: `reports/training/synria_kinematics_benchmark.json`

## Hardware-Later Activities

1. Real serial bring-up and firmware detection
   - Verify serial port discovery, `dialout` permissions, firmware version,
     state reads, and debug logging on the physical arm.
   - Output: `reports/synria/real_serial_bringup.md`

2. Safe first motion and zero calibration
   - Run torque enable/disable, zero calibration, reduced-speed joint motion,
     gripper checks, and emergency-stop validation.
   - Output: `reports/synria/first_safe_motion.md`

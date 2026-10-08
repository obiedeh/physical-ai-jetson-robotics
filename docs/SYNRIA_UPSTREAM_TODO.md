# Synria Upstream TODO Activities

This file tracks work derived from public Synria Robotics repositories. The goal
is to align our RTX simulation, ROS 2, MoveIt, C10 camera, and LeRobot paths with
the vendor ecosystem while deferring real Jetson/physical-arm execution until the
hardware phase.

## Upstream Sources

- `Synria-Robotics/Alicia-D-ROS2`: ROS 2 Humble packages for descriptions,
  driver, MoveIt, calibration, cube sorting, and 6D grasping.
- `Synria-Robotics/Alicia-D-SDK`: Python SDK with serial control, gripper
  control, state reads, drag teaching, trajectories, and RoboCore kinematics.
- `Synria-Robotics/Synria-C10-SDK`: USB C10 wrist-camera SDK and OpenCV examples.
- `Synria-Robotics/lerobot`: Synria/Alicia-D robot learning and recording path.
- `Synria-Robotics/Synria-Robot-Descriptions`: vendor robot description source.

## RTX-Now Activities

1. ✅ Vendor robot description parity audit
   - Compare our URDF, SRDF, joint limits, frame names, camera link, and gripper
     assumptions against Synria upstream descriptions.
   - Output: `docs/reports/synria_vendor_description_parity.md` — **created 2026-05-28**

2. ✅ ROS 2 Humble-to-Jazzy porting checklist
   - Track package manifest, controller, MoveIt, launch, and Python dependency
     differences needed to run Synria-style packages on this Ubuntu 24.04/Jazzy
     workstation.
   - Output: `docs/reports/synria_ros2_jazzy_port.md` — **created 2026-05-28**

3. ✅ Mock `ros2_control` hardware parity
   - Model vendor-equivalent joint state, joint command, gripper command,
     torque enable, zeroing, and speed-limit surfaces without connecting
     physical hardware yet.
   - Output: `ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml` — **created 2026-05-27**

4. ✅ C10 camera simulation contract
   - Define OpenCV capture assumptions, frame size, camera index discovery,
     ROS image topic mapping, and calibration metadata.
   - Output: `docs/reports/synria_c10_camera_contract.md` — **created 2026-05-28**

5. ✅ Simulated hand-eye calibration
   - Recreate the vendor ArUco / calibration-pose workflow in simulation before
     using the real C10 camera.
   - Output: `reports/synria/hand_eye_calibration_sim.md` — **created 2026-05-28**

6. ✅ Cube sorting simulation demo
   - Build a colored-cube detection, planning, and pick/place simulation aligned
     with Synria's cube-sort package.
   - Output: `reports/demo/synria_cube_sort_sim.md` — **created 2026-05-28**
   - Implementation: `synria_lerobot/cube_sort.py` — `CubeSortSimulation`, `MockColorDetector`,
     `CubeSortPlanner`; CLI command `physical-ai-lab lerobot-cube-sort-demo`

7. ✅ LeRobot recording schema contract
   - Align synthetic and future real demonstration data with Synria/Alicia-D
     record fields: camera, joint state, actions, FPS, episodes, and timing.
   - Output: `synria_lerobot/configs/synria_aloha_act_notes.yaml` — **created 2026-05-27**

8. ✅ RoboCore-style kinematics benchmark
   - Benchmark FK, IK, Jacobian, and trajectory targets against our RTX synthetic
     reaching policy.
   - Output: `reports/training/synria_kinematics_benchmark.json` — **created 2026-05-28**
   - CLI: `physical-ai-lab synria-kinematics-benchmark`

## All RTX-Now activities complete ✅

All 8 RTX-Now simulation activities are shipped and covered by CI tests.
See `tests/test_synria_upstream_outputs.py` for automated artifact verification.

## Flagship Phase-0 Physical Prerequisites

Recovered physical Synria/Alicia-D serial/state/MoveIt bring-up and
real-to-simulation mirroring exist in an external, uncommitted working tree.
They are **implemented, unmeasured**, not completed evidence for the items
below. The mirroring's GraspGlue behavior is kinematic attach/release, not a
measured physical grasp or autonomous policy result. Formalize the recovery in
this repository with reproducible commands, safety checks, provenance and
committed reports; do not backfill measurements. Recovery facts and current
limitations are frozen in
[`reports/synria/recovered_real_arm_state_2026-09-17.md`](../reports/synria/recovered_real_arm_state_2026-09-17.md).

1. Real serial bring-up and firmware detection
   - Verify serial port discovery, `dialout` permissions, firmware version,
     state reads, and debug logging on the physical arm.
   - Output: `reports/synria/real_serial_bringup.md`
   - Status: external recovery exists; uncommitted and unmeasured here

2. Safe first motion and zero calibration
   - After the read-only recovery artifact is accepted and the operator is
     physically present and explicitly authorizes motion, run reduced-speed
     joint motion, gripper checks, controlled stop/hold and emergency-stop
     validation. Use torque-off only in a verified support-safe pose or when
     required by the manufacturer/controller emergency procedure.
   - Output: `reports/synria/first_safe_motion.md`
   - Status: not established by the recovered mirroring; formal evidence pending

3. Versioned physical dataset state/action contract
   - Derive dimensions, ordering, units and timing from observed physical
     controller and recorder interfaces. Add a passive second-finger
     derivation only if a selected model requires it and physical validation
     supports it; do not assume a seven-to-eight mapping.
   - Output: `reports/synria/physical_dataset_contract.md`
   - Status: planned

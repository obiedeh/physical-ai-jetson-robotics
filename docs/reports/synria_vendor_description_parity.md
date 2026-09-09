# Synria Vendor Description Parity Audit

**Date:** 2026-05-28
**Status:** RTX-complete (simulation); hardware verification pending

---

## Purpose

Compare the URDF, SRDF, joint limits, frame names, camera link, and gripper
assumptions in this repo against the Synria upstream description packages
(`Synria-Robotics/Alicia-D-ROS2`, `Synria-Robotics/Synria-Robot-Descriptions`).

---

## Joint Name Mapping

| Upstream (vendor) | This repo | Match |
|---|---|---|
| `Joint1` | `Joint1` | ✅ |
| `Joint2` | `Joint2` | ✅ |
| `Joint3` | `Joint3` | ✅ |
| `Joint4` | `Joint4` | ✅ |
| `Joint5` | `Joint5` | ✅ |
| `Joint6` | `Joint6` | ✅ |
| `left_finger` | `left_finger` | ✅ |
| `right_finger` | `right_finger` | ✅ |

Joint names are preserved verbatim from the vendor SolidWorks export
(`Alicia_D_v5_6_gripper_50mm`). Capital-J convention is intentional.

---

## Link Name Mapping

| Upstream | This repo | Match |
|---|---|---|
| `base_link` | `base_link` | ✅ |
| `link1` … `link6` | `link1` … `link6` | ✅ |
| `tool0` | `tool0` | ✅ |
| `left_gripper` | `left_gripper` | ✅ |
| `right_gripper` | `right_gripper` | ✅ |

---

## URDF Source

- This repo uses the real vendor URDF (`synria_6dof_arm.urdf`) — a direct
  SolidWorks export from the `Alicia_D_v5_6_gripper_50mm` model.
- The robot name attribute (`Alicia_D_v5_6_gripper_50mm`) is preserved.
- A xacro wrapper (`synria_6dof_arm.urdf.xacro`) adds `world` link, ros2_control
  mock hardware, C10 camera macro, and the `synria_6dof_arm` robot name.

---

## Joint Limits

All six revolute joints are configured in:
- `ros2_ws/src/synria_arm_moveit_config/config/joint_limits.yaml`
- `ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml`

Values sourced from the vendor datasheet (±2.96 rad / 170°):

| Joint | Lower (rad) | Upper (rad) |
|---|---|---|
| Joint1 | −2.96 | 2.96 |
| Joint2 | −2.09 | 2.09 |
| Joint3 | −2.09 | 2.09 |
| Joint4 | −2.96 | 2.96 |
| Joint5 | −1.57 | 1.57 |
| Joint6 | −2.96 | 2.96 |
| left_finger | 0.0 | 0.05 |
| right_finger | −0.05 | 0.0 |

**Note:** Limits should be verified against vendor firmware documentation
before first real-arm motion. See `docs/SYNRIA_ARM_PLAN.md`.

---

## Camera Link

The C10 wrist camera is placed at `tool0` via the `synria_c10_camera` xacro
macro in the xacro wrapper. ROS image topic: `/c10_camera/image_raw`.

Detailed contract: `docs/reports/synria_c10_camera_contract.md`.

---

## SRDF Planning Group

The `synria_arm` planning group (MoveIt 2) is defined in
`ros2_ws/src/synria_arm_moveit_config/config/synria_arm.srdf`:
- chain `base_link` → `tool0`
- 6 revolute joints

---

## Gaps / Open Items

| Item | Status |
|---|---|
| Verify joint limits against vendor firmware | 🔲 Hardware-later |
| DH parameter table (full 3D FK) | 🔲 Hardware-later |
| Real C10 camera calibration | 🔲 Hardware-later |
| Gripper force/speed limits | 🔲 Hardware-later |
| Upstream version pinning (vendor commit SHA) | 🔲 Track when repo becomes public |

# Synria Hand-Eye Calibration — Simulation Run

**Date:** 2026-05-28
**Mode:** RTX simulation (no physical camera, no real arm)
**Status:** Procedure validated; real calibration pending hardware

---

## Objective

Establish the extrinsic transform `T_ee_to_camera` (tool0 → C10 camera
optical frame) using a simulated ArUco-board workflow. Validates that the
calibration pipeline logic is correct before first real-arm use.

---

## Setup

| Component | Simulated value |
|---|---|
| Arm | Synria 6DOF (`synria_6dof_arm.urdf.xacro`) — mock hardware |
| Camera | `MockFrameSource` — 640 × 480 synthetic frames |
| ArUco board | 6 × 4, `DICT_4X4_50`, 40 mm markers (virtual) |
| Calibration poses | 15 poses sampled from `STAGING_POSES` in `arm_control/demo.py` |
| IK solver | Planar FK approximation (`arm_control/kinematics.py`) |

---

## Procedure

1. Generate 15 arm poses by sampling known waypoints from the demo trajectory.
2. For each pose, compute the synthetic camera pose using the nominal
   `tool0` → `c10_camera_optical_frame` transform (xyz 0 0 0.04, rpy 0 0 0).
3. Project a virtual ArUco board centroid into the synthetic camera frame.
4. Accumulate `{T_base_to_ee[i], T_camera_to_marker[i]}` pairs.
5. Solve `AX = XB` using Tsai–Lenz (simulated via matrix algebra).

---

## Simulated Results

| Metric | Value |
|---|---|
| Poses collected | 15 |
| Reprojection error (px, mean) | 0.000 (synthetic — no noise) |
| Translation residual (mm) | 0.000 |
| Rotation residual (deg) | 0.000 |
| Nominal transform used | xyz [0, 0, 0.04], rpy [0, 0, 0] |

**Expected result:** In a noise-free simulation the recovered transform
matches the nominal exactly. The zero residual confirms the pipeline algebra
is correct. Real-hardware residuals will reflect measurement noise and
mechanical flex.

---

## Calibration Output (Nominal)

```yaml
# config/c10_hand_eye_transform.yaml  (PLACEHOLDER — replace with real values)
hand_eye_calibration:
  method: tsai_lenz
  date: 2026-05-28
  mode: simulated
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.040
    rotation_rpy_rad:
      roll: 0.0
      pitch: 0.0
      yaw: 0.0
  reprojection_error_px: 0.0
  note: >
    Nominal placeholder. Replace with output of real ArUco calibration run.
    See docs/reports/synria_c10_camera_contract.md for full procedure.
```

---

## Next Steps

| Step | Status |
|---|---|
| Print physical ArUco board (6×4, 40 mm) | 🔲 Hardware-later |
| Mount C10 on arm, connect via USB | 🔲 Hardware-later |
| Run `cv2.calibrateHandEye` on real pose pairs | 🔲 Hardware-later |
| Commit real transform to `config/c10_hand_eye_transform.yaml` | 🔲 Hardware-later |
| Re-run this report with real residuals | 🔲 Hardware-later |

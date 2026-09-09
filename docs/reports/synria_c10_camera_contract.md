# Synria C10 Camera Simulation Contract

**Date:** 2026-05-28
**Status:** RTX-simulation contract defined; real-hardware calibration pending

---

## Purpose

Define the interface contract for the Synria C10 wrist camera so that:
- Simulation (mock frame source) and real hardware use the same ROS topic API
- Hand-eye calibration (sim and real) share the same ArUco/marker pipeline
- LeRobot recording picks up the same `/c10_camera/image_raw` stream

---

## Hardware Specification

| Property | Value | Source |
|---|---|---|
| Model | Synria C10 | `Synria-Robotics/Synria-C10-SDK` |
| Interface | USB (UVC) | SDK README |
| Resolution (default) | 1280 × 720 | SDK default |
| Frame rate | 30 FPS | SDK default |
| Color encoding | BGR8 → convert to RGB8 for ROS | OpenCV default |
| Camera index (USB) | `/dev/video0` or autodetect | environment-specific |

---

## ROS 2 Topic Contract

| Topic | Message type | Notes |
|---|---|---|
| `/c10_camera/image_raw` | `sensor_msgs/Image` | RGB8 encoding |
| `/c10_camera/camera_info` | `sensor_msgs/CameraInfo` | calibration matrix |
| `/c10_camera/image_compressed` | `sensor_msgs/CompressedImage` | optional, JPEG |

Frame ID: `c10_camera_optical_frame`

---

## URDF Placement

The C10 camera link is declared via the `synria_c10_camera` xacro macro in
`ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro`.

Nominal placement (eye-in-hand): mounted at `tool0`, looking forward along the
arm's approach direction. Exact transform requires hardware measurement.

| Frame | Parent | Transform |
|---|---|---|
| `c10_camera_optical_frame` | `tool0` | TBD — hardware calibration |
| `c10_camera_link` | `tool0` | nominal: xyz 0 0 0.04, rpy 0 0 0 |

---

## Simulation Mock Contract

In CI and RTX simulation (no physical camera):
- `edge_ai.camera_inference.MockFrameSource` generates synthetic frames
- Frame size: 640 × 480 (reduced for speed)
- ONNX runner in mock mode returns zero-filled arrays
- ROS image topic is bridged by publishing numpy arrays via `cv_bridge`

```python
from edge_ai.camera_inference import MockFrameSource, CameraInferenceLoop

source = MockFrameSource(width=640, height=480, fps=30)
loop = CameraInferenceLoop(source, model_path="mock")
```

---

## Hand-Eye Calibration Pipeline

See `reports/synria/hand_eye_calibration_sim.md` for the simulated version.

Real pipeline (hardware-later):
1. Print 6 × 4 ArUco board (marker size 40 mm, `DICT_4X4_50`)
2. Place board at 15–20 known arm poses
3. Record `{T_base_to_ee, T_camera_to_marker}` pairs
4. Solve `AX = XB` (Tsai–Lenz method) via `cv2.calibrateHandEye`
5. Write extrinsic matrix to `config/c10_hand_eye_transform.yaml`

---

## LeRobot Schema Alignment

The `lerobot/configs/synria_aloha_act_notes.yaml` expects:

```yaml
observations:
  images:
    c10_camera:
      topic: /c10_camera/image_raw
      encoding: rgb8
```

This contract defines how that topic is populated in both simulation
and real hardware modes.

---

## Open Items

| Item | Status |
|---|---|
| Measure physical camera offset from `tool0` | 🔲 Hardware-later |
| Run real ArUco hand-eye calibration | 🔲 Hardware-later |
| Verify USB index on Jetson Orin NX | 🔲 Hardware-later |
| Publish calibration YAML to `config/` | 🔲 Hardware-later |

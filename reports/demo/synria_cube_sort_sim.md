# Synria Cube Sorting Simulation Demo

**Date:** 2026-05-28  
**Mode:** RTX simulation (no physical arm, no camera)  
**Seed:** 42  

## Summary

| Metric | Value |
|---|---|
| Cubes sorted | 6 |
| Passed | 6 |
| Failed | 0 |
| Pass rate | 100% |
| Total trajectory time | 90.0 s |

## Color → Zone Assignments

| Color | Zone | Count |
|---|---|---|
| blue | top | 1 (cubes: [2]) |
| green | left | 2 (cubes: [1, 5]) |
| red | right | 2 (cubes: [0, 4]) |
| yellow | bottom | 1 (cubes: [3]) |

## Per-Cube Results

| Cube | Color | Zone | Confidence | Traj steps | Duration (s) | Pass |
|---|---|---|---|---|---|---|
| 0 | red | right | 0.972 | 8 | 15.0 | ✅ |
| 1 | green | left | 0.825 | 8 | 15.0 | ✅ |
| 2 | blue | top | 0.825 | 8 | 15.0 | ✅ |
| 3 | yellow | bottom | 0.913 | 8 | 15.0 | ✅ |
| 4 | red | right | 0.958 | 8 | 15.0 | ✅ |
| 5 | green | left | 0.939 | 8 | 15.0 | ✅ |

## Pipeline

1. **MockColorDetector** — seeded RNG assigns color labels and board positions
   (simulates YOLOv8 / OpenCV HSV overhead detection)
2. **CubeSortPlanner** — maps color → staging zone; builds
   `home → board_approach → staging_zone → home` trajectory via `arm_control`
3. **ArmSafetyGate** — validates every waypoint against URDF joint limits

## Hardware Path

Replace `MockColorDetector` with the real overhead board camera pipeline
(YOLOv8 detection + ArUco homography for board-to-world transform) and
replace trajectory replay with live MoveIt 2 execution via
`physical_ai_ops_copilot` or the Synria arm ROS 2 driver.

See `docs/reports/synria_c10_camera_contract.md` for camera integration contract.

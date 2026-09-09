# Synria Robotic Arm Bring-Up Report

Date:

Operator:

Location:

## Hardware

- Arm model:
- Reach:
- Payload limit:
- Controller:
- C10 camera connected:
- End effector:
- Power notes:

## Software

- OS:
- ROS distro:
- MoveIt / MoveIt 2:
- Vendor SDK:
- Simulation packages built:
- Repo commit:

## Simulation-First Evidence

Complete this section before enabling motors or connecting to a serial controller.

Commands:

```bash
pytest -q tests/test_simulation_foundation.py
bash scripts/linux_rtx/smoke_robot_descriptions.sh
```

Results:

- `synria_arm_description` Xacro rendered:
- C10 camera frame present:
- Mock ros2_control system present:
- MoveIt config present:
- Hardware dependencies intentionally excluded:

Evidence paths:

- Static test log:
- URDF render log:
- RViz screenshot:
- Known model approximations:

## Inventory Evidence

Inventory JSON:

```text
reports/inventory/<file>.json
```

Commands run:

```bash
physical-ai-lab collect-inventory --target synria-host --output reports/inventory/synria_arm.json
```

## Camera Test

- Camera detected:
- Device path:
- ROS image topic:
- Resolution:
- FPS:
- Calibration file:
- Screenshot:

## MoveIt Mock Planning

- URDF loaded:
- Planning group:
- Mock hardware launched:
- Joint-space plan:
- Cartesian pose plan:
- Collision object test:

## Real Hardware Motion

Only complete after mock control and MoveIt planning evidence are captured.

- Joint states visible:
- Servo/motor enable:
- Reduced speed used:
- First joint move:
- First Cartesian move:
- Emergency stop tested:

## Eye-In-Hand Calibration

- Camera frame:
- End-effector frame:
- Transform method:
- Calibration target:
- Reprojection / quality notes:

## Learning / Demonstration Data

- LeRobot / ALOHA-compatible dataset path:
- Camera synchronized:
- Joint/action synchronized:
- Demonstration count:
- Task:

## Platform Integration Notes

- Upstream vendor source/version:
- Repo-owned files changed:
- Vendor files copied:
- Simulation gaps before hardware:
- Hardware-only assumptions found:

## Business Relevance

Explain what this test proves for manipulation, inspection, sorting, or sim-to-real robotics.

## Next Actions

- [ ] 
- [ ] 
- [ ] 

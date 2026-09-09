# Synria Robotic Arm Plan

This project targets the Synria AI Robotic Arm Kit as the primary manipulation platform.

## Hardware Profile

- Robot: Synria AI Robotic Arm Kit
- Degrees of freedom: 6DOF
- Reach: 650 mm
- Payload: 1 kg
- Camera: Synria C10 wrist camera
- Software ecosystem: ROS 1, ROS 2, MoveIt / MoveIt 2
- Learning ecosystem: LeRobot / ALOHA-compatible workflows

The C10 camera is treated as the default eye-in-hand sensor for manipulation, visual servoing, and data collection.

## Business Use Case

The arm track demonstrates Physical AI for small industrial workcells:

- visual inspection
- pick-place sorting
- bin interaction
- operator-guided demonstration learning
- sim-to-real manipulation
- edge robot control on Jetson

This is recruiter-relevant because it combines perception, motion planning, data collection, robot learning, and real deployment.

## Milestone A: Safe Bring-Up

Goals:

- verify power, controller, and emergency stop
- identify official SDK and ROS packages
- confirm URDF and MoveIt configuration availability
- run camera-only tests before robot motion
- run MoveIt with mock hardware
- document safe workspace bounds

Outputs:

- bring-up notes
- photos/screenshots
- camera test result
- MoveIt planning screenshot
- known limitations

Start from the report template:

```text
reports/synria/BRINGUP_TEMPLATE.md
```

Recommended first evidence command on the host connected to the arm:

```bash
physical-ai-lab collect-inventory --target synria-host --output reports/inventory/synria_arm.json
```

## Milestone B: ROS 2 + MoveIt 2 Control

Goals:

- compare against Synria upstream Alicia-D ROS 2 package structure
- document ROS 2 Humble-to-Jazzy compatibility gaps
- launch arm description
- validate joint states
- run planning scene
- execute simple joint-space moves
- execute simple Cartesian pose targets
- define collision objects for a tabletop workcell
- add mock parity for gripper command, torque enable, zeroing, and speed limits

Outputs:

- launch notes
- sample motion scripts
- MoveIt planning config notes
- safety checklist

## Milestone C: Eye-In-Hand Vision

Goals:

- define C10 OpenCV capture and ROS image topic contract
- launch C10 camera as a ROS 2 image source
- calibrate camera intrinsics
- estimate camera-to-end-effector transform
- detect fiducials or objects
- publish object pose into the planning frame
- simulate ArUco hand-eye calibration before real C10 capture

Outputs:

- calibration file
- camera transform notes
- object detection demo
- pose-to-grasp proof of concept

## Milestone D: OpenUSD / Isaac Digital Twin

Goals:

- create or import arm geometry
- build a tabletop workcell scene
- add camera, objects, bins, and collision geometry
- validate reachability and camera viewpoints
- prepare synthetic data scenes

Outputs:

- USD scene files
- screenshots
- scene setup guide
- synthetic data sample

## Milestone E: LeRobot / ALOHA-Style Data Collection

Goals:

- align with Synria/Alicia-D LeRobot recording schema
- record teleoperation or scripted demonstrations
- store synchronized camera, joint state, action, and task metadata
- define dataset cards for reproducibility
- train a first behavior-cloning policy where practical

Outputs:

- dataset schema
- demonstration collection script
- baseline learned policy report
- failure cases and next steps

## Milestone F: Sim-to-Real Manipulation

Goals:

- train reaching or pick-place in simulation
- randomize object pose, lighting, camera noise, mass, and friction
- deploy a constrained policy to the real arm
- compare simulation and real-world success rates

Outputs:

- training config
- sim evaluation metrics
- real-world evaluation metrics
- sim-to-real gap analysis

## Safety Rules

- Keep payload below vendor limits.
- Test motion with reduced speed first.
- Keep the arm workspace clear during autonomous execution.
- Use mock hardware planning before real execution.
- Require explicit operator confirmation for new autonomous tasks.
- Log all real robot commands with timestamps.

## Upstream Alignment

Detailed vendor-derived TODOs are tracked in:

```text
docs/SYNRIA_UPSTREAM_TODO.md
```

The current priority is RTX-first: description parity, Jazzy port notes,
simulation-only `ros2_control`, C10 camera contract, simulated hand-eye
calibration, cube sorting, LeRobot schema, and kinematics benchmarks. Real
serial bring-up and first motion stay in the hardware-later phase.

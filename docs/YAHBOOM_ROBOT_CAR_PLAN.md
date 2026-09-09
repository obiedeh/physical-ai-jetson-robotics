# Yahboom Orin NX Robot Car Plan

This project targets the Yahboom Orin NX 8GB ROS 2 robot car as the primary mobile robotics and mobile manipulation platform.

## Hardware Profile

- Robot: Yahboom Orin NX 8GB ROS 2 robot car
- Compute: NVIDIA Jetson Orin NX 8GB
- Chassis: mecanum wheel omnidirectional base
- Manipulator: 6DOF arm configuration
- Interface: display-equipped version
- Capabilities: multimodal AI large model support, AI voice interaction, vision recognition

The Yahboom platform is the best target for the robo car, SLAM, multimodal interaction, edge AI, and mobile manipulation tracks.

## Business Use Case

The robot car track demonstrates a smart-factory mobile manipulation workflow:

- navigate to a workcell
- inspect a scene with onboard vision
- understand an operator command
- move objects or trigger a robotic-arm task
- report telemetry, localization quality, and task status
- run edge inference on Jetson Orin NX

This ties directly to warehouse automation, factory logistics, autonomous inspection, and embodied AI.

## Milestone A: Safe Bring-Up

Goals:

- identify exact Yahboom model and tutorial package
- record JetPack, Ubuntu, ROS 2, CUDA, and TensorRT versions
- verify battery, power, emergency stop, and chassis safety
- test manual teleoperation
- validate display, microphone, speaker, camera, base, and arm topics

Outputs:

- bring-up checklist
- hardware/software version table
- topic inventory
- first teleoperation notes

Start from the report template:

```text
reports/yahboom/BRINGUP_TEMPLATE.md
```

Recommended first evidence command on the robot:

```bash
bash scripts/jetson/collect_inventory.sh jetson-orin reports/inventory/yahboom_orin_nx.json
```

## Milestone B: Mecanum Base Calibration

Goals:

- validate wheel direction and frame conventions
- calibrate linear and angular velocity commands
- measure odometry drift
- tune mecanum kinematics parameters
- compare commanded motion vs measured motion

Outputs:

- calibration procedure
- motion accuracy table
- odometry drift report
- recommended speed limits

## Milestone C: SLAM and Navigation

Goals:

- run mapping in a controlled indoor environment
- validate localization stability
- configure Nav2 or the vendor navigation stack
- test obstacle avoidance
- benchmark CPU/GPU load on Jetson Orin NX

Outputs:

- saved map
- SLAM launch notes
- navigation demo
- localization failure cases
- Jetson resource report

## Milestone D: Vision Recognition and Multimodal Interaction

Goals:

- validate camera streams
- run object detection or scene recognition
- validate voice input and output
- add operator command parsing
- connect recognized objects to robot actions

Outputs:

- vision recognition demo
- voice command demo
- latency notes
- safety-bounded command interface

## Milestone E: Mobile Manipulation

Goals:

- coordinate base pose, arm workspace, and camera observations
- navigate to a target station
- use the 6DOF arm for simple interaction
- define collision boundaries around the base and arm
- log task success and failure modes

Outputs:

- mobile manipulation task script
- base-to-arm frame notes
- task success report
- safety checklist

## Milestone F: OpenUSD / Isaac Digital Twin

Goals:

- create a digital twin of the mecanum base and 6DOF arm
- simulate indoor navigation scenes
- simulate camera viewpoints and obstacles
- validate mobile manipulation reachability
- produce screenshots and synthetic data samples

Outputs:

- USD scene files
- simulation screenshots
- OpenUSD setup guide
- sim-to-real comparison notes

## Safety Rules

- Test teleoperation before autonomy.
- Keep speed limits low during early navigation.
- Disable arm motion when testing base control.
- Disable base motion when testing arm control.
- Use a clear indoor area for first SLAM runs.
- Log all autonomous commands and operator interventions.

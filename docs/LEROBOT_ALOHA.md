# LeRobot and ALOHA-Compatible Learning

This repo uses LeRobot as the robot-learning and imitation-learning layer for the Synria 6DOF arm track. The Synria arm is described as LeRobot and ALOHA compatible, so the project treats ALOHA as a workflow compatibility target rather than a separate native host stack.

## Installed Status

- LeRobot: installed in `.venv`
- LeRobot version: `0.5.1`
- PyTorch: installed with CUDA 12.8 wheels
- CUDA availability: validated through `torch.cuda.is_available()`
- ROS 2 Jazzy: installed
- MoveIt 2: installed
- ROS 1: not installed natively on this Ubuntu 24.04 host

ROS 1 Noetic is not a native Ubuntu 24.04 target. If ROS 1 is needed for a vendor tool, keep it isolated in Docker, RoboStack, or a separate Ubuntu 20.04 machine/VM instead of mixing it into the host ROS 2 Jazzy environment.

## Project Roles

LeRobot handles:

- dataset format
- teleoperation and recording workflows
- ACT/ALOHA-style imitation learning
- policy training and evaluation
- model and dataset publishing through Hugging Face tooling

ROS 2 + MoveIt handles:

- robot description
- joint states
- kinematics
- planning
- control interfaces
- simulation integration

Gazebo handles:

- factory-cell simulation
- robot world layout
- sensor and control simulation

## Synria Observation and Action Contract

Initial observations:

- `observation.images.c10_camera`: Synria C10 camera RGB frames
- `observation.state`: six arm joint positions
- `observation.ee_pose`: optional end-effector pose from MoveIt/TF

Initial actions:

- `action`: six target joint positions

Later extensions:

- gripper command if the installed Synria kit exposes one
- wrist/current/effort telemetry if exposed by the vendor controller
- bimanual ALOHA-style paired-arm actions if a second arm is added

## Validated Commands

Activate the repo environment:

```bash
cd /home/oedeh/physical-ai-jetson-robotics-lab
source .venv/bin/activate
```

Check LeRobot:

```bash
bash scripts/linux_rtx/check_lerobot.sh
```

Inspect available cameras:

```bash
lerobot-find-cameras
```

Train with an ALOHA-style environment and ACT policy once a dataset exists:

```bash
lerobot-train \
  --dataset.repo_id obiedeh/synria-c10-aloha-demo \
  --env.type aloha \
  --policy.type act
```

## Dataset Layout

Local datasets should live outside git-tracked source files:

```text
data/lerobot/
```

Do not commit generated datasets, raw camera recordings, or trained model checkpoints. Publish shareable datasets or models through Hugging Face when appropriate.

## Next Steps

- map Synria vendor joint names to `joint_1` through `joint_6`
- connect the C10 camera to LeRobot camera discovery
- add a ROS 2 bridge script that samples `/joint_states`, TF, and camera frames into the LeRobot observation/action contract
- record a small calibration dataset
- train an ACT policy on the recorded dataset

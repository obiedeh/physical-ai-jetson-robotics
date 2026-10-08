# Phase 0A safety and recovery record

Status: template only. No hardware verification is recorded here yet.

- UTC date:
- Operator:
- Location:
- Repository commit:
- Controller host:
- External driver package version:

## Device identity

- Follower model and serial (`ADF...` expected, unverified):
- Follower stable `/dev/serial/by-id/` path:
- Leader model and serial (`ADL...` expected, unverified):
- Leader stable `/dev/serial/by-id/` path:
- C10 stable camera ID:
- Fixed webcam stable camera ID:
- Gripper type (`50mm` or `100mm`):

## Candidate interface facts to verify

- `/joint_states` names are `Joint1` through `Joint6` plus `Gripper`:
- Effective follower state rate:
- Effective leader state rate:
- Gripper convention is zero open and positive stroke closed:
- Baud rate is 1,000,000:
- Leader writes are refused:
- Follower command input remains owned by the existing teleoperation:

## Safety controls

- Arm secured to stand:
- E-stop within reach and tested per manufacturer procedure:
- Controlled stop/hold procedure:
- Support-safe pose:
- Manufacturer emergency procedure available:
- Power source and state:
- Joint and gripper limits verified by:
- Joint and gripper limits verified on:

## Read-only observations

- Commands used:
- Observed topics and rates:
- Camera identities, resolutions, and rates:
- Deviations or incidents:
- Operator acceptance:

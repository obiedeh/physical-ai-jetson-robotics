# D1 operator runbook

This runbook is for an operator-authorized future hardware session. The code
path is **implemented, unmeasured**; D1 is **planned** at 0/100 qualifying
episodes. Complete the Phase 0A and first-safe-motion templates before a
qualifying session.

## 1. Power-on and preflight

Secure the follower arm, place the e-stop within reach, inspect the work area,
and power the leader, follower, C10 wrist camera, and fixed webcam according to
their manufacturer procedures. Fill
[`phase0a_safety_recovery.md`](../../reports/synria/phase0a_safety_recovery.md).
Do not proceed while any identity, stop/hold behavior, gripper type, or limit is
unknown.

The confirmed default wiring uses the leader-to-follower hardware sync cable.
The leader is not connected to the PC; only follower `/joint_states` is read.
Recording therefore defaults to `next_state`: each action is the next sampled
follower position/gripper state, not a directly measured leader command.

Set session values using stable device identities:

```bash
export SESSION=YYYYMMDD-HHMM-operator
export DATASET_ROOT=/srv/synria-d1/$SESSION
export WRIST_CAMERA=/dev/v4l/by-id/REPLACE_WITH_C10_ID
export FRONT_CAMERA=/dev/v4l/by-id/REPLACE_WITH_FIXED_WEBCAM_ID
export OPERATOR="REPLACE_WITH_OPERATOR_NAME"
export FOLLOWER_SERIAL="REPLACE_WITH_ADF_SERIAL"
export LEADER_SERIAL="REPLACE_WITH_ADL_SERIAL"
export POWER_STATE_START="REPLACE_WITH_START_STATE"
export POWER_STATE_END="REPLACE_WITH_END_STATE"
export SCENE="REPLACE_WITH_SCENE_DESCRIPTION"
export GIT_SHA=$(git rev-parse HEAD)
```

Use the existing teleoperation with the operator's already validated procedure
and record its exact command in
[`first_safe_motion.md`](../../reports/synria/first_safe_motion.md). No command
for that external procedure is assumed by this repository. No additional
leader driver is needed for the default recording path.

### Optional: leader actions with an additional USB connection

Only if the operator also connects the leader to the PC by USB and chooses
direct leader-state actions, build and source the read-only leader package:

```bash
export LEADER_PORT=/dev/serial/by-id/REPLACE_WITH_LEADER_USB_ID
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select synria_arm_bringup
source install/setup.bash
cd ..
ros2 launch synria_arm_bringup leader_state.launch.py leader_port:=$LEADER_PORT
```

This optional launch starts one additional command-disabled driver instance
and remaps only its topics. It does not start, stop, or replace the existing follower
driver or hardware-sync teleoperation. For this alternative, add
`--action-source leader` to the recorder commands below and use
`--action-source leader` in the session summary. Use a separate dataset;
never mix action sources in one dataset or summary.

## 2. Disposable smoke episode

Run the recorder in a separate terminal. The smoke dataset uses a temporary
directory, stops at 20 seconds, is deleted on exit, and never counts toward D1.
Source ROS in every recorder terminal, including the default follower-only path:

```bash
source /opt/ros/jazzy/setup.bash
```

```bash
python -m synria_lerobot.recorder \
  --smoke \
  --repo-id local/synria-d1-smoke \
  --gripper-type 50mm \
  --wrist-camera "$WRIST_CAMERA" \
  --front-camera "$FRONT_CAMERA" \
  --fps 30
```

At the prompt enter `start`, operate for 20 seconds, then enter `success` or
`failure`. Confirm fresh follower state, derived next-state actions, both
camera streams, and a final front-camera still before proceeding.

## 3. Recording session

Use the same existing leader/follower teleoperation without inserting a
software bridge. With the default follower-only action source:

```bash
python -m synria_lerobot.recorder \
  --dataset-path "$DATASET_ROOT" \
  --repo-id "local/synria-d1-$SESSION" \
  --gripper-type 50mm \
  --wrist-camera "$WRIST_CAMERA" \
  --front-camera "$FRONT_CAMERA" \
  --fps 30
```

Use `start`, `stop`, `success`, `failure`, and `discard` as defined in the
[dataset protocol](D1_DATASET_PROTOCOL.md). Omitting `--action-source` selects
`next_state`; `--action-source next_state` is also accepted explicitly.

## 4. Gate and summarize the session

After the operator ends teleoperation using its established safe procedure and
records start/end power state, run:

```bash
physical-ai-lab d1-session-summary \
  --session-id "$SESSION" \
  --records "$DATASET_ROOT/physical_quality_records.jsonl" \
  --dataset-path "$DATASET_ROOT" \
  --follower-serial "$FOLLOWER_SERIAL" \
  --leader-serial "$LEADER_SERIAL" \
  --host "$(hostname)" \
  --git-sha "$GIT_SHA" \
  --operator "$OPERATOR" \
  --power-state-start "$POWER_STATE_START" \
  --power-state-end "$POWER_STATE_END" \
  --scene "$SCENE" \
  --wrist-camera-id "$WRIST_CAMERA" \
  --front-camera-id "$FRONT_CAMERA" \
  --width 224 --height 224 --rate-hz 30 \
  --gripper-type 50mm --action-source next_state \
  --utc-date "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

physical-ai-lab d1-dataset-summary
```

Review every gate result and complete the generated `session_notes.md`. The
limits file must contain operator verification before episodes become
qualifying D1 progress.

## 5. Commit the evidence

Raw datasets and video remain outside git. Commit only the small session and
aggregate artifacts in the same change:

```bash
git add "reports/ludo_flagship/data/$SESSION" \
  reports/ludo_flagship/data/D1_dataset_summary.json \
  docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria D1 session $SESSION"
```

Object success is the operator's label plus a camera still; no independent sensor confirms it.

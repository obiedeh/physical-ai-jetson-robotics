# D1 operator runbook

For an explicitly operator-authorized future physical Synria/Alicia-D session.
The path is **implemented, unmeasured**; D1 is **planned** at 0/100 qualifying
episodes. Checks used fakes and synthetic images, including upstream LeRobot
0.6.2. They do not verify the robot, cameras, or ROS runtime.

## 1. Safety, wiring, and recording environment

Complete and accept the
[Phase 0A record](../../reports/synria/phase0a_safety_recovery.md) and
[first-safe-motion record](../../reports/synria/first_safe_motion.md) before
qualifying collection. Secure the arm, keep the e-stop within reach, inspect
the workspace, and follow manufacturer power-on and controlled stop/hold
procedures. Do not proceed with unknown identity, gripper type, limits, or
stop behavior. Do not switch torque off on an unsupported arm.

Default wiring uses the leader-to-follower hardware sync cable. The leader is
not connected to the PC; only follower `/joint_states` is read. Use the
operator's existing validated teleoperation unchanged and record its procedure
in the first-safe-motion record. This repository does not start it or insert
software between the arms. Mark leader USB fields in safety records as not
applicable when it is not connected.

Use a separate recording terminal/environment; never alter the working
teleoperation environment. Recording requires Python 3.12, compatible ROS 2
Jazzy `rclpy`/`sensor_msgs`, and upstream LeRobot 0.6.x. Core/fake tests support
Python 3.10, but that does not install the optional 0.6 dataset dependency.
From the repository root, create a new dedicated environment at an
operator-owned path, once:

```bash
export RECORDING_VENV=/srv/venvs/synria-d1-py312
test ! -e "$RECORDING_VENV" &&
  python3.12 -m venv --system-site-packages "$RECORDING_VENV" &&
  source "$RECORDING_VENV/bin/activate" &&
  python -m pip install -e '.[robot-learning]'
```

If creation or activation fails, stop; do not install into another environment.
The extra supplies OpenCV and dataset/video dependencies. Verify host
compatibility during authorized preflight; do not substitute the vendor fork.
In every recorder or summary terminal, restore the session variables below
and source the recording environment and ROS, even without a leader launch:

```bash
source "$RECORDING_VENV/bin/activate"
source /opt/ros/jazzy/setup.bash
```

## 2. Select and retain session settings

For a new session only, initialize the values below and replace all placeholders.
`GRIPPER_TYPE` must be the installed `50mm` or
`100mm`; there is no default. Use stable device IDs and operator-writable
storage outside git with sufficient space. Do not create `DATASET_ROOT`
itself: the library creates it or resumes a matching dataset.

```bash
export SESSION=YYYYMMDD-HHMM-operator
export DATASET_ROOT=/srv/synria-d1/$SESSION
export GRIPPER_TYPE=REPLACE_WITH_INSTALLED_TYPE
export ACTION_SOURCE=next_state
export ACTION_LOOKAHEAD_STEPS=1
export FPS=15
export IMAGE_WIDTH=224
export IMAGE_HEIGHT=224
export WRIST_CAMERA=/dev/v4l/by-id/REPLACE_WITH_C10_ID
export FRONT_CAMERA=/dev/v4l/by-id/REPLACE_WITH_FIXED_WEBCAM_ID
export OPERATOR="REPLACE_WITH_OPERATOR_NAME"
export FOLLOWER_SERIAL="REPLACE_WITH_ADF_SERIAL"
export LEADER_SERIAL="REPLACE_WITH_ADL_SERIAL_FROM_ARM_LABEL"
export POWER_STATE_START="REPLACE_WITH_OBSERVED_START_STATE"
export SCENE="REPLACE_WITH_SCENE_DESCRIPTION"
export GIT_SHA=$(git rev-parse HEAD)
```

`--fps` is required and positive integer-valued. The vendor suggests 15 or 30;
`FPS=15` above is an explicit candidate, not a measurement. Each process counts
actual follower callbacks over two seconds and checks source/header freshness
before opening a dataset or camera. It refuses missing/stale data, failed
sources, or a requested rate above the measured rate. Correct the cause before
retrying; never assume the driver's timer rate. Requested, incoming, and
achieved sample rates are separate facts.

Image width/height default to 224. Capture converts BGR to RGB and resizes
before buffering; provenance records actual native/stored resolutions and
camera IDs. Leave `--state-has-velocity` absent for the seven-value state.
Use it consistently only for a new contract requiring six additional reported
finite joint velocities; missing required velocities fail preflight/capture.

`next_state` is the default even if `--action-source` is omitted. Frame i uses
follower state i+k, clamped to the last frame, with nonnegative integer
`k=ACTION_LOOKAHEAD_STEPS` (default 1). Nominal delay k/FPS is not measured
latency; original action/state timestamps preserve actual offsets. The
tail-clamped offsets become shorter or zero.

### Optional: direct leader actions over additional USB

Only when the operator also connects the leader by USB and authorizes the
additional read-only instance, choose a separate `SESSION`/`DATASET_ROOT` and
set `ACTION_SOURCE=leader` in the recording terminal. With the external driver
workspace already installed/sourced, use a separate ROS terminal from the
repository root:

```bash
export LEADER_PORT=/dev/serial/by-id/REPLACE_WITH_LEADER_USB_ID
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select synria_arm_bringup
source install/setup.bash
cd ..
ros2 launch synria_arm_bringup leader_state.launch.py leader_port:="$LEADER_PORT"
```

This configuration starts one additional leader instance with
`joint_commands_enabled=false`, without changing the follower or teleoperation.
Its four absolute remaps are:

| Original topic | Isolated topic |
| --- | --- |
| `/joint_states` | `/leader/joint_states` |
| `/joint_commands` | `/leader/disabled_joint_commands` |
| `/zero_calibrate` | `/leader/disabled_zero_calibrate` |
| `/demonstration` | `/leader/disabled_demonstration` |

Supplied vendor interface facts warn that zero calibration can switch torque
off. Do not issue calibration/demonstration commands to this instance. The
launch has not been validated on hardware. Leader actions stay direct:
configured k is recorded, but effective k and nominal delay are zero.

## 3. Disposable smoke episode

After authorized preflight and the existing teleoperation procedure, run:

```bash
python -m synria_lerobot.recorder \
  --smoke \
  --repo-id local/synria-d1-smoke \
  --gripper-type "$GRIPPER_TYPE" \
  --action-source "$ACTION_SOURCE" \
  --action-lookahead-steps "$ACTION_LOOKAHEAD_STEPS" \
  --wrist-camera "$WRIST_CAMERA" \
  --front-camera "$FRONT_CAMERA" \
  --image-width "$IMAGE_WIDTH" --image-height "$IMAGE_HEIGHT" \
  --fps "$FPS"
```

Enter `start`, teleoperate for 20 seconds until the automatic cap stops capture,
then enter `success` or `failure` under the protocol rules. After one saved
episode, the process finalizes and exits. Its temporary dataset and still are
deleted on exit and never count toward D1. Smoke checks the write path; there
is no live preview or automatic quality-gate report, and no retained visual
evidence from its deleted still. Resolve failures, then review retained data.

## 4. Multi-episode recording; review the first before scaling

```bash
python -m synria_lerobot.recorder \
  --dataset-path "$DATASET_ROOT" \
  --repo-id "local/synria-d1-$SESSION" \
  --gripper-type "$GRIPPER_TYPE" \
  --action-source "$ACTION_SOURCE" \
  --action-lookahead-steps "$ACTION_LOOKAHEAD_STEPS" \
  --wrist-camera "$WRIST_CAMERA" \
  --front-camera "$FRONT_CAMERA" \
  --image-width "$IMAGE_WIDTH" --image-height "$IMAGE_HEIGHT" \
  --fps "$FPS"
```

At the prompt:

- `start`: begin the next episode while idle.
- `stop`: finish capture after 20–30 seconds. The 30-second hard cap also stops
  capture. Stopping alone neither saves nor labels the episode.
- `success` or `failure`: label and save the stopped episode plus final still.
  The saved line reports achieved sample rate.
- `retry`: retry a failed save with the same retained frames and label.
- `discard`: explicitly abandon pending frames after a setup/recording fault;
  note the reason. Do not erase genuine task failures.
- `quit`: finalize and exit while idle. Save or explicitly discard pending
  frames first; a new `start` cannot replace them.

Normal saves return to idle for another episode in the same process. Save
errors retain frames in memory for retry/discard. If recovery is blocked, stop
collection: the lock/journal remain and further writes are refused. Do not
remove them to bypass refusal. Pending frames are not durable across a process
kill, EOF, interrupt, or host failure.

First save one retained episode and `quit`. Run section 5, then manually review
both recorded views and the timestamp-linked still for orientation, color,
framing, freshness, and action/state alignment. Image gates cannot establish
task visibility. Only then rerun the recorder command to scale collection;
there is no automatic viewer or physical-success detector.

Resume using the same `DATASET_ROOT`, `SESSION`, metadata/source revision,
physical setup, and contract/image settings. Episode numbering continues from
metadata; each run retains its incoming-rate evidence. Update the same
session's cumulative summary, never a second summary for the same dataset.
Restore the original exports, including `GIT_SHA`, `POWER_STATE_START`, and
identity/scene values; do not rerun the initialization block or recompute
`GIT_SHA` after an evidence-only commit. Retain the original recording-code
revision while that code/configuration is unchanged.
Record restarts and power transitions in session notes. Changed requested FPS,
k, source, gripper, velocity mode, image size, camera identity/native size,
operator, scene, or recording code/configuration requires a separate dataset/session.
Incompatible or incomplete roots are refused, not overwritten.

## 5. Gate, review, and summarize retained data

End the recorder and safely pause/end teleoperation by its established
procedure. Enter observed end power state. From the repository root, with the
same variables and recording environment:

```bash
export POWER_STATE_END="REPLACE_WITH_OBSERVED_END_STATE"
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
  --width "$IMAGE_WIDTH" --height "$IMAGE_HEIGHT" --rate-hz "$FPS" \
  --gripper-type "$GRIPPER_TYPE" --action-source "$ACTION_SOURCE" \
  --action-lookahead-steps "$ACTION_LOOKAHEAD_STEPS" \
  --utc-date "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

physical-ai-lab d1-dataset-summary
```

Review every gate and fill the generated session notes with visual review,
resets, failures, and restarts. Final-still paths are in the external dataset's
episode/quality records. Provenance includes native/stored RGB image facts,
requested rate, incoming count/interval/rate evidence, achieved episode rates,
and configured/effective lookahead. Contradictory metadata or missing incoming
rate evidence blocks a physical summary.

Default freshness limits are 0.2 seconds of source age and ROS-header delay,
0.02 seconds of future-header offset, and 0.05 seconds of contemporaneous source
skew. The summary CLI exposes freshness thresholds and records them; do not
relax them retrospectively to conceal failed capture. Derived actions are
checked against their target frame, not treated as current-camera skew.
Candidate [limits](../../config/synria_limits.yaml) require operator
`verified_by` and `verified_on`. While empty, summaries say
"limits unverified by operator" and count zero qualifying D1 episodes.

## 6. Commit the evidence

Datasets, videos, and frame records stay outside git. Small generated session
artifacts belong under `reports/ludo_flagship/data/$SESSION/`; commit them with
the aggregate and timeline after review:

```bash
git add "reports/ludo_flagship/data/$SESSION" \
  reports/ludo_flagship/data/D1_dataset_summary.json \
  docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria D1 session $SESSION"
```

Resumption updates the same committed session record rather than counting old
episodes again. No delivery stage changes without qualifying committed evidence.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

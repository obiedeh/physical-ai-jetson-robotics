# D1 physical demonstration dataset protocol

Status: **implemented, unmeasured** recording and quality-gate path. D1 remains
**planned** until at least 100 qualifying physical Synria/Alicia-D episodes and
their committed summaries exist.

## Task

Each episode begins with one Ludo token upright in square A. The operator uses
the existing Alicia-D leader/follower teleoperation to pick up the token, move
it to square B, place it upright within the square boundary, release it, and
retract the follower arm clear of the board. The recording process observes
the existing teleoperation; it does not mediate or replace it.

In the operator-confirmed wiring, the leader drives the follower through a
hardware sync cable and is not connected to the PC. The PC observes only the
follower's `/joint_states`. The default `next_state` action is the following
sampled follower position/gripper state, a proxy rather than a directly read
leader command. This matches the operator-reported vendor recording convention
for this wiring; no physical recording or vendor runtime was tested here.

Direct `leader` actions are optional when the leader is additionally connected
to the PC by USB. Use the isolated read-only launch described in the
[runbook](D1_OPERATOR_RUNBOOK.md), explicitly select `--action-source leader`,
and keep its dataset separate. Existing hardware-sync teleoperation is unchanged.

Episodes target 20–30 seconds and stop at a hard 30-second cap. The C10 wrist
camera records `observation.images.wrist`; one fixed USB webcam records
`observation.images.front`. Every episode stores a timestamp-linked final
front-camera still.

## Operator labels

Mark `success` only when all of these are visually true in the final state:

1. The intended token moved from square A to square B.
2. The token is upright and entirely within square B.
3. The gripper released the token.
4. The follower arm retracted clear of the token and board.

Mark `failure` when any condition is false or uncertain. Use `discard` only
for a setup interruption, accidental key press, missing consent, or a recorder
fault that makes the episode unusable. Do not discard a genuine task failure.

Quality-valid failed demonstrations remain labeled and count toward the D1
episode total. Training selection may exclude them later, but it must not erase
them from the session record.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Scene randomisation

Pre-register square A and square B for every episode. Across a session, vary
reachable start and target squares, token colour, token orientation within the
start square, and lighting within safe operating bounds. Keep the board, fixed
camera, arm mount, and safety perimeter stationary. Record every reset or
deviation in `session_notes.md`.

## Pre-register the dataset settings

Declare the follower `STATE_SOURCE` as `standalone_driver` or `ros2_control`,
its absolute `FOLLOWER_TOPIC`, and all guarded command topics. This is an
operator declaration, not automatic identification of the state publisher.
The source must have arm-command writes disabled and leave torque unchanged
at startup. The operator confirms that standalone-driver default startup
releases torque; that default is prohibited. Use the
[runbook](D1_OPERATOR_RUNBOOK.md) for the unverified explicit-parameter
standalone candidate and operator-verified `ros2_control` preflight. Never start
a second follower or change the working hardware-sync teleoperation for recording.

Publisher-count checks always include `/joint_commands` and
`/policy_joint_targets`, plus configured additional absolute topics, before
dataset/camera startup and before each episode. Any publisher or graph failure
refuses recording. Discovery is eventually consistent; this snapshot is not
a hardware interlock or proof that later/non-topic writes cannot occur.
Episode and capture records retain the declared source, follower topic, and
guarded topics alongside measured incoming-rate evidence. Physical summaries
require consistent, complete declarations; changed source settings require a
separate dataset/session.

The contract is `synria_physical_v1`. Each dataset declares the gripper type
(`50mm` or `100mm`) and action source (default `next_state`, optional `leader`).
Set the runbook's `GRIPPER_TYPE` variable from the installed gripper; there is
no default. State contains six joint positions in radians and gripper in metres
using the driver convention, zero open and positive stroke closed. By default,
reported velocities are dropped. A deliberate velocity contract adds six
reported joint velocities; missing required values fail before recording.

Pre-register the requested integer `FPS`, `ACTION_LOOKAHEAD_STEPS` (k, default
1), `ACTION_SOURCE`, and stored `IMAGE_WIDTH`/`IMAGE_HEIGHT` (default 224 each).
The recorder requires explicit `--fps`. Vendor suggestions of 15 or 30 are not
source-rate evidence: every process counts incoming follower callbacks for
two seconds, checks freshness, and refuses a requested rate above the measured
rate before creating the dataset or opening cameras. The measurement and
achieved per-episode sample rate remain separate from the requested rate.

For `next_state`, action i uses follower state min(i+k, final frame); k may be
zero. Actual source timestamps are retained. The contract's nominal lookahead
is k/FPS, not measured latency, and clamping shortens the final offsets. Direct
`leader` actions are not shifted: configured k is stored but effective k and
nominal delay are zero. Never mix configured k, requested FPS, action source,
gripper, velocity mode, or image shape in one dataset. The writer rejects
incompatible resumes rather than silently changing the contract.

Images are converted BGR-to-RGB and resized at capture time. Provenance records
both actual native and stored resolutions, camera identities, requested rate,
per-run incoming-rate count/interval evidence, achieved sample rate, and
configured/effective lookahead. Camera identity or native size changes require
a new dataset. These stored software facts do not establish camera calibration
or real object success.

The final front-board still uses native RGB pixels from the same last accepted
sample, not an upscaled dataset frame or a later camera read. Only one native
front still is retained per pending episode, including through a save retry;
dataset frames keep their configured stored size. Capture/session provenance
records the still's native resolution and exact source timestamp. Legacy
records lacking still-resolution evidence remain explicitly unknown.

## Collection, review, and resumption

Use one process for many episodes: `start`, capture 20–30 seconds, `stop`, then
`success` or `failure` to label and save. The hard cap stops capture, not the
session. After saving, `start` begins the next episode. `retry` preserves the
same frames/label after a save error; `discard` explicitly abandons unusable
pending frames, and `quit` finalizes only once pending data is resolved.
Record discarded setup/recording faults and recovery problems in session notes;
genuine task failures remain labeled. Pending frames are memory-only, so do
not terminate a process expecting unsaved data to survive.

Smoke mode creates one disposable 20-second episode and deletes its temporary
dataset/still on exit. It is excluded from progress and provides neither a
live preview nor an automatic quality-gate report. Next record one retained
episode, quit, generate its summary, and manually inspect both views and the
final still before scaling collection. Follow the exact commands in the
[runbook](D1_OPERATOR_RUNBOOK.md); software tests are not hardware approval.

Resume the same dataset with the same `SESSION` and summary directory so the
cumulative record is replaced, not counted twice. Episode indices persist.
Keep session metadata/source revision and physical setup consistent; record
restarts and power transitions in notes. A changed contract or session identity
requires a new dataset and session. Incoming-rate measurements may differ on
resumed runs and are retained per episode, not overwritten by the latest run.

## Quality gates and qualifying counts

Limits in [`config/synria_limits.yaml`](../../config/synria_limits.yaml) remain
candidates until the operator fills `verified_by` and `verified_on`. While
empty, summaries carry "limits unverified by operator" and count zero
qualifying D1 episodes, even if the data is quality-valid.

Every episode must pass frame-count, finite-value, dual-camera presence,
state/action limit, timestamp-skew, source-staleness, duration,
gripper-dimensionality, and visual-sanity gates. Staleness uses original source
arrival times at sampling and compares ROS headers to ROS-clock arrivals, not
to a different clock epoch. Derived-action timestamps must match their target
follower frame. Missing timing or incoming-rate evidence cannot qualify.
The visual gate rejects black, frozen, and duplicated streams; operator review
still checks task visibility. Gate thresholds are recorded in provenance and
must not be relaxed retrospectively to hide failures. Preserve rejected data
and its reasons; quality-valid failed demonstrations may count as specified
above, but smoke never counts.

# D1 physical demonstration dataset protocol

Status: **implemented, unmeasured** recording and quality-gate path. D1 remains
**planned** until at least 100 qualifying physical Synria/Alicia-D episodes and
their committed summaries exist.

## Task

Prospective scope correction, 2026-10-08: the work-order task definition is now
roll-first: `die_into_cup`, `roll_and_dump`, and `cup_return` from the
[fixed-scene registry](../../config/synria_tasks.json). Token moves and any
goal-conditioning design are deferred until the roll works. This prospectively
replaces the work order's collection task, without changing stage thresholds.
Use one skill per dataset:

| Task ID | Collection protocol | Final pass rule |
|---|---|---|
| `die_into_cup` | [Die into cup](DIE_INTO_CUP_PROTOCOL.md) | Die inside the cup. |
| `roll_and_dump` | [Roll and dump](ROLL_AND_DUMP_PROTOCOL.md) | Die at rest on the tray; cup was not dropped. |
| `cup_return` | [Cup return](CUP_RETURN_PROTOCOL.md) | Cup upright on its mark. |

Each registry task needs operator timing and a prospective configured window;
all real windows are currently unset. The recorder requires `--task-id` and
selects the exact instruction and window from `--task-registry` (default
`config/synria_tasks.json`) before opening any source. Qualifying collection
refuses unconfigured timing; command-line duration/text overrides are not accepted.
The operator demonstrates the selected skill using the existing Alicia-D
leader/follower teleoperation. The recording process observes that path; it
does not mediate or replace it. Record each skill's start setup, reset and
observed final state using its protocol.

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

Episodes use the chosen task's configured minimum/maximum, with that maximum
also the hard cap; there is no qualifying duration default. The C10 wrist
camera records `observation.images.wrist`; one fixed USB webcam records
`observation.images.front`. Every episode stores a timestamp-linked final
front-camera still.

## Operator labels

Mark `success` only when the selected skill's pass rule is visually confirmed
and its timestamp-linked native front still is attached. For `roll_and_dump`,
the operator must also observe that the cup was not dropped during the episode;
the still alone cannot establish that history. Mark `failure` when any required
condition is false or uncertain. Use `discard` only
for a setup interruption, accidental key press, missing consent, or a recorder
fault that makes the episode unusable. Do not discard a genuine task failure.

Quality-valid failed demonstrations remain labeled and count toward the D1
episode total. Training selection may exclude them later, but it must not erase
them from the session record.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Scene randomisation

Use a fixed scene with a marked cup position, marked die start zone and fixed
landing tray. The skill protocols define each start state. Between episodes,
the die position may vary inside its marked start zone and its face up may vary;
these variations apply where the die begins outside the cup. For a loaded-cup
start, record the face used when loading without introducing a new cup or tray
location. Keep the arm mount, camera poses, cup mark, zone, tray and lighting
fixed within the declared session conditions. Record each reset and variation
in `session_notes.md`. Stop and declare a new scene before changing this setup;
do not introduce token targets or an unobserved goal.

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
gripper, velocity mode, image shape, task ID, task definition or episode window
in one dataset. The writer rejects
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

Use one process for many episodes: `start`, capture within the configured task window, `stop`, then
`success` or `failure` to label and save. The hard cap stops capture, not the
session. After saving, `start` begins the next episode. `retry` preserves the
same frames/label after a save error; `discard` explicitly abandons unusable
pending frames, and `quit` finalizes only once pending data is resolved.
Record discarded setup/recording faults and recovery problems in session notes;
genuine task failures remain labeled. Pending frames are memory-only, so do
not terminate a process expecting unsaved data to survive.

Smoke mode creates one disposable 20-second episode and deletes its temporary
dataset/still on exit. Its original task snapshot/hash remains unchanged, even
when the registry window is null. Separate `recording_purpose=disposable_smoke`
and explicit 20/20-second override metadata identify it as nonqualifying, not an
operator-timed task window. Smoke and qualifying data cannot mix or resume into
each other; disposable contracts are also refused by training and motion paths.
It is excluded from progress and provides neither a
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

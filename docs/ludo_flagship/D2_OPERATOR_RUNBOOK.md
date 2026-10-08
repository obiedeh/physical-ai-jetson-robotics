# D2 operator runbook

Software: **implemented, unmeasured**. D2: **planned**. All commands here are
for a future operator-authorized session. This implementation was tested only
with fakes and synthetic images.

The prospective [2026-10-08 decision](DECISIONS.md) makes the first D2 task
`die_into_cup`. Use its [collection protocol](DIE_INTO_CUP_PROTOCOL.md) and
complete, hash-register and commit the
[20-trial evaluation protocol](D2_EVAL_PROTOCOL.md) before running. The pass
rule is die inside the cup at the end with a camera still attached; the
threshold remains 14 of 20. Token moves and goal-conditioning design are deferred.
The fixed D2 path does not load board coordinates or send variable goal text.

## Preconditions and device bindings

Accept the [Phase 0A record](../../reports/synria/phase0a_safety_recovery.md) and
[first-safe-motion record](../../reports/synria/first_safe_motion.md). Complete
the operator fields in the [limits](../../config/synria_limits.yaml),
[policy safety configuration](../../config/synria_policy.json) and fixed scene.
Real task windows remain unset until operator timing and a prospective decision.
Use the trained task's configured registry and matching dataset/checkpoint;
no default task window or changing target is inferred.

Copy the [session configuration](../../config/synria_session.json) and
[ROS adapter configuration](../../config/synria_ros_adapter.json) to
operator-owned files. Complete the common source/provenance settings and
`d2_policy` entry: checkpoint,
trainer completion receipt, endpoint, per-policy response timeout and positive
command budget. Bind these to the committed protocol's complete task contract,
checkpoint hash, ordered scene schedule and cadence. Keep `max_attempts` equal
to the protocol. The endpoint metadata must match the receipt before sources
open; see the [serving guide](ACT_TRAINING_RUNBOOK.md). Legacy token schedules
cannot enable motion under the current roll-first contract. Confirm the fixed
scene/reset before each trial. Grade the native timestamp-linked front still,
then independently answer each funnel diagnostic yes/no/unknown. Funnel
observations do not determine object success; unknown values remain explicit.
The session's `adapter`
selects the included [read-only-first adapter](../../synria_lerobot/ros_adapter.py);
`adapter_config` points to the completed copy. No driver, bridge, controller,
launch file or teleoperation is started, stopped or reconfigured by this code.
The configured source must be the existing ros2_control stack, with both
operator-confirmed arm and gripper `FollowJointTrajectory` servers present
and no standalone driver node in any namespace. The arm action name and
physical gripper action joint/unit mapping are deliberately unconfigured:
do not infer them from the mock simulation controller. Fill and verify the
gripper joint name, units (`m` or `rad`), fully-open and fully-closed action
positions, `verified_by` and `verified_on` before enabling motion.

Configure absolute state, policy-target, armed-state and action names plus
distinct stable camera paths. The adapter strictly accepts `Joint1` through
`Joint6` and `Gripper`, using D1's zero-open, positive-closed metres; incompatible
state interfaces refuse startup. Set `state_has_velocity` to match the dataset:
false drops unused velocities, true requires all six. Preserve its
`action_source` and `action_lookahead_steps`, not an assumed default. Images
are captured as RGB at the configured stored size. State callbacks, camera
arrivals and ROS header delay must remain fresh. Preflight measures the incoming
state rate and records the result, configured endpoints and bridge timing.

The candidate bridge move time is 0.4 s, **unverified**. Explicitly configure
`bridge_move_time_s` from the deployed bridge and require session
`command_period_s` to be at least that long; a 10 Hz example does not override
the bridge duration. The adapter enforces the period between normal offers
and derives its own bounds from the measured state and configured speeds.

Use the selected trained policy's physical-contract HTTP endpoint. Requests
contain the contract, observation keys, task, reset flag, request id and host
observation timestamp. The shared [wire codec](../../synria_lerobot/policy_codec.py)
wraps requests as `{"codec":"synria_rgb_uint8_v1","payload":{...}}`. Wrist and
front images are raw uint8 RGB bytes encoded as base64, with explicit HWC
`shape`, `dtype`, `color_space` and `encoding`; JSON pixel lists are rejected.
Each image is at most 512×512×3 bytes, both decoded images at most 1.5 MiB,
and the entire encoded request at most 3 MiB. Responses are bounded to 1 MiB
and must be finite JSON objects; no executable deserialization is used.
Two 224×224 RGB views are covered by synthetic round-trip tests, not a
hardware/network performance measurement. Responses must echo `request_id` and
`observation_timestamp_s`, and contain seven absolute action floats plus
`inference_s`. Server-side monotonic clocks are not compared across hosts.
Set the required session `command_period_s` and the trained policy's explicit
`response_timeout_s`; the null template values are not usable defaults. There
is no shared response timeout: choose it for this policy and record it in the
policy ledger. Session provenance records the applied safety settings.
The client clamps absolute limits and derives each step bound from six joint
speed limits (rad/s) and one gripper speed limit (m/s), multiplied by that
command period. Candidate speeds remain unverified; both limits and policy
safety files need nonempty operator `verified_by` and `verified_on` before a
command sink can be created, even with `--enable-motion`. Motion-loop periods,
including roll phases, must match the session period. Do not use a shorter
period than the deployed command path supports. Missing/stale/invalid
responses request hold. Verify the actual controller hold and timeout behavior
before use; no torque-off fallback exists. Latency records contain server
inference time and local request-through-command-offer time. Separate
`request_encode_s` and `response_decode_s` fields measure local request byte
encoding and response JSON decoding using a monotonic performance clock.
They do not measure server-side image decoding. Request-local records avoid
cross-request timing reuse; unavailable timings, including unfinished timed-out
requests and non-HTTP fakes, are null rather than invented zero measurements.

## Pre-register and commit

Keep [frozen checkpoint diagnostics](CHECKPOINT_PROBE_RUNBOOK.md) separate from
D2. Freeze their held-out episodes and physical trials before training, and
pre-register the D2 policy-selection rule before viewing their results. They
must not be used to choose a D2 checkpoint retrospectively.

Choose fresh values for `SESSION`, `SESSION_CONFIG` (absolute path) and
`PROTOCOL` (a new repository-relative campaign protocol filename).

```bash
python3 -m pip install -e '.[dev,vision]'
cp docs/ludo_flagship/D2_EVAL_PROTOCOL.md "$PROTOCOL"
```

Fill the protocol operator fields and its 20-entry `scene_schedule` JSON list.
Each entry contains `trial_id`, `task_id: "die_into_cup"` and the fixed `scene`
fields documented in the template, with die position/face variations. The
committed protocol alone supplies the schedule; no legacy `d2_trials` copy is used.
Freeze the policy/checkpoint, retries, conditions and scene schedule before
collecting labels. The template threshold is 14 of 20; changing it requires
a prospective operator decision.

```bash
python3 -m synria_lerobot.evaluation register-protocol "$PROTOCOL"
git add "$PROTOCOL"
git commit -m "Register Synria evaluation protocol"
```

Copy the printed hash into `protocol_sha256` in the session configuration and
set `protocol` to the campaign file. The scorer verifies both committed bytes
and this hash. The hash excludes only its own value, avoiding self-reference.

## Validate, collect, score

For a future authorized session, follow this order: existing ros2_control
source; operator-validated bridge **disarmed**; policy server; read-only session
preflight. Use the site's validated ROS/recording environment with dependencies
already sourced. Do not launch another owner of the follower port. From the
repository root, run actual read-only graph/source preflight, with no command
publisher or action client created:

```bash
python3 -m synria_lerobot.sessions d2 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/eval/${SESSION}-preflight"
```

Unverified safety files may be inspected read-only; they cannot enable motion.
Preflight requires an observed actual `false` armed-state message, not a missing
message or an already-armed bridge. After reviewing the read-only result and
all physical prerequisites, leave the bridge disarmed and run:
Use a fresh motion output directory; never reuse the preflight directory.

```bash
python3 -m synria_lerobot.sessions d2 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/eval/$SESSION" --enable-motion
```

The process repeats read-only preflight, asks the operator to confirm the
leader hardware sync is **OFF**, records that confirmation with operator and
time, then asks for manual bridge arming. It waits for a new actual `true`
message after the confirmation. It never publishes an arming message. Only
then can it create its policy-target publisher and gripper action client.
The candidate armed topic is an observed operator command channel, not an
independent bridge acknowledgment. The operator must verify the configured
topic's meaning and the bridge's actual state; observing a Bool alone does not
prove that the bridge accepted it or that hardware is safe.
The policy target contains six arm joints only; gripper commands use the
separate verified mapping/action. It never publishes to `/joint_commands` or
sends a goal to the arm action server. Every offer rechecks graph ownership,
armed state, configuration verification and fresh measured state. Clamps use
the final measured snapshot before arm submission and refresh the gripper
snapshot before its separate submission. Normal command cadence is anchored
after successful arm publication, including eligible holds. Competing
policy publishers, any direct-command publisher, graph errors or missing
controller servers refuse motion. Its own policy endpoint is pinned by graph
identity, not excluded by subtracting one publisher from a count.

Graph discovery and armed-state callbacks are best-effort snapshots: they
cannot eliminate discovery delay or a change immediately after a check and
are **not a hardware safety interlock**. On timeout, fault, abort or cleanup,
an eligible hold sends at most one fresh measured six-joint arm target before
any bounded gripper-cancellation wait, then stops. It sends no gripper position
or torque-off command. A disarmed, stale,
unverified or graph-conflicted path sends no hold; use the bridge's separately
validated stop procedure. Accepted gripper goals owned by this session alone
are cancelled with bounded waits on failure/shutdown, including late acceptance.
Cancellation acceptance is not proof of physical stopping; unresolved outcomes
are reported and owned resources retained for cleanup. Verify this behavior on
hardware before use. Type `abort` at prompts or interrupt the process to abort.

Follow the committed scene schedule. Grade die inside the cup at the end from
the timestamp-linked native front still, then record the five independent
funnel observations as yes/no/unknown. They are not extra pass conditions.
Preserve every failed attempt. Confirm a reset before retrying. A fault,
closed input or uncertain scene requires the site's verified hold and operator
reconciliation. An incomplete trial set cannot pass.

The session writes EvalLog, provenance, frozen statistics, latency records and
end-power state. Attempt and terminal records preserve faults and interruptions
separately from the original object labels. Review the
first-try rate, funnel counts, trial result and threshold. To independently
re-score into a fresh file:

```bash
python3 -m synria_lerobot.evaluation score \
  --log "reports/ludo_flagship/eval/$SESSION/EvalLog.jsonl" \
  --protocol "$PROTOCOL" --output "/tmp/$SESSION-recheck.json"
```

Add exactly one policy row with eight fields in the
[policy ledger](POLICY_LEDGER.md): date, policy id, dataset/episode count/hash,
recipe/hyperparameters, checkpoint hash, evaluation artifact, result, decision.
The helper requires this policy's explicit `RESPONSE_TIMEOUT_S` and records it
in the recipe/hyperparameter cell; historical rows are unchanged.

```bash
python3 -m synria_lerobot.evaluation policy-row \
  --ledger docs/ludo_flagship/POLICY_LEDGER.md \
  --response-timeout-s "$RESPONSE_TIMEOUT_S" \
  "$UTC_DATE" "$POLICY_ID" "$DATA_DESCRIPTION" "$RECIPE" \
  "$CHECKPOINT_HASH" "$EVALUATION_ARTIFACT" "$RESULT" "$DECISION"
git add "reports/ludo_flagship/eval/$SESSION" docs/ludo_flagship/POLICY_LEDGER.md \
  docs/ludo_flagship/ACTIVITY_LOG.md docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria evaluation evidence"
```

Append the session event to ACTIVITY_LOG and timeline before the commit. Keep
raw datasets outside git; commit cited small stills and artifacts with their
hashes. Nothing automatically upgrades D2.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

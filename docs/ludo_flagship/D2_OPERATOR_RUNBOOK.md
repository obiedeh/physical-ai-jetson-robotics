# D2 operator runbook

Software: **implemented, unmeasured**. D2: **planned**. All commands here are
for a future operator-authorized session. This implementation was tested only
with fakes and synthetic images.

## Preconditions and device bindings

Accept the [Phase 0A record](../../reports/synria/phase0a_safety_recovery.md) and
[first-safe-motion record](../../reports/synria/first_safe_motion.md). Complete
the operator fields in the [limits](../../config/synria_limits.yaml),
[calibration](../../config/synria_board_calibration.json) and
[reachable squares](../../config/synria_reachable_squares.json). Coordinates
are XYZ metres in the named task frame. Track squares use `track:0` through
`track:51`; yard/home squares use colour, kind and index, such as `red:yard:0`
or `red:home:5`. All capture-return squares must also be reachable.

Copy the [session configuration](../../config/synria_session.json) to an
operator-owned configuration file and fill every field. Set `adapter` to an
installed `module:factory` implementing [SessionIO](../../synria_lerobot/sessions.py).
The site adapter binds the existing follower state and two camera sources,
operator completion/label/reset prompts, final still capture and the verified
controller command/hold path. No driver-specific adapter or default device
is assumed here. Its factory must create no command publisher;
`command_sink()` is the only publisher factory, invoked after motion gates.
Every observation uses the physical seven-value state and driver gripper
sense. Supply velocities only when reported by the driver.

Use the selected trained policy's physical-contract HTTP endpoint. Requests
contain the contract, observation keys, task, reset flag, request id and host
observation timestamp. Responses must echo `request_id` and
`observation_timestamp_s`, and contain seven absolute action floats plus
`inference_s`. Server-side monotonic clocks are not compared across hosts.
The client clamps absolute limits and per-step deltas. Missing/stale/invalid
responses request hold. Verify the actual controller hold and timeout behavior
before use; no torque-off fallback exists. Latency records contain server
inference time and local request-through-command-offer time.

## Pre-register and commit

Choose fresh values for `SESSION`, `SESSION_CONFIG` (absolute path) and
`PROTOCOL` (a new repository-relative campaign protocol filename).

```bash
python3 -m pip install -e '.[dev,vision]'
cp docs/ludo_flagship/D2_EVAL_PROTOCOL.md "$PROTOCOL"
```

Fill the protocol operator fields and its 20-entry `scene_schedule` JSON list.
Each entry contains `source`, `target`, `piece`, plus scene/randomisation
details. Copy that exact list into `d2_trials` in the session configuration.
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

From the repository root, validate configuration without loading the adapter:

```bash
python3 -m synria_lerobot.sessions d2 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/eval/$SESSION"
```

After explicit motion authorization, power-on/preflight and the site's
validated policy-server startup procedure, run:

```bash
python3 -m synria_lerobot.sessions d2 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/eval/$SESSION" --enable-motion
```

Follow the committed scene schedule. Grade reached/grasped/lifted/placed/
released and success/failure from the front still. Success requires the token
upright within the intended square, gripper released and arm retracted.
Preserve every failed attempt. Confirm a reset before retrying. A fault,
closed input or uncertain scene requires the site's verified hold and operator
reconciliation. An incomplete trial set cannot pass.

The session writes EvalLog, provenance, frozen statistics, latency records and
end-power state. Faults are recorded separately when present. Review the
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

```bash
python3 -m synria_lerobot.evaluation policy-row \
  --ledger docs/ludo_flagship/POLICY_LEDGER.md \
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

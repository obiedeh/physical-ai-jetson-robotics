# D2 die-into-cup evaluation pre-registration template

Software status: **implemented, unmeasured**. D2: **planned**.
Copy this template for an evaluation campaign, fill operator fields, run the
protocol registration command, and commit it before collecting any trials.
The SHA-256 covers this entire UTF-8 document with the protocol_sha256 value
cleared to an empty string. The EvalLog must repeat that hash. Registration
does not constitute operator approval or hardware evidence.
The [2026-10-08 operator decision](DECISIONS.md) prospectively replaces token
pick-and-place with `die_into_cup` for the first D2 evaluation. Token moves and
goal-conditioning design remain deferred until the roll works. The stage
threshold stays 14 successes in 20 trials.

```json
{
  "kind": "synria_fixed_skill_d2_v1",
  "task_id": "die_into_cup",
  "task_definition_sha256": "",
  "physical_contract": null,
  "policy_id": "",
  "checkpoint_content_sha256": "",
  "registered_by": "",
  "registered_on": "",
  "selection_rule": "",
  "randomisation_seed": null,
  "command_period_s": null,
  "response_timeout_s": null,
  "max_steps": null,
  "trials": 20,
  "success_threshold": 14,
  "max_attempts": 1,
  "scene_schedule": [],
  "protocol_sha256": ""
}
```

- Operator:
- UTC date:
- Dataset id/hash and committed summary:
- Device, controller, serving host and power:
- Accepted safety records and verified limits:
- Scene schedule and randomisation seed:
- Camera identities, settings and calibration:

Copy the full task-bound physical contract and task-definition hash from the
chosen finalized checkpoint receipt; it includes the operator-timed episode
window, action source and lookahead. Real registry windows are currently unset
and must be configured prospectively after task timing. Freeze the checkpoint,
command period, per-policy response timeout and command budget before trials.
Record an explicit policy-selection rule independent of diagnostic probe
results; probe curves and physical diagnostic attempts must not select D2's
checkpoint. No automatic metric-based selection is provided.

Pre-register 20 ordered entries in `scene_schedule`, each with unique
`trial_id`, `task_id: "die_into_cup"`, and `scene`. The scene fields are
`scene_id`, `cup_mark`, `die_start_zone`, `landing_tray`, `start_state`,
`die_position_in_zone`, and integer `die_face_up` (1–6). Randomise die position
inside the marked zone and face up using the declared seed; freeze the order
before registration. Keep the marked cup position, zone, tray, arm/camera
mounts, start setup and lighting fixed. Each trial begins with an empty cup on
its mark and the die inside the marked zone. These scene fields are operator
declarations. No token target or unobserved goal is accepted. The complete
schedule, actual scene/provenance and selected checkpoint must be committed
before the run; this blank template cannot run.

Pass rule: **die inside the cup at the end, camera still attached**. The operator
labels every trial success or failure and attaches its timestamp-linked native
front-camera still. Uncertain or occluded outcomes are failures. Release is a
demonstration instruction, not an additional object-success gate. Joint state
cannot prove die location. Separately record reached, grasped, lifted, placed
and released as yes/no/unknown diagnostics; they do not decide success and
missing observations are not invented zeros. All failures remain in the EvalLog. Retries
are individually recorded; first-try rate uses attempt one only. The trial
result is the final attempt; the threshold applies to all 20 trials.
The original label/still and raw object outcome are retained even when execution
faults or aborts. Only a completed execution with a success label counts as a
qualifying success. Fault/hold/abort status remains explicit, never rewritten as
an operator failure. Declined scenes and incomplete/aborted sessions remain
recorded and cannot pass. No additional all-funnel or fault-free gate is added.

The threshold is read from the committed configuration above. Change it only
prospectively with a new protocol and documented decision. An incomplete
evaluation cannot pass. Synthetic evaluations never advance D2.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

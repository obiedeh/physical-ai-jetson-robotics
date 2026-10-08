# Cup return demonstration protocol

Status: **implemented, unmeasured**. D1–D5 remain **planned**. This is the
prospective 2026-10-08 roll-first task definition; stage thresholds are unchanged.

Task ID: `cup_return`. The authoritative instruction, success rule, scene
requirements and episode window are in the
[task registry](../../config/synria_tasks.json). Follow the
[common D1 protocol](D1_DATASET_PROTOCOL.md) and
[operator runbook](D1_OPERATOR_RUNBOOK.md) for safety, capture and summaries.

## Scene and reset

Keep the cup return mark, marked die start zone and landing tray fixed. Begin
with the cup in the declared post-dump pose visible to the cameras and the die
at rest on the tray. Record that starting cup pose and the operator-verified
reset procedure in session provenance; it must match the preceding roll-and-dump
end setup. Keep the cup and its return mark visible in the fixed front view
and retain the C10 wrist view.

Die position inside the marked start zone and die face up may vary during the
earlier loading setup. They are not new grasp targets for this skill: the die
remains on the tray throughout cup return. Keep the starting cup pose, return
mark, tray, camera poses, arm mount and lighting fixed for this scene. No
variable return target or token move is part of this task.

## Demonstration and label

Return the cup from its declared post-dump pose to its fixed mark and release
it upright. Pass only when the cup is upright on its mark at the end,
confirmed by the operator and the timestamp-linked native-resolution front
still. Mark failure if the cup is off its mark, tipped, or the result is
uncertain. Do not infer success from joint state.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Timing and dataset identity

The real episode window is currently unset. Time this skill, prospectively
record the chosen minimum and maximum with rationale in
[DECISIONS.md](DECISIONS.md), and configure its registry entry before
qualifying collection. The recorder uses that window and hard cap; there is
no qualifying duration default. Store only this task in its dataset and do
not mix instruction, window or task-definition hashes.

Quality-valid failed demonstrations remain labeled, retained and eligible for
the D1 count under the common protocol; verified limits and all other gates
still apply. Discard only unusable setup or recording faults. Disposable smoke
never counts. A successful return alone does not establish a physical-roll
turn or advance any stage.

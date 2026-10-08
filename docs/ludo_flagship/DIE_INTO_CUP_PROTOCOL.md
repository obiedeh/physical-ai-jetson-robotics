# Die into cup demonstration protocol

Status: **implemented, unmeasured**. D1–D5 remain **planned**. This is the
prospective 2026-10-08 roll-first task definition; stage thresholds are unchanged.

Task ID: `die_into_cup`. The authoritative instruction, success rule, scene
requirements and episode window are in the
[task registry](../../config/synria_tasks.json). Follow the
[common D1 protocol](D1_DATASET_PROTOCOL.md) and
[operator runbook](D1_OPERATOR_RUNBOOK.md) for safety, capture and summaries.

## Scene and reset

Fix the cup at its visible marked position, a marked die start zone within
reach, and the landing tray at its declared location. Keep the cup, die and
gripper visible in the fixed front camera; retain the C10 wrist view. Record
the actual layout in session provenance. Begin with the die in the start zone
and the cup empty on its mark. Use the operator-verified reset procedure.

Only the die position inside the marked zone and its face up may vary between
episodes. Record both. Keep the cup mark, zone boundary, tray, camera poses,
arm mount and lighting fixed for this scene. No token move or variable target
is part of this skill.

## Demonstration and label

Pick the die from the marked start zone and release it inside the cup at its
marked location. Pass only when the die is inside the cup at the end and the
operator confirms this in the timestamp-linked native-resolution front still.
Mark failure if the die is outside or its location is uncertain.
Do not infer success from joint state. Inspect the recorded views if the cup
occludes the die; uncertain evidence is a failure.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Timing and dataset identity

The real episode window is currently unset. Time the task first, prospectively
record the chosen minimum and maximum with rationale in
[DECISIONS.md](DECISIONS.md), and configure this registry entry before
qualifying collection. The recorder uses that window and hard cap; there is
no qualifying duration default. Store only this task in its dataset and do
not mix instruction, window or task-definition hashes.

Quality-valid failed demonstrations remain labeled, retained and eligible for
the D1 count under the common protocol; verified limits and all other gates
still apply. Discard only unusable setup or recording faults. Disposable smoke
never counts. D2 trial registration and scoring are separate from D1 collection.

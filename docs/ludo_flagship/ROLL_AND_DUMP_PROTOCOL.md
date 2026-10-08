# Roll and dump demonstration protocol

Status: **implemented, unmeasured**. D1–D5 remain **planned**. This is the
prospective 2026-10-08 roll-first task definition; stage thresholds are unchanged.

Task ID: `roll_and_dump`. The authoritative instruction, success rule, scene
requirements and episode window are in the
[task registry](../../config/synria_tasks.json). Follow the
[common D1 protocol](D1_DATASET_PROTOCOL.md) and
[operator runbook](D1_OPERATOR_RUNBOOK.md) for safety, capture and summaries.

## Scene and reset

Fix the cup at its visible marked position, the marked die start zone and the
landing tray at their declared locations. Begin with the die inside the cup
and the cup on its mark. Keep the cup, gripper and tray visible throughout the
episode in the fixed front view; retain the C10 wrist view. Record the actual
layout and operator-verified loading/reset procedure in session provenance.

The die's face up when loaded may vary and must be recorded. Die position
variation inside the marked start zone belongs to the preceding die-into-cup
setup; this skill starts with a loaded cup. Keep the cup position, tray, zone,
camera poses, arm mount and lighting fixed. Do not vary the dump target.

## Demonstration and label

Pick the loaded cup from its mark, shake it and invert it over the fixed tray.
Pass only when the die is at rest on the tray at the end and the cup was not
dropped at any point. The operator observes the episode and confirms the final
die location with its timestamp-linked native-resolution front still. Retain
the episode views for the no-drop judgment; a final still cannot prove it.
The cup may remain held in the declared post-dump pose for the return skill.

Mark failure if the die is moving, is outside the tray, remains in the cup, the
cup was dropped, or any condition is uncertain. The die value does not affect
this pass rule and may later be entered by the operator. Perception is optional.

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
never counts. A successful isolated skill does not establish a D4 physical-roll
turn or advance any stage.

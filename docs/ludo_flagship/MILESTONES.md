# Ludo flagship milestones

D1–D5 remain **planned**. The D2–D5 software is **implemented, unmeasured**
as of 2026-10-08. Fake evaluation, turn, perception and game tests are software
validation and do not populate the physical milestone cells below. See the
[D2 runbook](D2_OPERATOR_RUNBOOK.md), [D3 runbook](D3_OPERATOR_RUNBOOK.md),
[D4 runbook](D4_OPERATOR_RUNBOOK.md) and [D5 runbook](D5_OPERATOR_RUNBOOK.md).

| Stage | Definition | Reached (UTC) | Artifact | Numbers |
|---|---|---|---|---|
| D1 | 100+ physical Synria/Alicia-D teleoperation episodes with provenance and labels | | | |
| D2 | Pre-registered 20-trial evaluation with >=14 operator-confirmed, camera-backed physical successes | | | |
| D3 | One SUCCESSFUL engine-planned, camera-backed and operator-confirmed physical Ludo turn | | | |
| D4 | Three consecutive SUCCESSFUL camera-backed, operator-confirmed physical-roll turns | | | |
| D5 | One successful full physical-board game to a winner, every turn scored | | | |

Intermediate milestones (dated when reached): first episode recorded; 10
episodes; 50 episodes; first policy served; first real grasp by a policy;
first scored turn. Yahboom/M3 artifacts cannot fill any milestone cell.
For D1 only, a quality-valid demonstration that failed the task may be
retained, labeled and counted toward the dataset episode total if the
pre-registered D1 dataset protocol permits it; its failed outcome remains
explicit. Failed D2 evaluations, D3 turns, D4 physical-roll turns and D5 games
are appended to the appropriate ledgers but do not fill or advance those
stage cells.

- 2026-10-08 — **Prospective scope, not achievement:** D1 collection is roll-first across `die_into_cup`, `roll_and_dump`, and `cup_return`; the first D2 target is die into cup in a fixed scene. Mark the cup location and die start zone and fix the dump tray. This prospectively changes the work-order task definition. Token moves and any goal-conditioning design are deferred until the roll works. The 100+ episode D1 target and pre-registered 14-of-20 D2 threshold are unchanged; no milestone cell is filled. Task windows remain unconfigured until operator timing and a prospective decision are recorded. See [the task registry](../../config/synria_tasks.json) and [the decision record](DECISIONS.md).

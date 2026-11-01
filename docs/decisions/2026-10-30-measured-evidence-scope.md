# Decision: scope of measured flagship evidence

Date: 2026-10-30. This record defines how the README, evidence index and roadmap
describe existing committed results. It changes no experiment, frozen policy,
acceptance threshold or measured record.

## Decision

Keep the existing measured, simulation-only ACT cup-lift classification and its
committed trial reference. Add the existing planning campaign, stationary SLAM
recording and recorded-data replay to the compact evidence index. A planner's
returned trajectory is planning evidence; a stationary recording or replay is
resource and map-output evidence. Neither establishes a learned physical skill,
driven navigation, loop closure, drift or aligned map accuracy.

Correct the README's earlier no-overlap summary to the saved-grid comparison in
the replay record. The maps have independent frames; their raw occupied-cell
comparison does not validate an alignment or navigation performance. Preserve
the record unchanged and name the prose correction explicitly.

Describe the exercised reset as pose/placement variation with a configurable
layout-draw range. `--randomize-scale` scales that draw range, not object geometry.
The opt-in light and piece-colour event terms are implemented but unmeasured. Do not turn
their existence into a randomization-robustness result, remove their documented
status, or claim physics randomization. Execution remains a separately frozen
simulation protocol for a later pass.

## Source records and code

- ACT simulation trials: [`trials_simcup17oct3_n045_step-000020000_20261018T192549Z.json`](../../reports/sim_policy/paired_seed2026/trials_simcup17oct3_n045_step-000020000_20261018T192549Z.json).
- Simulation planning: [`synria_moveit_planning.json`](../../reports/simulation/synria_moveit_planning.json).
- Stationary device recording: [`session.json`](../../reports/slam/sessions/2026-09-16_stationary/session.json).
- Recorded-data replay and raw-grid comparison: [`metrics.json`](../../reports/slam/sessions/2026-09-16_stationary/offline_replay_2026-10-30/metrics.json).
- Opt-in visual event configuration: [`env_cfg.py`](../../isaac/isaaclab_tasks/synria_pickplace/env_cfg.py), with event implementations in [`mdp.py`](../../isaac/isaaclab_tasks/synria_pickplace/mdp.py).

## Consequences

Existing measurements can be found without promoting unmeasured physical or
field capabilities. Documentation changes require committed source links and
full CPU-suite and hygiene checks. This change authorizes no simulation,
hardware execution, publication or change to frozen experimental decisions.

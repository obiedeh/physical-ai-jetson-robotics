# pick_cup_17oct3_margin

Prepared retrain of the 45-take row of the frozen first pick-cup dataset on the
plain recipe (seed 17, 20,000 steps, no weights, no augmentation) with
`action_limit_margin_deg` 1.0: every training joint target is clamped one
degree inside the operator bounds of `config/synria_limits.yaml` (verified
2026-10-17). The margin covers the largest sag the limits audit observed on this
dataset (Joint2 read 100.6 deg against its 100 deg stop; Joint3 -35.5 against
-35). It answers the holds counted in `../paired_seed2026`: 16 percent of all
steps for v2 and 638 for the corpus policy, each on a state a fraction past a
bound.

Status: **completed** 2026-10-22: 24 of 50 against 28; see `STUDY.md`, `FAILURE_MODES.md` and `../paired_seed2026/PAIRED.md`. The measurement is the paired
50-trial `pick_cup` campaign at placement seed 2026 with the gripper guard on,
against the plain 45-take row's 28 of 50. Simulation only; never physical
evidence.

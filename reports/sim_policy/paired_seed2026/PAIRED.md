# Paired campaigns at placement seed 2026

Simulation lane; never physical evidence. Every campaign below faced the same 50
seeded cup placements (seed 2026 plus the trial index), with the gripper guard
(close above 0.016 m after three consecutive requests) in the simulation safety
file, a 0.2 s command period and a 0.5 s response timeout. The Wilson 95%
interval is given with each rate; `FAILURE_MODES.md` beside this file sorts every
trial into a mode. "avg4" is the average of the last four checkpoints.

| checkpoint | successes of 50 | rate [95% interval] | closest approach, median |
|---|---|---|---|
| plain 35 takes, step 20000 | 0 | 0% [0, 7] | 16.1 cm |
| plain 40 takes, step 20000 | 4 | 8% [3, 19] | 7.0 cm |
| plain 45 takes, step 5000 | 0 | 0% [0, 7] | 14.2 cm |
| plain 45 takes, step 10000 | 1 | 2% [0, 10] | 7.4 cm |
| plain 45 takes, step 15000 | 6 | 12% [6, 24] | 8.1 cm |
| plain 45 takes, step 20000 | 28 | 56% [42, 69] | 1.8 cm |
| plain 45 takes, avg4 of 5000 to 20000 | 9 | 18% [10, 31] | 5.4 cm |
| v2 (chunk 15, six transforms incl. affine, ensembling 0.01), step 20000 | 0 | 0% [0, 7] | 22.2 cm |
| v2, avg4 | 0 | 0% [0, 7] | 23.3 cm |
| plain recipe, second seed (23), 35 takes, step 20000 | 4 | 8% [3, 19] | 5.9 cm |
| plain recipe, second seed (23), 40 takes, step 20000 | 5 | 10% [4, 21] | 3.0 cm |
| plain recipe, second seed (23), 45 takes, step 20000 | 25 | 50% [37, 63] | 1.8 cm |
| 60000-step run of 45 takes (seed 17), step 20000 | 21 | 42% [29, 56] | 2.3 cm |
| 60000-step run, step 40000 | 29 | 58% [44, 71] | 3.2 cm |
| 60000-step run, step 60000 | 26 | 52% [39, 65] | 4.5 cm |
| 60000-step run, avg4 of 45000 to 60000 | 25 | 50% [37, 63] | 3.8 cm |
| randomized takes weighted 3 to 1 (w3), 45 takes, step 20000 | 11 | 22% [13, 35] | 5.1 cm |
| w3, avg4 | 12 | 24% [14, 37] | 5.2 cm |
| approach recipe (w3 weighting, grasp-phase oversampling, state noise, action smoothing), 45 takes, step 15000, 20000 and avg4 | 0, 0, 0 | 0% [0, 7] | 20 cm |
| plain 45 takes, step 20000, gripper guard off, `reports/sim_policy/paired_seed2026_noguard` | 35 | 70% [56, 81] | at the cup |
| staged executor (scripted what-where-how-plan-execute, ground-truth poses), `reports/sim_policy/staged_seed2026` | 50 | 100% [93, 100] | at the cup |
| staged executor, `cup_pick_and_place` on the same placements (place score, 3 cm) | 50 | 100% [93, 100] | final distance median 0.4 cm, p90 0.7 cm |

What the pairing shows, with the caveat that one seed is one draw of placements:

- Training is far from converged at 20000 steps: the same run goes 0, 1, 6, 28
  across its four checkpoints. The 60000-step retrain in phase B tests that.
- Averaging the last four checkpoints hurts closed loop as it hurt held-out error:
  9 against 28. It is not a free lever here.
- The data-scaling curve is steeper than the earlier 20-trial campaigns said: 35
  takes 0, 40 takes 4, 45 takes 28; the earlier 7 of 20 for 35 takes was a
  placement draw, not a measurement.
- v2 is broken closed loop whatever the checkpoint, with the gripper never within
  22 cm of the cup, while holding the best held-out error of the study. The
  ablations queued behind the approach recipe separate ensembling from the rest;
  the affine transform is now refused by the trainer on the strategy decision.
- The 35-take checkpoint scored 7 of 20 on the earlier unseeded placements and 0
  of 50 here: twenty trials carry no decision.
- Phase C, 2026-10-19: the second seed reproduces the plain data-scaling curve
  (35 takes 4, 40 takes 5, 45 takes 25 of 50 against 0, 4 and 28), so the jump
  between 40 and 45 takes is a property of the data, not of one seed. Training
  longer does not move the 45-take plateau: the 60000-step run scores 21, 29 and
  26 at 20k, 40k and 60k, all inside one interval with the plain run's 28, and its
  own 20k checkpoint at 21 against the plain run's 28 on the same recipe and seed
  puts the run-to-run noise near seven trials. Weighting the randomized takes 3 to
  1 hurts, 11 and 12 against 25 to 28. Averaging is neutral on the long run (25
  against 26) and harmful on the plain run (9 against 28). At 45 takes the plain
  recipe sits near one lift in two, and no training lever tried so far lifts it;
  the levers left are data and the goal channels.
- Chain 2 and the first ablation, 2026-10-19: the combined approach recipe (all
  four data levers on the 45-take row) scores 0 of 50 at every checkpoint with
  the gripper 20 cm from the cup and the client holding 6 to 10 percent of steps,
  the same collapse as v2; the levers were bundled, so which one breaks the
  policy is unknown, and the next step is one lever at a time. The gripper guard
  does not explain the plain run's 28: with the guard off the same checkpoint on
  the same placements scores 35, inside one interval of 28, so the guard is at
  best neutral for a policy that already closes correctly.
- The staged executor, with the scene's own poses as its where, lifts the cup on
  every one of the same placements with no holds and no stalls, in about eight
  seconds a trial. That is the ceiling of the staged design and the measure of
  what the learned policy is being asked to learn: the geometry, which the planner
  does, and the squeeze, which the demonstrations show. It is scripted; it is not
  a learned policy and its where is ground truth, so the next measurement replaces
  the where with a camera locator.

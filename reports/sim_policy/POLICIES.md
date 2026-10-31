# Simulation policy register

Simulation lane only; none of these is physical evidence. Every score is a 50-trial
campaign on the paired placements (placement seed 2026 unless stated, 0.2 s command
period, 0.5 s response timeout, gripper guard on unless stated). Per-campaign detail
is in `paired_seed2026/PAIRED.md`; the trial reports sit in each study's folder.

Kept checkpoints are copied whole, with their checkpoint record, to
`~/synria-policies/<task>/<policy>-step<step>/` on the RTX 5090 workstation, outside
the training folder, so clearing `~/synria-sim-training` cannot remove them. Each copy
is checked against the record's content hash and serves with
`scripts/serve_synria_policy.py --checkpoint <copy>/checkpoint --checkpoint-record
<copy>/checkpoint_record.json --lane simulation`.

## Kept

| task | policy, step | data | score | why it is kept |
|---|---|---|---|---|
| `pick_cup` (3 cm lift) | `simcup17oct3_n045`, 20000 | 45 teleoperated takes | 28 of 50; 35 guard off; 23 on seed 2027 | best learned pick; held on a second seed and with the guard off |
| `pick_cup` | `simcup17oct3long_n045`, 40000 | the same 45 takes, 60000-step run | 29 of 50 | ties the champion inside the run-to-run noise; the long-training reference |
| `cup_pick_and_place` (place score) | `simplace02_n122`, 20000 | 122 staged-executor takes, seated grasp | 16 of 50 (15 traced); 38 lifted | best learned place |

The scripted staged executor (`staged_expert_v1`, ground-truth poses) places 50 of
50 and lifts 50 of 50 on the same placements. It is code, versioned in git, not a
checkpoint, and it is the ceiling a learned policy is measured against.

## Not kept

| policy, step | score | what it showed |
|---|---|---|
| `simcup17oct3_n035`, `_n040` (and seed 23) | 0 to 5 of 50 | fewer than 45 teleoperated takes do not lift |
| `simcup17oct3_n045` at 5000 to 15000, and the four-checkpoint average | 0 to 9 of 50 | 20000 steps is the point the 45-take recipe reaches |
| `simcup17oct3s23_n045`, 20000 | 25 of 50 | second training seed; reproduces the champion |
| `simcup17oct3m1_n045`, 20000 | 24 of 50 | action-limit margin; safe, inside noise |
| `simcup17oct3w3_n045`, 20000 | 11 and 12 of 50 | weighting randomized takes 3 to 1 hurts |
| `simcup17oct3appr_n045`, `simcup17oct3v2_n045` | 0 of 50 | bundled recipe changes broke the reach |
| `simplace01_n050`, `_n100`, `_n131` | 0 of 50 | scripted takes without recovery never lift |
| `simplace03_n130`, 20000 | 3 of 50; 40 lifted | takes with back-outs and a return to rest confuse the place |

## When to keep a policy

1. A new checkpoint replaces the kept one for its task only when, on the same 50
   paired placements and score, it wins with an exact McNemar p below 0.05, or scores
   at least as many and holds on a second placement seed.
2. The same recipe retrained differs by about seven trials in 50, so a gap under
   eight is a tie, and a tie keeps the incumbent.
3. The replaced policy stays in the store until its successor has held on a second
   seed; then it is listed under Not kept and its copy may be removed.

## When to add fresh teleoperated data

1. The data-scaling curve is still rising at the largest subset: 35, 40 and 45
   teleoperated takes lifted 0, 4 and 28, the steepest lever measured, and no
   training lever since has moved the 45-take plateau.
2. Two or more recipe changes in a row land inside the noise of the kept policy, as
   the margin, the 60000-step run and the weighting did.
3. The failures cluster in a situation the corpus never shows, such as a retry after
   a fingertip push. Then record that situation, with its cause visible to the
   cameras.

Add scripted executor takes instead when the behaviour is geometric and repeatable,
such as the approach, the seat and the carry. Leave out of any corpus a motion whose
trigger the policy cannot see: in `staged_place_03` the back-out fired on a pose
measurement the cameras do not show, and the policy learned to leave a cup it was
holding.

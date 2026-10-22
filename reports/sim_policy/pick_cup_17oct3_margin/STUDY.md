# Simulation data-scaling study `pick_cup_17oct3_margin`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `/home/oedeh/synria-sim-datasets/pick_cup_17oct3_frozen_2026-10-18/dataset`
- held-out episodes: [12, 24, 36, 44]
- episodes available for training: 45
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 45 | simcup17oct3m1_n045 | 0.0568 | 0.1132 | 0.0016 | 48% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/pick_cup_17oct3_margin/trials_simcup17oct3m1_n045_20261022T045922Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

# Simulation data-scaling study `pick_cup_17oct3_w3`

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
| 45 | simcup17oct3w3_n045 | 0.0655 | 0.1345 | 0.0015 | not measured |  |  |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

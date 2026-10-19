# Simulation data-scaling study `pick_cup_17oct3_seed23`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `/home/oedeh/synria-sim-datasets/pick_cup_17oct3_frozen_2026-10-18/dataset`
- held-out episodes: [12, 24, 36, 44]
- episodes available for training: 45
- seed 23, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 35 | simcup17oct3s23_n035 | 0.0704 | 0.1468 | 0.0017 | not measured |  |  |
| 40 | simcup17oct3s23_n040 | 0.0648 | 0.1421 | 0.0018 | not measured |  |  |
| 45 | simcup17oct3s23_n045 | 0.0657 | 0.1464 | 0.0014 | not measured |  |  |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

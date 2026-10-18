# Simulation data-scaling study `pick_cup_17oct3`

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
| 5 | simcup17oct3_n005 | 0.1218 | 0.2482 | 0.0038 | not measured |  |  |
| 10 | simcup17oct3_n010 | 0.1064 | 0.2083 | 0.0033 | not measured |  |  |
| 15 | simcup17oct3_n015 | 0.0933 | 0.1843 | 0.0020 | not measured |  |  |
| 20 | simcup17oct3_n020 | 0.0841 | 0.1679 | 0.0018 | not measured |  |  |
| 25 | simcup17oct3_n025 | 0.0712 | 0.1442 | 0.0018 | not measured |  |  |
| 30 | simcup17oct3_n030 | 0.0723 | 0.1469 | 0.0018 | not measured |  |  |
| 35 | simcup17oct3_n035 | 0.0663 | 0.1377 | 0.0018 | not measured |  |  |
| 40 | simcup17oct3_n040 | 0.0653 | 0.1391 | 0.0014 | not measured |  |  |
| 45 | simcup17oct3_n045 | 0.0591 | 0.1232 | 0.0017 | not measured |  |  |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

# Simulation data-scaling study `staged_place_02`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `/home/oedeh/synria-sim-datasets/staged_place_02/dataset`
- held-out episodes: [15, 31, 45, 61, 79, 99, 116, 135]
- episodes available for training: 122
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 122 | simplace02_n122 | 0.0375 | 0.0466 | 0.0010 | 32% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_02/trials_simplace02_n122_20261022T091708Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

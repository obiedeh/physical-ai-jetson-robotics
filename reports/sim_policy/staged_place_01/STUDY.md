# Simulation data-scaling study `staged_place_01`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `/home/oedeh/synria-sim-datasets/staged_place_01/dataset`
- held-out episodes: [16, 34, 49, 65, 80, 99, 114, 132]
- episodes available for training: 131
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 50 | simplace01_n050 | 0.0664 | 0.0804 | 0.0022 | 0% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_01/trials_simplace01_n050_20261019T122801Z.json` |
| 100 | simplace01_n100 | 0.0338 | 0.0445 | 0.0008 | 0% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_01/trials_simplace01_n100_20261019T125718Z.json` |
| 131 | simplace01_n131 | 0.0269 | 0.0331 | 0.0007 | 0% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_01/trials_simplace01_n131_20261019T132635Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

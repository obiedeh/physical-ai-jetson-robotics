# Simulation data-scaling study `staged_place_05`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `<home>/synria-sim-datasets/staged_place_05/dataset`
- held-out episodes: [33, 66, 100, 133, 166, 199, 233, 267]
- episodes available for training: 291
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 122 | simplace05_n122 | 0.0196 | 0.0319 | 0.0008 | 30% | 50 | `<home>/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_05/trials_simplace05_n122_20261030T061326Z.json` |
| 291 | simplace05_n291 | 0.0215 | 0.0364 | 0.0009 | 40% | 50 | `<home>/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_05/trials_simplace05_n291_20261030T075342Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

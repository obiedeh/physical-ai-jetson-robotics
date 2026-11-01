# Simulation data-scaling study `staged_place_04`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `<home>/synria-sim-datasets/staged_place_04/dataset`
- held-out episodes: [15, 32, 49, 66, 82, 96, 113, 135]
- episodes available for training: 111
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 111 | simplace04_n111 | 0.0360 | 0.0410 | 0.0009 | 6% | 50 | `<home>/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_04/trials_simplace04_n111_20261029T230754Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

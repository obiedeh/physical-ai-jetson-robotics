# Simulation data-scaling study `staged_place_03_trim`

Simulation lane; never physical evidence. Every run trains the same frozen
dataset with the same held-out episodes, seed, step count and architecture;
subsets are nested by recording order, so each row contains every row above it.
Held-out error is the trainer's own measure on the final checkpoint; the success
rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.

- dataset: `/home/oedeh/synria-sim-datasets/staged_place_03/dataset`
- held-out episodes: [17, 36, 53, 69, 87, 102, 118, 135]
- episodes available for training: 130
- seed 17, 20000 steps per run, status completed

| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) | gripper MAE (m) | sim success | trials | campaign report |
|---|---|---|---|---|---|---|---|
| 130 | simplace03trim_n130 | 0.0811 | 0.1454 | 0.0009 | 4% | 50 | `/home/oedeh/github/physical-ai-jetson-robotics/reports/sim_policy/staged_place_03_trim/trials_simplace03trim_n130_20261023T032324Z.json` |

Caveats: recording order is not random, since the operator's picks improved
across the session, so later episodes are not interchangeable with earlier ones;
one seed per size gives no variance estimate; a simulation success says nothing
about the physical arm.

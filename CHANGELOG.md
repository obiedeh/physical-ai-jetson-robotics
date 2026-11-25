# Changelog

- 2026-11-25: Measured eight training data loader workers on the 45-take cup recipe: training took 11 min 25 s instead of 42 min, but the policy lifted 9 of 50 on the seed-2026 placements against 27 of 50 for the original (p 0.0003). The batches the workers deliver are identical to those without workers (40 shuffled batches compared), so the gap comes from the training run, not the data. Workers stay off by default until a repeat run settles it. Records: `reports/sim_policy/pick_cup_17oct3/train_n045_w8/` and its trial report. Simulation lane only.

- 2026-11-25: The policy trained with 167 generated cup takes added (`mimiccup01_merged`) is now the kept `pick_cup` policy. On the second placement seed (2027) it lifted 44 of 50 against 26 of 50 for the 45-take policy re-scored in the same session (19 placements new-only, 1 old-only, exact McNemar p 0.00004); over both seeds 75 of 100 against 53. Register, README status row and evidence page updated. Simulation lane only.

- 2026-11-24: Scored the policy trained on the 45 teleoperated cup takes plus 167 generated ones (`mimiccup01_merged`, 20,000 steps) on the 50 paired placements at seed 2026: 31 of 50 lifts, against 27 of 50 for the 45-take baseline re-scored in the same session (28 of 50 in October). 13 placements lifted only by the new policy, 9 only by the baseline; exact McNemar p 0.52, so a tie under the register's rule and the kept policy stays. Median time to lift fell from 17.6 s to 13.4 s. Simulation lane only. Reports: `reports/sim_policy/mimic_cup_01/`.

- 2026-11-23: `scripts/merge_sim_datasets.py` now keeps each source video file as its own file instead of packing every episode into one long file per camera. The packed merge made each training sample about 5.7 times slower to decode (68 ms against 12 ms per sample with the PyAV reader), so a 20,000-step run took about 4 hours instead of 42 minutes with the GPU mostly idle. Re-merging the 250-episode cup dataset this way gave identical samples (zero difference in actions, states and both camera frames) at 12 ms per sample.

- 2026-11-23: Generated 200 cup takes in the Isaac scene from the teleoperated takes (placement seed 6026): 167 lifted the cup and are kept for training; 17 could not be solved for their placement and 16 slipped. Every slip came from two source takes (episode 47: 3 of 13 held; episode 45: 11 of 17); every other source held on all its moves. Merged with the 50-take dataset into 250 episodes (212 trainable: 45 teleoperated, 167 generated), registered probe set `mimic_cup_01_sim` (same four held-out takes) and a training config identical to the 45-take baseline except for the dataset. Records: `reports/sim_policy/mimic_cup_01/`.

- 2026-11-23: Add `scripts/merge_sim_datasets.py`: joins two simulation datasets with identical contracts for one training run (LeRobot aggregation for frames, videos and statistics; the three sidecar logs, final stills and exclusion logs renumbered), keeping the base dataset's episode indices so a frozen probe set's held-out takes stay the same, with tests. Checked by joining the 50-take cup dataset to itself: 100 episodes, episode 12 and its copy 62 identical in actions, states and both camera frames.

- 2026-11-23: The MimicGen-style recorder now writes generated takes under the teleoperated dataset's exact contract (no cup-pose goal channel, integer 15 Hz rate), so the two datasets can be merged for training; a two-take check matched contract and features.

- 2026-11-23: First scene run of the MimicGen-style generator: 10 takes on placement seed 5026 (clear of the evaluation seeds), all 10 lifted the cup past the 3 cm score (maximum lift 10.9 to 18.6 cm); moves of up to 19.2 degrees in bearing and 1.9 cm in radius from 8 different source takes; one take ended early after its lift at a wrist singularity; no command was clamped. Pilot only: the moves stay close to the source takes, nothing has been trained on the generated takes, and no policy result is claimed. Record: `reports/sim_policy/mimic_cup_pilot01/`.

- 2026-11-23: Add MimicGen-style demonstration generation in the Isaac scene: `synria_lerobot/mimic_generation.py` finds each teleoperated take's grasp, keeps its final approach, grasp and lift, moves that segment to a new cup placement (a turn about the base plus a radial shift), solves it back to joint targets and replays it; `scripts/record_mimic_demonstrations.py` records and scores the replays with the staged recorder, with tests. Isaac Lab Mimic itself was not used: it needs an Isaac Lab environment, and the scored scene is not one. Offline check on the 45-take cup dataset: 33 of the 45 successful takes outside the four held out usable as sources (the grasp detector found no grasp in the other 12); 27 of 30 random placements on the evaluation ring solved. Not yet run in the scene; nothing trained.

- 2026-11-18: Add a Ludo board-reading benchmark for cloud Gemini models: `isaac/scripts/render_ludo_board_bench.py` renders 30 engine-generated board states in Isaac Sim, `physical_ai_lab/gemini_robotics/board_bench.py` maps the model's pointed tokens to board squares and scores them, with tests. First run on rendered frames with no arm in view: `gemini-4-hydrogen` read 30 of 30 boards exactly at 672 px and at 224 px; `gemini-robotics-er-2-preview` read 29 of 30 at 672 px and 27 of 30 at 224 px. No physical board, move choice or arm motion tested. Records in `reports/gemini_robotics/` and `reports/ludo_board_bench/`.

- 2026-11-18: Add `physical_ai_lab/gemini_robotics/point_bench.py`, a pointing benchmark for cloud Gemini models on the 60 overhead cup frames, with tests. First run, same prompt for both models: `gemini-4-hydrogen` answered 51 of 60 (median 2.4 mm, max 4.6 mm; the other 9 were service errors), `gemini-robotics-er-2-preview` answered 59 of 60 (median 2.3 mm, one answer 187.6 mm off). Perception only; no board reading or arm motion tested. Records in `reports/gemini_robotics/`.

- 2026-11-14: Teleop console guide status changed from implemented, unmeasured to measured in simulation only: simulation-mode sessions on 17 and 18 October 2026 recorded 50 takes, 21,746 frames, with the real leader arm driving the Isaac Sim follower; an ACT policy trained from scratch on 45 of them lifted the cup on 28 of 50 seeded placements (25 of 50 on a second training seed, 0 of 50 at 35 demonstrations). No take has been recorded with the physical follower arm: physical mode is built but has not run on hardware. The README status row for the console says the same.

- 2026-11-14: CI: run the ROS 2 description package build on `ubuntu-24.04` instead of `ubuntu-latest`. ROS 2 Jazzy packages are built for Ubuntu 24.04 only, and the job failed with `Unable to locate package ros-jazzy-desktop` once the latest image moved on. No package or test change.

- 2026-11-03: Evidence page and NOT_CLAIMED state the learned placement results (0, 16 and 20 of 50 on seed 2026) and what they do not establish: simulation only, one placement seed, the lift-task policy not directly comparable.

- 2026-11-03: README states the learned pick-and-place results from 2026-10-19 to 2026-10-30 (0, 16 and 20 of 50 placements on seed 2026, recovery corpora 3 of 50) with their records, and lists the seed-2027 confirmation run as next.
- 2026-11-02: Restore 15 package version numbers (TensorRT, cuRAND, pip requirement lines) that the private-address scrub had replaced with `<lan-address>`, from the pre-scrub originals; the hygiene check now tells four-part package versions from private addresses, with tests. Real addresses stay replaced.

- 2026-10-30: Keep study Markdown portable and normalize metadata before JSON escaping.

- 2026-10-30: Normalize native Windows home paths and keep fake process signals portable.

- 2026-10-30: Validate visual RNG scalar return types when optional tensor-framework types are unavailable.

- 2026-10-30: Enforce repository prose hygiene and preserve camera-probe failure diagnostics with bounded CPU CI threads.

## Unreleased

- 2026-10-30: Index committed simulation-planning and stationary SLAM evidence,
  correct the saved-grid overlap summary, and distinguish exercised reset
  variation from unmeasured visual event terms and absent physics randomization.

- 2026-10-30: Link flagship instructions to the shared repository hygiene policy and summarize board labels, private identifiers, hooks, immutable history and external handoff notes.

- 2026-10-30: Align simulation-description and action-clamp contract checks with the shipped model and exact tensor preservation. Normalize OpenCV region masks to their specified unsigned-byte type. Wait for owned browser shutdown before profile cleanup, with regression coverage for each contract.

# Changelog

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

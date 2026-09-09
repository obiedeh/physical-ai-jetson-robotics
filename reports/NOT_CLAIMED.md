# What Is Not Established

Updated 2026-09-07. Active work; these exclusions apply to the repository's
current public claims.

- Autonomous real-arm pick and place.
- Physical object-success ground truth. Yahboom arm state is command-derived,
  not observed joint feedback; a recorded command does not prove object success.
- TensorRT/PyTorch numerical parity. The Thor benchmark records no numerical
  deltas or tolerance-checked comparison.
- Sustained thermal and safety validation. The committed Thor benchmark and Orin
  day-one probes are short runs, not thermal soak or safety qualification.

## Ludo Results

1. Reported executor scores are Isaac Sim results. Kinematic attach and
   release-hold compensate for documented contact-model limits; they are not
   real-hardware manipulation evidence.
2. The default expert executor combines a PPO reaching policy with scripted
   grasp, placement, endgame and fallback behavior. The separate GR00T
   evaluation lane is not the policy behind the expert aggregate.
3. Earlier fixed-board sessions moved one token per color. The ledger later
   records a seed-5 capture with a two-command plan. That single exercised path
   does not establish general multi-token play or physical capture reliability.
4. The benchmarked sessions used game-engine RNG rolls. Later executor options
   include simulated dice manipulation and face reading. Those remain
   simulation and do not establish physical dice perception.
5. The selected seeds 10-12 block uses fixed board pose and a cup object, without
   distractor or lighting/pose-randomized evaluation. Its scoring rule and
   configuration are not pooled with the GR00T lane or earlier configurations.
6. GR00T eval01 is **0/20 corrected**, not the raw 1/20. The untouched cup
   satisfied the old geometric rule; the current policy lane requires a grasp.
   Its target-blind placement result is not a fair goal-conditioned comparison.

## Evidence

- [Thor benchmark and missing parity record](thor_trt_benchmark/thor_trt_benchmark.json)
- [Orin short-run probes](jetson/yahboom_day_one/day_one_smoke.json)
- [GR00T corrected sidecar](ludo_groot17_eval01/session_summary_corrected.json)
- [Experiment ledger](../docs/SYNRIA_EXPERIMENT_LEDGER.md)
- [Executor](../isaac/scripts/ludo_turn_executor.py)

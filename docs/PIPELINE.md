# Game-Playing Pipeline

Updated 2026-09-07. The implemented executor is an Isaac Sim pipeline. Thor
inference and the Yahboom hardware recording path are separate recorded paths;
an autonomous physical Synria game-playing pipeline is not established.

## Implemented Components

- `game_core/` and `ludo_engine/` emit game commands. Chess uses python-chess;
  checkers remains a stub.
- `isaac/scripts/ludo_turn_executor.py` normally loads an RSL-RL PPO checkpoint
  for parts of approach/carry, with scripted grasp, placement and fallback
  behavior. Kinematic attach and release-hold are simulator interventions.
- `--policy-port` selects the separately served GR00T evaluation lane.
  It is not the policy behind the expert aggregate.
- GR00T post-training uses external source/checkpoint environments. The Thor
  [eager/TRT benchmark](../reports/thor_trt_benchmark/thor_trt_benchmark.json)
  measures inference, not a physical robot control loop.
- The verified real-board recording path is Yahboom on Orin with an RTX keyboard
  client, implemented in `robots/lerobot_robot_rosmaster_m3pro/`.
  It is not evidence of autonomous Alicia-D manipulation.
- Gemini perception/phase-gate experiments have separate ledgers and runs;
  their results are not interchangeable with executor placement scores.

Architecture views: [system](diagrams/system-architecture.mmd),
[deployment](diagrams/deployment-view.mmd), [runtime](diagrams/runtime-flow.mmd),
[data flow](diagrams/data-flow.mmd).

## Evidence and Reproduction

Runs emit provenance, per-attempt `turns.jsonl` and session summaries.
The executor uses numbered files to preserve repeated provenance/summary writes.
`isaac/scripts/ludo_stats.py` aggregates selected records; choose a homogeneous
campaign explicitly. External vendor inputs, game-table scenes, checkpoints and
engines are required. One-command reproduction of every historical stage is not
established after asset removal.

## Results and Corrections

The post-fix seeds 10-12 block records 35/36 turns with one retry, 32/36 first
tries, and successful first-try error p50 19.2 mm / p95 24.6 mm.
[Separate frozen aggregate](../reports/ludo_stats_frozen/stats_2026-08-20T050701Z_reports_ludo_soak_s1%5B012%5D.json).
It is not pooled here with the pre-fix configuration. The earlier 8-35 mm
description was withdrawn because 35 mm was a scoring threshold.

GR00T eval01 is **0/20 corrected**: the raw apparent success never grasped the
cup. The current GR00T score adds a grasp requirement; do not pool it with
earlier geometric-only scores. The placement goal was unobservable in that
target-blind run, so it is not a fair goal-conditioned comparison.
[Corrected sidecar](../reports/ludo_groot17_eval01/session_summary_corrected.json).

The **128 ms / 7.8 Hz TensorRT attribution is retracted**: that was a
torch.compile baseline whose TRT stage failed. The recovered Thor TensorRT
measurement is 101.6 ms median / 9.8 Hz. Numeric parity remains unrecorded.
[Correction](notes/instrumentation_pass_2026-08-19.md).

See the [README](../README.md) for conditions and [NOT_CLAIMED](../reports/NOT_CLAIMED.md)
for the limits. Raw reports and historical ledgers remain available.

# Experiment ledger — GR00T closed-loop reliability program

Mission: raise randomized full-task closed-loop success. Primary metric:
measured full completions over randomized episodes. Milestones: 25%/40 eps
(minimum), 50%/50 eps (strong). Never report success from loss alone.

Environment: Franka + Robotiq 2F-85 surrogate (FrankaCupEnvCfg), ground
cup r=0.02 h=0.04 m=50g, spawn (0.5,0)±0.08, fixed 900-step windows.
Scripted-expert reference: ~65% of windows convert (200-demo recording).
Detailed narrative history: docs/SYNRIA_GR00T_LEDGER.md.

## Known limitations (standing)

- L1: No train/val split exists for the 200-demo corpus — v4/v6x
  open-loop MAE was computed on TRAINING trajectories. Treat those
  numbers as fit measures, not generalization. Enforce episode-level
  splits for all future datasets/training (held-out spawn poses).
- L2: Frozen DiT (32 GB VRAM; NVIDIA full fine-tune floor 40 GB). Only
  projector/VLLN adapt. Cloud unfrozen package ready
  (reports/training/franka_cloud_package.tar.gz + docs/CLOUD_FINETUNE_FRANKA.md).
- L3: Closed-loop evals to date used one prompt; prompt variants untested.

## Experiment records

### E-v4 | 60 rand demos | eff.batch 1 | 5k steps
- Hypothesis: clean surrogate demos suffice for frozen-DiT fine-tune.
- Open-loop MAE 0.228 (train trajs; zeros 0.60). Closed-loop 0/24 rand.
- Failure (frame dumps): averaged pick motion, no visual retargeting;
  pre-shaped fingers at demo-matching 0.349, closed in air beside cup.
- Decision: control experiment on spawn variance (E-v5).

### E-v5 | 22 FIXED-spawn demos | eff.batch 1 | 5k steps
- Hypothesis: failure is spatial grounding, not skill execution.
- Open-loop MAE 0.212. Closed-loop 4 picks / 3 places over 32 (fixed).
- CONFIRMED: skill executes when no spatial generalization is required.

### E-v6 | 200 rand demos | eff.batch 1 | 5k steps
- Hypothesis: more diverse demos improve grounding.
- Open-loop MAE 0.570 = zeros baseline. Trained to the mean. Closed-loop
  skipped (no signal to evaluate).

### E-v6b | 200 rand demos | eff.batch 1 | 15k steps
- Hypothesis: v6 was under-trained (0.08 epoch).
- Open-loop MAE 0.808 — WORSE than baseline. Longer batch-1 training on
  diverse data degrades. Closed-loop skipped.

### E-v6c | 200 rand demos | eff.batch 8 (grad accum) | 5k opt steps
- Hypothesis: batch-1 gradient noise breaks learning on diverse corpora.
- Train loss 0.098. Open-loop MAE 0.0599 (10x below baseline; best).
- Closed-loop 1/32 randomized (3.1%, 95% CI ~0.6-15.7%) — first
  randomized completion. CONFIRMED batch size critical; open/closed gap
  remains dominant (drift/covariate shift or frozen-backbone limit).

### E-v6d | 200 rand demos | eff.batch 8 | 16k opt steps (RUNNING)
- Hypothesis: deeper batch-8 training reduces compounding error.
- Config: lr 1e-4 (cosine), frozen DiT/LLM/visual, save 8000/16000.
- Pending: open-loop (same 5 trajs for comparability — see L1),
  closed-loop >=40 randomized episodes via evaluate_stage_pipeline.py
  with stage ledger + failure distribution, isolated skill tests
  (takeover grasp/transport/placement), prompt variants.

## Evaluation protocol (fixed)

- evaluate_stage_pipeline.py: 900-step windows, spawn ±0.08 (or fixed),
  EXEC_HORIZON 8, safety clamps on, JSONL ledger per episode with
  measured stage signals and one primary failure label.
- analyze_failures.py: success rate + Wilson 95% CI + failure
  distribution + stage medians; writes reports/stage_metrics.csv.
- Isolated skills: scripted GT-based setup then policy takeover at
  grasp / transport / placement; policy budget 400 steps.

### E-v6d | 200 rand demos | eff.batch 8 | 16k opt steps — COMPLETE
- Train loss 0.061 (4.0 h). Open-loop MAE 0.0355 (train trajs).
- Full task 0/40 (CI 0-8.8%) — WORSE than v6c 1/32 at 3x fit quality:
  Goal-8 overfitting signature (schedule memorized, contingency not).
- Stage results: approach SOLVED (median min dxy 0.019); grasp isolation
  4/24; transport isolation 0/11 valid (7/11 re-opened mid-air);
  placement isolation 5/24.
- Dominant failure: gripper channel ignores context (premature close
  70%, mid-carry re-open) — full analysis reports/failure_analysis.md.
- Running: E-latch (eval-time gripper latch shim, 40 eps) testing H1.

### E-eval-knobs (2026-07-31): all eval-time levers measured at ZERO
- checkpoint-16000: 0/40. checkpoint-8000: 0/40 (overfit axis: no rescue).
- gripper latch: never armed (0 lifts/16). gripper override (scripted
  gripper, policy arm): 0/40 — arm flies through the grasp pose (36/40
  aligned at grasp height but no station-keeping).
- exec_horizon 8->2: 0/40. Replan frequency: no rescue.
- CONCLUSION: deficit is IN the trained policy (conditional-dwell /
  contingency), not the evaluation configuration. Proceeding to the
  data lever: E-v7 dwell-heavy corpus (40 dwell demos + 200 base).

### E-v7 | 226 eps (200 base + 26 dwell) | eff.batch 8 | 6k opt steps — COMPLETE
- Train 91 min, loss 0.093. Closed-loop 0/40 (stop criterion met).
- Distribution shift vs v6d (measurable but insufficient): premature
  closes 28->23, closes-near-cup 12->17, failed_grasp 11->16, lifts 2->2,
  one object_slip (first mid-air slip vs pure drops). The dwell data
  moved gripper timing in the right direction without producing task
  success.

## VERIFIED BLOCKER (2026-07-31, per operating-mode condition 2)

Every locally-available lever has been measured at or near zero
randomized closed-loop success while open-loop imitation is excellent:

  lever                     | evidence
  --------------------------|---------------------------------------
  demo quality              | v3-era fix (clean expert)
  demo quantity 60->200->226| v4 0/24, v6c 1/32, v6d 0/40, v7 0/40
  effective batch 1->8      | v6 baseline-collapse -> v6c best-fit
  training depth 5k->16k    | v6c 1/32 -> v6d 0/40 (overfits schedule)
  checkpoint selection      | 8000: 0/40
  replan frequency 8->2     | 0/40
  gripper shims (latch/override) | never arms / arm flies through
  targeted dwell data       | v7 0/40 (distribution nudged only)

The consistent residual: the policy cannot produce STATE-CONDITIONED
CONTINGENT behavior (station-keep at the grasp pose, commit and hold
the gripper) despite that behavior being well-represented in the data
and despite near-perfect open-loop action reproduction (MAE 0.036-0.06
vs 0.57 baseline). With tuning restricted to projector/VLLN (32 GB
frozen-DiT constraint; NVIDIA's full fine-tune floor is 40 GB), the
visuomotor pathway that must couple scene state to action timing cannot
adapt. This is a hardware-forced architectural limitation, not a data
or protocol deficiency.

Remaining paths:
1. Cloud unfrozen-DiT fine-tune on 40 GB+ (package:
   reports/training/franka_cloud_package.tar.gz + updated corpus
   franka_ds_v3 available; docs/CLOUD_FINETUNE_FRANKA.md; batch >= 8
   proven essential). Requires user-provided cloud access.
2. Local hardware upgrade to >= 40 GB VRAM.
Reference ceiling: scripted expert ~65% per window on the same task.

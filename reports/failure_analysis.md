# Failure analysis — v6d (checkpoint-16000, 200 rand demos, batch 8, 16k steps)

Date: 2026-07-31. Ledgers: reports/closed_loop_v6d.jsonl,
reports/skill_{grasp,transport,placement}_v6d.jsonl. Protocol:
evaluate_stage_pipeline.py (900-step windows, spawn ±0.08, safety
clamps; takeover setup uses ground truth — the policy under test never
sees GT).

## Observations (measured)

- Open-loop MAE 0.0355 on 5 training trajectories (v6c: 0.0599;
  zeros baseline 0.5702). [Caveat L1: training set — fit, not
  generalization.]
- Full task: 0/40 (95% CI 0–8.8%). v6c (5k steps, same recipe): 1/32.
- Failure distribution (full task): premature_gripper_close 28/40
  (70%), failed_grasp 11/40 (27.5%), transport_drift(+physics blowup)
  1/40 (2.5%).
- Approach quality: median closest hand-to-cup xy 0.019 m; many
  episodes reach <10 mm. (v4 frame dumps: never < ~50 mm.)
- Grasp isolation (policy from verified pregrasp): 4/24 = 16.7%.
  Median final finger 0.32 ≈ pre-shape — the policy frequently never
  commits to closing.
- Transport isolation: 0/24 overall; of 11 valid scripted handovers
  (setup_ok), 0 retained; 7/11 ended with fingers commanded back to
  0.25–0.31 (pre-shape) while the cup was mid-air.
- Placement isolation: 5/24 = 20.8%.
- Harness note: scripted takeover setup verified lift in only 11/24
  (synchronized batch servo weaker than the per-env expert) — transport
  numbers are conditioned on setup_ok.

## Interpretation

The v4-era spatial-grounding failure is resolved: the policy visually
retargets its approach to the randomized cup (medians ~2 cm). The
dominant remaining defect is the GRIPPER CHANNEL: its commands follow
the demos' marginal gripper schedule (long open/pre-shape phases, brief
close) rather than the visual/proprioceptive context — closing early on
approach, failing to commit at the pregrasp, and re-opening mid-carry.
Deeper training (v6d vs v6c) improved action fit but slightly worsened
task success (0/40 vs 1/32) — consistent with Goal-8 overfitting to the
schedule rather than the contingency.

## Hypotheses (open)

- H1: With the gripper channel externally corrected, the arm policy is
  already good enough for ≥10% full-task success. → tested by the
  --gripper_latch shim (running).
- H2: The demos under-represent the closed-holding state (top-down
  cameras also barely see the cup inside the gripper during carry);
  rebalancing (longer holds, more carry frames, or a gripper-state
  weighted loss) fixes retention.
- H3: checkpoint-8000 (mid-training) trades fit for less schedule
  overfit → better closed-loop than 16000.
- H4: Frozen-DiT ceiling — the visuomotor coupling for gripper decisions
  needs backbone adaptation (cloud arm).

## Decisions

- Smallest decisive test = H1 latch shim (running, 40 eps).
- Next after latch: H3 (checkpoint comparison — zero training cost)
  and/or H2 (targeted hold-heavy demo batch), ordered by latch outcome.

## Safety

Simulation only; joint-limit and gripper clamps enforced every step; one
physics blowup observed (cup launched, no damage semantics in sim);
classifier gains a collision/blowup guard next revision.

## Addendum: latch + override shim results (2026-07-31 afternoon)

- E-latch (hold 0.79 after genuine stall+lift): NEVER ARMED — 0 lifts in
  16 episodes; premature closes (14/16) gate everything upstream.
  Aborted; latch cannot test H1 as designed.
- E-override r1 (gripper fully scripted, trigger dxy<0.03 & z 0.17-0.21):
  scripted close never fired in 8 eps — motivated finer metrics.
- E-override r2 (trigger dxy<0.045 & z 0.16-0.22, 40 eps, new metrics):
  0/40 lifts DESPITE the scripted gripper. But: min_hand_z median 0.153
  (all 40 descend below 0.23; many overshoot below the 0.187 band) and
  36/40 achieve dxy<0.045 AT grasp height.
- INTERPRETATION (refined final): the arm policy approaches, descends,
  and crosses the correct grasp pose — but FLIES THROUGH it. It never
  station-keeps, so even a correctly-timed scripted close drags the
  fingers off the cup (the expert freezes the arm for the ~1 s close).
  The closed-loop deficit is a CONDITIONAL-DWELL failure in BOTH
  channels: the arm does not hold at the grasp/place poses, the gripper
  does not commit/hold its closure. The policy reproduces demo MOTION
  but not demo STATE-CONDITIONED PAUSES, despite those pauses being
  ~20-30% of demo frames. Consistent with frozen-DiT limits on
  visuomotor contingency, and possibly with observation-frequency
  aliasing (policy replans every 8 steps through a dwell that demos
  execute nearly statically).
- H1 is now REFUTED as "gripper-only": fixing the gripper channel alone
  is insufficient. Highest-value next steps: (a) checkpoint-8000
  comparison (running) for the overfit axis; (b) dwell-heavy targeted
  demos (longer explicit holds at pregrasp with zero arm motion) —
  attacks the deficit in data; (c) execution-horizon reduction
  (EXEC_HORIZON 8 -> 4 or 2; cheap eval-time experiment — more frequent
  replanning may let the policy self-correct into dwell); (d) cloud
  unfrozen-DiT (structural fix; package ready).

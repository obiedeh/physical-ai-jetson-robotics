# GR00T-1.7 Ludo closed-loop eval01 — findings & root cause (2026-08-20)

**Correction, 2026-09-07:** GR00T eval01 is **0/20 corrected**, not the raw
1/20 reported in the historical text below. The apparent success never grasped,
lifted or released the cup. See the [corrected sidecar](../../reports/ludo_groot17_eval01/session_summary_corrected.json).
The target-blind run is not a fair goal-conditioned placement comparison.
Historical raw tables and subsequent interpretations below are retained as the
record of the investigation, not current claims. The assertion that the only
success grasped/dropped the cup, and conclusions based on a 5%-versus-0% win,
are superseded. Differences in grasp counts do not alone establish the proposed
causal explanation about language conditioning.

## Result
Trained policy **ludo_groot17_v1** (LoRA on ludo_corpus01) evaluated closed-loop
in Isaac Sim over 20 random pick→place turns (serve_groot17 :5591, seed 300):

| metric | value |
|---|---|
| first-try success | **1 / 20 (5%)** |
| grasped | 9 / 20 |
| lifted | 7 / 20 |
| released | 9 / 20 |
| timed_out | 11 / 20 |
| placement err (mm) | min 30 · **p50 193** · p90 368 · max 476 |

Scripted expert baseline on the same task: **97.2%**.

## Root cause: the place target is UNOBSERVABLE in the training corpus
- The board exposes **46 reachable target squares**. Exact-target chance ≈ 1/46
  ≈ 2%. Observed 5% (1/20) is at chance for the small sample.
- The **only** success was `track42 → track43` — physically **adjacent** tracks
  (err 30 mm): the policy dropped the piece ~where it grasped and it happened to
  coincide with the target. Not target-directed behaviour.
- Placement error is uncorrelated with the commanded target (**p50 193 mm** —
  half a board). The grasp funnel (9/20) is healthy; the collapse is entirely in
  **placement**.
- Why: `ludo_turn_executor.py` corpus mode wrote a **constant** language string
  (`"pick up the cup and place it on the target square"`) for every turn. The
  place square lived only in the trial-manager state — never in the observation
  or the instruction. GR00T-1.7 is language-conditioned; with a constant
  instruction it has **no signal for which square is the target**, so it grasps
  (target-independent) and then places at chance.

This reproduces the corpus-design gap flagged during corpus01 generation
(memory obs 7542): constant instruction ⇒ target not observable.

## Fix (committed)
`ludo_turn_executor.py` corpus mode now renders a **target-observable**
instruction per turn: when `--instruction` contains `{pick}`/`{place}`, it
expands to the actual track indices (matching the eval-mode template) and
records `target_observable=true` + the rendered instruction in the manifest.
Backward-compatible: without a placeholder it falls back to the constant string.

```
--instruction "pick up the cup and place it on track {place}"
  -> per turn: "...place it on track 43", "...track 27", "...track 45", ...
```

## Next experiment (controlled)
Generate **ludo_corpus03_tobs**: identical pipeline/size to corpus01, the ONLY
change being the target-observable instruction. Then retrain the same LoRA and
re-eval closed-loop. This isolates the instruction variable and answers: *does
naming the target square recover placement?*

**Caveat to watch:** ~127–150 episodes across 46 targets is sparse (≈3 ex/target)
for learning 46-way language→position grounding. If placement improves but
plateaus, the follow-up is a denser target encoding — appending the target (x,y)
to the state vector (most sample-efficient for GR00T's proprio input) or
restricting the target set for a first proof — rather than more constant data.

---

## Update 2026-08-21 — target-observable retrain pipeline (v2_tobs)

corpus03_tobs generated (150 eps, all `target_observable=True`, 44 distinct
place targets) and taken through the full pipeline:
1. **convert** — `convert_synria_lerobot.py` → LeRobot v2.1 (150 eps, 626,782
   frames, 8D state/action, 44 distinct tasks preserved from the manifest).
2. **stats + migrate** — `gen_episodes_stats_v21.py` then
   `convert_dataset_v21_to_v30.py` → v3.0 in place. Load-verified:
   frame0 task = `"pick up the cup and place it on track 29"` (the fix survives).
3. **train** — `ludo_groot17_v2_tobs`, EXACT v1 recipe (groot on gr00t_n17_base,
   `--peft.method_type=LORA --peft.r=16 --peft.target_modules=all-linear`, 20k
   steps, bs8, seed1000, `--tolerance_s=0.001`, run outside repo). Only the
   dataset differs from v1 → clean controlled experiment. 39M trainable params,
   started at loss ~1.285.
   - **Gotcha:** reusing a finished checkpoint's `train_config.json` via
     `--config_path` fails (`use_peft=True` → tries to LOAD an adapter that
     isn't there). Must use explicit `--peft.*` args to CREATE a fresh LoRA.

NEXT (auto, on train completion): serve v2_tobs via serve_groot17.py → closed-
loop eval (same 20-turn seed-300 protocol) → compare first-try % and placement
err (mm) vs eval01 (5%, p50 193mm). Hypothesis: naming the target recovers
placement.

---

## Update 2 (2026-08-21) — eval02 result: target-observable did NOT help (clean A/B)

**Correction to Update 1's framing:** v1 and v2 trained to the SAME loss
(v1 last=0.11, v2 last=0.116) — the earlier "v2 overfit to 0.116 vs v1 1.08"
was a mistake (compared v2 final to v1 step-800). Fitting is identical; the gap
is closed-loop generalization, not overfitting.

**Clean A/B: 20/20 identical scenarios (seed 300).**

| metric      | v1 constant | v2 target-observable |
|-------------|-------------|----------------------|
| success     | 1/20 (5%)   | **0/20 (0%)**        |
| grasped     | 9           | 3                    |
| lifted      | 7           | 1                    |
| released    | 9           | 3                    |
| timed_out   | 11          | 17                   |
| place err p50 | 193 mm    | 161 mm               |

**Result: naming the target in language did NOT recover placement — it made the
policy worse, most visibly on grasping (9→3) and lifting (7→1), which are
target-INDEPENDENT.** Both models fit training equally well (~0.11), so this is
a robustness/generalization regression, not a fit problem.

**Interpretation.** At ~150 demos across 44 target tasks (~3.4 demos/target), the
added per-target language conditioning is under-determined: the model must now
attend to a language token it has only ~3 examples of, and that burden degraded
the *whole* closed-loop policy rather than sharpening placement. Neither variant
is usable (5% / 0%); the dominant bottleneck at this scale is data/behavior
robustness, not target observability per se.

**Recommended next directions (need a decision — each is real compute):**
1. **Target-in-state (recommended):** append target (x,y) to the 8D proprio
   state → 10D. The policy regresses toward a numeric goal instead of grounding
   44-way language — far more sample-efficient. Requires regenerating the corpus
   with the target written into states.npy + retrain. Most principled, most
   distinctive as a portfolio contribution.
2. **Dense small target set:** restrict to ~8 targets so ~19 demos/target — a
   tractable proof that a target signal helps when it's actually learnable.
3. **Scale demos:** 600+ episodes for real per-target coverage (~40h generation).
4. **Bank as a documented negative result** and pivot compute (e.g. to the M3 Pro
   real-robot leg).

---

## Update 3 (2026-08-21) — target-in-state: scoped & feasibility-verified (ready to run)

Decision (operator said "no preference" among next directions → proceed with the
recommended one, but checkpoint before GPU): target-in-state is the next
experiment. Feasibility VERIFIED; not yet launched (6-touchpoint change, better
run acknowledged than fired unattended after 2 negatives).

**Verified feasible cheaply (no 12h sim regen):**
- `ludo_engine.board.BoardGeometry(board_size=0.34, center_x=0.15, center_y=0.0)`
  imports standalone and gives each track's world (x,y):
  track29→(0.173,0.068), track25→(0.173,0.159), track44→(0.173,-0.045)…
  So the place target can be appended to existing `states.npy` OFFLINE.

**GR00T obs is grouped (not flat):** eval `groot_obs()` sends
`state={"arm":(6),"gripper":(2)}` + `language`; `modality.json` slices the flat
LeRobot `observation.state` into those groups (arm[0:6], gripper[6:8]).

**The 6 coordinated touchpoints (all must agree on target at state[8:10]):**
1. `isaac/isaaclab_tasks/synria_pickplace/gr00t/dataset.py::modality_json_spec` —
   add state group `"target":{"start":8,"end":10}` (make it opt-in / non-breaking).
2. State names 8→10 (add target_x, target_y).
3. Offline augmentation: for each corpus03 raw episode, read `place=["track",ib]`,
   compute `BoardGeometry.world_xy` → append constant (x,y) columns to
   `states.npy` → new raw dir `ludo_corpus03_tgt`.
4. `convert_synria_lerobot.py` — emit 10D state + target modality (auto-detect
   width, or a `--with-target` flag). Then gen_stats + v2.1→v3.0 as before.
5. `ludo_turn_executor.py::groot_obs` (eval) — add `"target": place_xy[None,None]`
   to the state dict, gated by a `--target-in-state` flag. `place_xy` = `cmd.place_xy`
   (already in scope as `place_t`). MUST match training convention exactly.
6. Retrain (same LoRA recipe, state now 10D) → eval (seed 300, `--target-in-state`,
   with or without the language target — test both) → compare vs v1 (5%) / v2 (0%).

Rationale over language grounding: the policy regresses toward a numeric goal in
proprio space (2 extra dims) instead of grounding 44-way language from ~3
demos/target — far more sample-efficient. If this ALSO fails at 150 demos, the
bottleneck is confirmed as data scale/task difficulty, motivating scale-up or
the M3 Pro real-robot pivot.

**Status: green-light ready.** On "run it" I execute touchpoints 1–6 with a
verify gate after each (dataset loads 10D, modality parses, eval obs shape
matches) before spending the ~3h train+eval.

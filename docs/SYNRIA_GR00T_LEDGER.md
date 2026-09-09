# Synria × GR00T Integration — Experiment Ledger

Track owner: GR00T session (separate from RL session). Branch:
`gr00t-integration`, worktree `~/github/physical-ai-jetson-robotics-gr00t`.
Mission: integrate GR00T N1.7 as an alternative policy for the Synria arm
and measure whether it improves the Ludo cup pick/move/place task vs the
RL baseline. Simulation only — no physical-arm control without explicit
user approval.

Isolation contract:
- Branch/worktree separate from RL session's `main` checkout (which has
  live uncommitted changes — never touch it).
- Venv `~/.venv/gr00t` (RL uses its own Isaac Lab venv).
- Checkpoints → `runs/gr00t_*`, logs → scratchpad + `reports/training/gr00t_*`.
- GPU shared (RTX 5090 32 GB): RL training uses ~6-9 GB; coordinate before
  any fine-tune (10-24 GB). Check `nvidia-smi` before every GPU phase.
- Never disable joint limits, collision, velocity/accel limits, workspace
  bounds, gripper-force limits, or e-stop paths.

## System facts (objective-1 inspection, 2026-07-28)

- Task: `Synria-Ludo-PickPlace-v0` (also chess/checkers variants) in
  `isaac/isaaclab_tasks/synria_pickplace/`; scene
  `ludotable_norobot_left.usda` (mirrored scenes dir).
- Action space (8 floats, absolute position targets):
  `[0..5]` Joint1..Joint6 targets in **radians**, clipped to URDF limits
  (J1 ±2.749, J2 ±2.0, J3 −0.5..π, J4 ±2.79, J5 ±1.57, J6 ±π);
  `[6]` left finger 0..0.025 m (0=closed), `[7]` right finger 0..−0.025 m.
  PD tracking: stiffness 400, damping 40 (guessed values, task #16).
- Observations (state-based policy group): joint_state (6 pos + 6 vel +
  gripper width), ee_pose (7), piece_pos (3), target ids, target_pos (4),
  trial_phase (5). Cameras NOT yet wired (TODO in env_cfg) — GR00T needs
  wrist + overhead RGB via a recorder-only variant. RL observation space
  must not change.
- Control rate: sim dt 1/60 s (Isaac Lab default, not overridden),
  decimation 2 → **30 Hz** control (verify at runtime).
- Episode: 30 s (=900 control steps). E8 kinematic attach-on-grasp glues
  the cup to the hand during carry (gripper-fidelity workaround) — demos
  recorded with it carry a fidelity caveat that applies to any comparison.
- Demonstrations: **none exist yet** in LeRobot format
  (`reports/training/synria_*_lerobot_demos/` absent). Recorder script:
  `isaac/scripts/record_synria_demos.py` (hybrid expert: policy picks +
  scripted release/retreat, attach enabled). Fixed eval harness:
  `isaac/scripts/eval_synria_sequence.py` (seed 123, 256 envs).
- RL baseline checkpoints: `reports/training/synria_chess_pickplace_v1`
  (E6_final.pt etc.). Ludo-task baseline numbers TBD from RL ledger.

## GR00T stack facts

- Repo: `~/Isaac-GR00T` (N1.7 era). Python 3.12 venv `~/.venv/gr00t`
  (repo supports 3.12 on dGPU; old scaffold docstrings claiming "3.10
  strict" are stale).
- Pins: **torch==2.9.0** + torchvision==0.24.0 (cu128 via uv sources),
  flash-attn==2.8.3, numpy==1.26.4. The handoff's `torch==2.10.0`
  instruction was WRONG (predates checking pyproject).
- Base model: `nvidia/GR00T-N1.7-3B`; VLM backbone
  `nvidia/Cosmos-Reason2-2B` is gated on HF — user token already has
  access (verified via API, HTTP 200 on both).
- Fine-tune entry: `gr00t/experiment/launch_finetune.py`, tyro CLI over
  `FinetuneConfig`: `--base-model-path --dataset-path --embodiment-tag
  --modality-config-path --output-dir --max-steps --global-batch-size
  --learning-rate ...`. No `--num-epochs`/`--batch-size` (scaffold drift).
- Embodiment: custom robots use tag `new_embodiment` (EmbodimentTag enum;
  arbitrary strings like `synria_6dof_v0` are rejected). Modality config
  is a **Python file** calling `register_modality_config(cfg,
  EmbodimentTag.NEW_EMBODIMENT)` passed via `--modality-config-path`
  (see `examples/SO100/so100_config.py`), plus `meta/modality.json`
  inside the LeRobot dataset.
- Scaffold drift found in `isaac/isaaclab_tasks/synria_pickplace/gr00t/`:
  wrong CLI flags, wrong embodiment tag, dict-based modality config, and
  `gr00t.data.dataset.LeRobotSingleDataset` no longer exists at that
  path. Needs rework against installed API (planned fix, not yet applied).

## Experiment log

### G0 — install attempt 1 (from handoff instructions) — FAILED
- `pip install torch==2.10.0` (cu128) then `pip install -e . --no-build-isolation`.
- torch 2.10.0+cu128 installed OK; editable install failed at
  **flash-attn 2.8.3 metadata generation**: missing `psutil` (+ numpy
  warning) in venv. Also torch 2.10.0 contradicts GR00T's ==2.9.0 pin.
- Kept: nothing (torch later replaced). Log: scratchpad/gr00t_install.log

### G1 — install attempt 2 (torch 2.9.0 + prebuilt flash-attn) — PARTIAL
- Hypothesis: honor the ==2.9.0 pin and skip flash-attn source build via
  the official wheel `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312`
  (no cu12+torch2.10 wheel exists; only cu13 for torch2.10).
- Result: torch 2.9.0+cu128 ✓, flash-attn 2.8.3 ✓. Editable install
  failed: `BackendUnavailable: Cannot import 'wheel_stub.buildapi'`
  (an NVIDIA sdist needs the wheel_stub build backend preinstalled when
  using --no-build-isolation). Log: scratchpad/gr00t_install2.log
- Kept: torch 2.9.0+cu128, torchvision 0.24.0+cu128, flash-attn 2.8.3.

### G2 — install attempt 3 (+wheel_stub) — PASSED
- `pip install wheel_stub` then retry `pip install -e . --no-build-isolation`.
- Result: `gr00t` imports; torch 2.9.0+cu128; flash-attn 2.8.3;
  `torch.cuda.is_available()` True on RTX 5090. **Gate 1 PASSED.**
- Weights cached: `nvidia/GR00T-N1.7-3B` (6.5 GB, 27 files) and gated
  backbone `nvidia/Cosmos-Reason2-2B` both downloaded to HF cache.
- Log: scratchpad/gr00t_install3.log
- Working install recipe (for reproduction): torch==2.9.0 +
  torchvision==0.24.0 from cu128 index → prebuilt wheel
  `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312` from GitHub releases
  → `pip install numpy psutil ninja packaging einops wheel wheel_stub` →
  `pip install -e ~/Isaac-GR00T --no-build-isolation`.

### G3 — scaffold rework against installed N1.7 API — PASSED
- Changes (branch `gr00t-integration`):
  - NEW `gr00t/modality_config.py` — SO100-style registration under
    `EmbodimentTag.NEW_EMBODIMENT`; state/action split arm[0:6] +
    gripper[6:8] (raw 8-dim = env action layout, 1:1 comparison later);
    arm actions RELATIVE, gripper ABSOLUTE; video wrist+overhead;
    16-step action horizon.
  - `embodiment.py`: tag `synria_6dof_v0` → `new_embodiment`; camera
    keys `wrist`/`overhead`.
  - `dataset.py`: dropped obsolete `LeRobotSingleDataset` helper (class
    no longer exists); added `modality_json_spec()`/`write_modality_json()`.
  - `finetune.py`: `--num-epochs`→`--max-steps`, `--batch-size`→
    `--global-batch-size`, added `--modality-config-path`.
- Validation: modality config import registers `new_embodiment` ✓;
  wrapper `--dry-run` emits command ✓; all 7 flags verified present in
  `launch_finetune --help` ✓.
- Still stale: `inference.py` (Gr00tPolicy signature unverified) — will
  be reworked at gate 7 (offline inference).

## Staged validation gates (must ALL pass before any long training)

1. [x] `import gr00t` + CUDA available in `~/.venv/gr00t` (G2)
2. [x] Dataset loads — 7-episode merged set (v10+v11+v12), 3250 frames;
       GR00T DatasetFactory built shards from it (smoke run log)
3. [x] Per-episode validation — 8-point validator PASS on all 7
4. [x] One-batch forward pass — smoke run completed forward; OOM was in
       loss.backward(), i.e. AFTER a successful forward
5. [x] Minimal training smoke test — 20 steps, loss ~1.2-1.3, sane grad
       norms. VRAM saga (G20): NVIDIA docs set a 40 GB minimum for
       fine-tuning; on the 32 GB 5090 the default config OOMs at ~30.8 GB
       REGARDLESS of batch size. Fits with `--no-tune-diffusion-model`
       (projector + new-embodiment heads trained, DiT backbone frozen).
       The real fine-tune inherits this constraint — note in comparison.
6. [x] Checkpoint save + reload — checkpoint-10/-20 written with model
       + processor + experiment_cfg; checkpoint-20 reloads cleanly
7. [x] Offline inference — checkpoint-20 accepts new_embodiment, runs
       open-loop on our episodes (per-limit action check to be repeated
       on the real fine-tuned checkpoint)
8. [x] Predicted-vs-recorded comparison — trajs 0-1: MSE 6.89,
       MAE 1.81 rad (smoke-level checkpoint; baseline number only)

ALL GATES PASSED (mechanics) — full fine-tune may launch once enough
demos are banked. GPU coordination: fine-tune must run SOLO on the GPU.

### G21 — pure-policy demo source (E10 checkpoint) — BREAKTHROUGH
- Expert banking (bank1-3, G19 included) plateaued at 1-2 demos/run —
  bank3's takeover rate even collapsed 4× (seed/dynamics variance).
- RL ledger showed E9/E10 STAGED cycle rates of 15-19% for the policy
  itself — the policy is now a better demonstrator than the expert.
- NEW `--pure_policy` recorder mode (no expert, attach event untouched)
  with `E10_partial_1900.pt`: **9 episodes in one 20k-step run** (~3×
  the best expert run; setdowns 15). Demo source switched.
- Banked: 20 raw episodes merged (v10-12 + bank1-4; mixed expert/pure-
  policy provenance, all post-G13 action fix) → `synria_cup_ds_v1`:
  20 episodes / 11,368 frames, 8-point validation PASS.
- bank5 result: **29 episodes** (60k steps). Total banked: 49.

### G22 — dataset v2 + REAL fine-tune launched
- `synria_cup_ds_v2`: 49 episodes / 28,805 frames, 8-point validation
  PASS. Provenance mix: 11 expert-driven (E8) + 38 pure-policy (E10).
  Durable copies: `reports/training/synria_cup_lerobot_demos` (+ raw).
- Fine-tune v1 LAUNCHED (solo GPU): N1.7-3B, new_embodiment, frozen
  DiT (32 GB constraint), batch 1, lr 1e-4, max-steps 5000, checkpoints
  every 1000. Logs: scratchpad/gr00t_finetune_v1.log + _mem.log
  (60 s GPU/RAM tracker per mission telemetry requirement).
- Fine-tune v1 result: loss 1.35→0.80 BUT open-loop MAE 1.79 rad ≈
  smoke checkpoint (1.81) and WORSE than predict-zeros (1.0). Loss
  improved without task-level improvement → stop-and-diagnose fired.

## FIXED-HARNESS COMPARISON (task #5) — measured

Setup: same task id, ground-plane scene, seed 123, 16 envs; GR00T
served from checkpoint `gr00t_synria_v2/checkpoint-5000` over ZeroMQ
into the Isaac harness (`isaac/scripts/eval_gr00t_sequence.py`), 16-step
action chunks, 8-step execution horizon; 3600 steps ≈ 64 ep-eq.

| stage      | RL baseline (E8 eval) | GR00T v2 (scratch starts) |
|------------|----------------------|---------------------------|
| grasp      | 0.746                | **0.000**                 |
| carry5s    | (transport 0.077)    | **0.000**                 |
| set-down   | 0.000                | **0.000**                 |
| full cycle | 0.000                | **0.000**                 |

Verdict (v1 data/compute budget): GR00T does NOT improve on the RL
baseline; it is strictly worse at the grasp stage. Neither method
completes the full task from scratch starts.

Ranked causes for the GR00T zeros: (1) distribution mismatch — 78% of
demos start zone-staged, eval is scratch; (2) 49 demos from a noisy
teacher (wrist joints std ~2 rad); (3) frozen DiT (32 GB < NVIDIA's
40 GB fine-tune floor); (4) smoke-scale training (5k steps, batch 1).

--staged eval result (demo distribution, 2700 steps): **also all
zeros** — cause (1) RULED OUT as primary. The checkpoint has not
learned a usable policy at all; open-loop MAE ≈ zeros-baseline
corroborates. Primary causes: (2)+(3)+(4).

## FINAL DIAGNOSIS & RECOMMENDATIONS (mission close-out, v1 budget)

Delivered: validated GR00T N1.7 integration (all 8 staged gates),
reusable record→convert→validate→train→serve→eval pipeline, offline-
tested checkpoints, and an evidence-based comparison. The comparison's
answer is negative at this budget: GR00T (49 demos, frozen DiT, 5k
steps) scores 0.000 on every task stage where the RL baseline grasps
0.746 — GR00T does NOT improve the Ludo-cup task today.

To give GR00T a fair shot (in order of expected impact):
1. MORE + BETTER DEMOS: hundreds of demos from a stronger teacher —
   wait for RL E10+ checkpoints with real place skill, or add GR00T-
   Mimic amplification (Path C) on the best 50 demos.
2. UNFREEZE THE ACTION DECODER: needs ≥40 GB VRAM (cloud A100/L40) or
   memory surgery (8-bit optimizer / gradient checkpointing not
   exposed by launch_finetune today).
3. LONGER TRAINING at real batch sizes (global 32+ per the recipe) —
   also cloud-bound on current hardware.
4. Teacher noise reduction: record demos with a deterministic-mode
   policy or filter high-jerk episodes.

## POST-CLOSEOUT: vision-scripted pick baseline (user-directed)

### G24 — stage 1+2 perception — PASS (first try)
- `diag_cup_perception.py`: overhead cam → HSV red segmentation →
  centroid+area → analytic top-down back-projection (4.42 mm/px at
  0.85 m) → env-local XY + diameter.
- Scored vs ground truth: 120/120 detections, XY error 0.90 cm mean
  (p95 1.36), diameter 40 mm ± 5 mm. Mapping: image −v → env +X,
  image −u → env +Y from camera centre (0.10, 0).
- Gripper sizing rule: open Ø+8 mm, close Ø−4 mm.

### Stage 3 plan (next): scripted motion to vision target
- Pivot J1 to bearing → DifferentialIKController pose-mode approach to
  pregrasp above vision XY (tool offset PREGRASP_CENTER_LOCAL
  [-0.031, 0, -0.053]) → descend → close to vision-sized width (attach
  engages) → lift → carry (hold 150 steps aloft, G9 commanded-target +
  G18 droop lessons) → LOWER → release (G14 gated detach + G15 exit).
- Prior IK attempt failed pre-attach on grasp physics (RL obs
  4807-4822); attach + the G-series choreography changes the odds.
- Doubles as a CLEAN demo generator for a GR00T rematch if it works.

### G25 — stage-3 motion campaign (probes 1-13) — POSITIONED, GRIP WALL
- Fixed in sequence, each from measured evidence: squeeze sizing
  (vision +5 mm bias), grasp-centre steering (tool-frame offset
  [-0.031,0,-0.053]), locked grasp attitude (6-DOF DLS; position-only
  IK tilted ~45°), full-stroke descent + frozen target (cup knocked
  over), tight 6 mm gate + servo-through-CLOSE (PD-sag integrator;
  residuals were config-dependent 10-23 mm, NOT a constant bias —
  bias-hack reverted). Wrist-cam FINE servo abandoned: boot-pose tool
  tilt projects ~12 cm and image axes rotate with tool yaw — needs a
  real camera-pose model.
- Final state: CLOSE-entry accuracy down to (1,6,19) mm; fingers close
  ON the cup. **Zero lifts** — the static scripted lift loses the cup
  instantly: the SAME gripper-contact wall that killed the RL session's
  IK expert (their ledger: "offset calibrated but lift still fails")
  and motivated E8 attach. Physical grasp-under-lift works only with
  dynamic corrections (the policy) or the attach bridge.
- Next: script-side attach-on-closure (the sanctioned workaround, same
  caveat class as RL training and all demos), then carry/set-down via
  the recorder choreography → full visible pick-move-place baseline.

### G27 — COMPLETE PICK-MOVE-PLACE — 4 placements/run
- Joint-space tail fixed the carry (attitude-locked DLS cannot
  translate laterally on this arm — carry height bled 0.33→0.09 m):
  CARRY = base-yaw ramp (0.5 rad) at the lift config, PLACE = grasp-
  height config rotated to the new bearing, RELEASE = drop hold + open
  + withdraw. No-squeeze hold post-grasp (G6 rattle).
- Probe 21 (8 envs × 3 episodes): 4 solid lifts, **4 completed
  pick-move-place cycles** (envs 0, 1, 6×2) — cup set down resting at
  the carry destination. Camera-only perception throughout.
- The vision-scripted baseline the user specified is DONE: see → know
  position → size gripper → pivot → approach → descend → grasp → lift
  → carry → place. Caveat: script kinematic hold bridges the known
  grip-physics wall (same class as E8 attach).

### G26 — VISION-SCRIPTED PICK WORKS — 3/8 (retry build)
- First success probe 16 (env0 DONE, cup held at +0.115 m); engage-at-
  entry (squeeze ejects by close-end: 117 mm flight) → 2/8; state-
  machine reset on episode reset (FAIL was permanent) → **3/8**.
- Full chain, camera-only: see → position (0.9 cm) → size grip (Ø+
  margin) → pivot → 6-DOF IK approach → servo descend (≤10 mm) →
  close → script kinematic hold (grip-wall bridge, same caveat class
  as E8) → lift. Remaining loss: APPROACH-gate vs vision noise
  (retry recovers most).
- Next: CARRY + PLACE tail, then GUI demonstration.

### G28 — GR00T rematch: vision-expert demo corpus — GENERATING
- `vision_scripted_pick.py --record`: success-filtered (PLACED only)
  raw LeRobot episodes, G13 applied-target convention, instruction
  "pick up the cup and set it down to the side".
- Probe: 4 episodes / 24 ep-eq (0.17/ep — 12× the pure-policy rate),
  convert + 8-point validate PASS (2553 frames).
- Main run: 16 envs × 40k steps, cap 200, projecting ~120 episodes.
  Then: convert → validate → fine-tune v3 (frozen DiT locally; if flat
  on clean data, that's the definitive cloud/unfrozen escalation
  signal) → fixed-harness comparison vs RL and GR00T v2.
- Demo quality vs old corpus: smooth deliberate IK actions (no ±12 rad
  teacher noise), consistent trajectory shapes, purposeful camera
  frames — the corpus GR00T's recipe assumes.

### G29 — REMATCH RESULTS (v3 on the 91-demo vision corpus)
- Corpus: 91 episodes / 59,102 frames from the vision-scripted expert,
  8-point validation PASS. Fine-tune v3: frozen DiT, 5k steps.
- **Open-loop: decisive improvement.** MAE 0.36 rad vs zeros-baseline
  0.66 (45% below trivial); MSE 0.31 (v2: 2.82, 9× worse; v2 sat AT its
  baseline). Clean demonstrations fixed the learning signal exactly as
  hypothesized — GR00T now predicts the demonstrated behavior.
- **In-sim fixed harness: still 0 task stages** (27 ep-eq, stopped
  early per flat-eval rule). Structural ceiling identified BEFORE the
  run: the demos' grasps used the script's kinematic hold; the eval env
  provides no such bridge, and static grasps physically fail on this
  contact model (G25 wall — same reason the scripted expert itself
  scores 0 without its bridge).

Three-way comparison (fixed harness, seed 123):
| method            | grasp | full task | open-loop MAE vs baseline |
|-------------------|-------|-----------|---------------------------|
| RL (E8, attach)   | 0.746 | 0.000     | n/a                       |
| GR00T v2 (RL demos) | 0.000 | 0.000   | 1.11 vs 1.0 (at baseline) |
| GR00T v3 (vision demos) | 0.000 | 0.000 | **0.36 vs 0.66**       |

FINAL DIAGNOSIS: imitation learning now WORKS at the action level; the
remaining gap is not data or model — it is the simulator's gripper
contact model, which no policy (RL, scripted, or imitation) can grasp
through statically. Every track converges on the same prerequisite:
**the gripper-fidelity overhaul (vendor datasheet calibration, RL task
#16)**. Once physical grasping is possible, the v3 recipe (vision-
expert demos → fine-tune) is validated and ready to produce a policy
whose actions the environment can actually execute.

## G30 — GRIPPER PHYSICS CALIBRATION CAMPAIGN (final)

Datasheet finding: **no vendor datasheet exists.** Repo URDF, vendor
URDF, and vendor MJCF (Synria-Robot-Descriptions upstream, verified)
all carry the same effort=5/velocity=12 exporter placeholders; even the
vendor's MuJoCo models use ±5 actuator ranges. Calibration is therefore
derived-from-published-spec (1 kg @ 650 mm ⇒ shoulder ≥8.2 Nm) plus
empirical contact repair.

Applied (each from measured evidence, no-hold probes 1-7):
1. Arm efforts 5 → 12 Nm shoulder / 4 Nm wrist (payload-consistent).
2. Piece material static 0.8→1.3, dynamic 0.6→1.1, combine_mode="max"
   (fingers carry no material; pair-averaged μ≈0.65 was the slip).
3. Finger colliders convexHull → convexDecomposition
   (synria_6dof_arm_g30.usd — hull phantom volume filled the concavity).
4. Grasp point: guessed tool-offset → measured finger-body midpoint +
   3.5 cm pad drop (probes showed closes on air → rim → pad line).
5. Compliant close: finger stiffness 2000→400 (stiff close TIPPED the
   cup on asymmetric first contact).
6. Script fixes: no-hold CLOSE→LIFT path, track-vision-through-descent
   (sticky fingers DRAG the cup; frozen target went 4 cm stale).

Result: contact now visibly real (cup gets tipped/dragged — previously
ignored), close converges on the pad line — but **zero normal-force
transfer persists**: piece_z never moves during LIFT in any probe.

REMAINING GAP: pad-cup force closure in PhysX with these mesh
colliders. Recommended next steps, in order of information value:
1. SDF finger colliders (PhysX thin-shell-accurate; needs re-import).
2. Cross-check the identical grasp in MuJoCo with the vendor MJCF
   (~30 min; isolates engine-specific behavior).
3. Swap in a validated gripper asset (Robotiq 2F-85, ledger track 2)
   to separate asset-broken from engine-limited.
4. Real-hardware validation — the physical arm has a real gripper; sim
   contact fidelity stops mattering for deployment.

The kinematic hold remains the sanctioned sim bridge; the G30
calibration still improved realism (no arm sag, no ejection, honest
efforts) and stays in place.

Standing caveats for any future comparison: kinematic attach in demos
(same caveat as RL track), scripted force-detach in expert-recorded
episodes, 32 GB VRAM constraint on-box. No physical-arm deployment —
that requires explicit user approval per mission rules.

### G23 — arm action rep RELATIVE → ABSOLUTE — CONFIRMED
- Diagnosis: recorded actions are offsets from a CONSTANT default
  (absolute up to a shift); RELATIVE rep mixed PD-lagged state into the
  target. Deleted stale meta/stats caches, retrained (v2, 8 min).
- v2 checkpoint-5000 open-loop: **MAE 1.11 (was 1.79), MSE 2.82 (was
  6.55)** on trajs {0,10,25,40,48}. Still ≈ zeros-baseline in aggregate
  — dominated by the teacher's noisy wrist joints (J4/J6 std ~2 rad).
  Open-loop metrics exhausted; the decisive measure is in-sim rollouts
  on the fixed harness (next: policy-server + Isaac client shim).

### G4 — recorder camera variant — PASSED
- NEW `recorder_env_cfg.py`: `add_recorder_cameras(cfg)` bolts wrist +
  overhead 224×224 RGB TiledCameras onto ANY synria_pickplace env cfg
  post-`parse_env_cfg` (so the chess-task RL checkpoint runs unchanged);
  `SynriaLudoPickPlaceRecorderEnvCfg` subclass for the Ludo variant.
- Wrist: tool0 flange, pos (-0.031, 0, 0.06) tool-frame, looking down
  tool -Z, focal 6.0 (first try at focal 12/z 0.02 saw only blank
  tabletop — fingers outside FOV). Overhead: (0.10, 0, table+0.85)
  looking straight down, focal 18.
- Probe `isaac/scripts/diag_recorder_cameras.py` (2 envs, 30 steps,
  headless, alongside the RL session's run): both cameras 224×224 RGB;
  wrist frames both gripper fingers + tabletop; overhead frames board +
  arm + cup. PNGs: scratchpad/cam_probe2/.
- Note: the recorder script must run the CHESS task id
  (`Synria-Chess-PickPlace-v0`) — that's what the E6 checkpoint driving
  the hybrid expert was trained on; the manipulated piece is the cup.

### G5 — demo recorder probe 1 + carry-funnel diagnosis — DIAGNOSED
- Pipeline built: `record_synria_lerobot_demos.py` (success-filtered
  episodes, JPEG-buffered frames, npz+mp4 raw output),
  `convert_synria_lerobot.py` (LeRobot v2.1 per cube_to_bowl_5
  reference; stats auto-generated by GR00T at train time),
  `validate_synria_lerobot.py` (8-point per-episode checks, gate 3).
- Probe (16 envs × 1200 steps, E8_final expert): **0 episodes saved.**
  Matches RL ledger E8 eval: transport 0.077, place 0.000, full 0.000.
- [dbg] telemetry root cause: cup attached ✓, move_dist 0.8-5.4 m ≫
  0.20 ✓, but **hold_steps kept resetting** — carry_done needs 150
  CONSECUTIVE lifted steps and the piece height oscillated around the
  +0.02 m lift threshold. Two mechanisms: (1) pose freeze at
  LIFT_SOLID=0.05 settles ~2-3 cm lower → piece flickers at threshold;
  (2) WANDER slip-exit at <+0.02 (same value as the threshold) churned
  WANDER→POLICY→WANDER, and POLICY carry drags the cup down.

### G6 — carry-funnel fix v2 (takeover bar + LOWER stage) — PARTIAL
- Change: LIFT_SOLID 0.05→0.08, slip-exit 0.02→0.005 (true drop only),
  LOWER stage (60-step interpolation to grasp-moment pose before open).
  Recorder-script change only — env/task untouched.
- Probe v2 (16 envs × 1500): still 0 episodes, but the churn is fixed —
  envs stay in WANDER for 400-500+ steps (was ≤11).
- NEW root cause visible in [dbg]: at a FROZEN arm target the piece
  height oscillates ±5 cm every ~30 steps and actual width bounces
  0↔10 mm. The commanded 31 mm squeeze drives the fingers into the
  kinematically pose-locked 40 mm cup — the contact fight rattles the
  arm and resets the consecutive hold clock. Also: move_dist reads
  1-11 m ≫ 0.20 during the POLICY carry itself — the yaw sweep is
  unnecessary for the path requirement.

### G8 — GR00T zero-shot inference smoke (bundled DROID sample) — PASSED
- `open_loop_eval.py --model-path nvidia/GR00T-N1.7-3B` on
  `demo_data/droid_sample` traj 1: loads N1.7-3B + Cosmos backbone from
  HF cache, runs on the RTX 5090, MSE 0.0027 / MAE 0.033 vs ground
  truth. De-risks the mechanics of gates 4 and 7 (model load +
  inference path) independent of our dataset.
- GOTCHA: the cached HF token file is NOT picked up by in-process
  loads — `HF_TOKEN=$(cat ~/.cache/huggingface/token)` must be
  exported for every gr00t model load (first attempt died 401 on the
  gated backbone despite `hf download` having worked).

### G7 — static hold, no-squeeze fingers — PARTIAL
- Change: YAW_AMP 0 (path req. already met in policy carry), fingers at
  38 mm. Probe v3: arm now QUIET (piece height rock-stable) but stable
  at +0.007 — BELOW the +0.02 lift threshold. The cup dropped ~7 cm at
  takeover: freezing at the MEASURED pose commands the sagging
  equilibrium (arm effort limit is 5 Nm; PD saturates under load) —
  the policy held altitude only by commanding targets above it.

### G9 — freeze at the policy's commanded target — CARRY FIXED
- Change: freeze_pose = default + last policy action (the target that
  demonstrably held the cup up); fingers hold at ACTUAL positions.
- Probe v4: **carry_done FIRED** — hold reached 166/167 ≥ 150, envs
  advanced phase 1→2, LOWER brought the cup back to table height
  (piece_z-rest ≈ −0.01). Modes reached RELEASE and RETREAT.
- Remaining failure: RELEASE cannot open the fingers — commanded 25 mm,
  measured pinned at 0-10 mm. The fingers sit crushed INSIDE the glued
  cup; the kinematic attach cannot yield, so the 42 mm detach width is
  unreachable. Circular deadlock: no detach without opening, no opening
  while attached. 0 episodes.

### G10 — force-detach once in RELEASE — FAILED (re-attach race)
- Change: clear `mgr.attached` at RELEASE t==10.
- Probe v5 (16×2500): 0 episodes, `attached=False` never observed. The
  env's attach event RE-GLUES the next step: width 5 mm is inside the
  [4, 45] mm attach band, phase 2 is carryish, cup is at the fingers —
  all re-attach conditions hold. Envs timed out to RETREAT still
  attached; no env reached phase 3.
- Caveat for any comparison: demos rely on attach + scripted detach —
  gripper-fidelity caveat applies (as it does to the RL track).

### G11 — continuous detach suppression + close-below-band — FIRST DEMO
- Probe v6 (16×2500): **1 episode saved** (len 518 — full pick → carry
  5 s → set-down → return cycle). First successful demo end-to-end.
  Rate still awful (~0.02/ep): pre-step clears can't stop IN-step
  re-attach (the event runs inside env.step), so opening-in-place only
  succeeds when the ratchet gets lucky.
- Visual check of the episode: wrist cam frames the red cup in the
  gripper; overhead shows arm on the GROUND-PLANE scratch scene (the
  recorder inherits the train-time tabletop→ground-plane override).
  TODO: confirm the eval harness uses the same scene, else record demos
  on the eval scene for visual consistency.

### G12 — withdraw-home-closed, open-far — IN PROGRESS
- Hypothesis: stop fighting the attach band; break the `near < 9 cm`
  condition instead. RELEASE: close below the band (t<10), detach, then
  withdraw the arm HOME with fingers still closed (no re-attach
  possible at any width once >9 cm away). RETREAT: fingers open far
  from the cup → measured width clears the 40 mm released bar
  unobstructed → set-down fires (home detection is arm-joints-only).

### G13 — record applied targets, not raw policy output — FIXED
- End-to-end convert+validate on the G11 episode caught it: raw policy
  actions reach ±12 rad (J4) / ±14 m (fingers) — the env clips before
  applying. Storing raw output would poison GR00T normalization stats.
  Recorder now stores clip(default+action)−default (applied-target
  offsets); env still receives the raw action unchanged. Converter +
  8-point validator otherwise PASSED on the real episode (sync,
  timestamps, indices, instructions all clean).
- Probe v7 (G12+G13): 16 envs × 2500, max 12 episodes.
  Log: scratchpad/lerobot_rec_v7.log

### G12/G13 probe v7 — 1 episode; withdraw defeated by cup-rides-hand
- Result: 1 episode (0.02/ep — unchanged). 26 phase-2 sightings, ZERO
  phase-3 (set-down never fires).
- Root cause of the withdraw failure: while attached the cup FOLLOWS the
  hand at the attach offset, so `near < 9 cm` can never be broken by
  moving the arm — the cup comes along. And wedge cases (e13: width
  pinned at 10 mm with fingers commanded closed) cannot get below the
  4 mm attach band either. In-step re-attach cannot be beaten by any
  pre-step bookkeeping.

### G14 — gate the attach EVENT itself (recorder cfg only) — IN PROGRESS
- Hypothesis: wrap `mdp.attach_carried_pieces` on the recorder's parsed
  env_cfg (`env_cfg.events.attach_carried.func = _gated_attach`; repo
  env untouched). For script-flagged envs (RELEASE t≥10, carryish
  RETREAT): clear glue, block new attaches via the event's own
  `pin_steps==0` guard, skip the kinematic write → cup fully physical
  during release/retreat. Plus stuck-RETREAT recovery (>200 steps →
  back to POLICY).
- Probe v8: 16 envs × 2500. Log: scratchpad/lerobot_rec_v8.log
- Result: 1 episode. Detach now WORKS (`attached=False` observed) but a
  new physical constraint surfaced: fingers opening IN the cup stall at
  ~36 mm = the cup's inner diameter — below the 40 mm released bar. The
  fingers must exit the cup vertically before they can open.

### G15 — vertical exit before open — PARTIAL (1/run, env variety)
- Change: RELEASE = settle (no squeeze) → lift back to the aloft freeze
  pose (reverse of LOWER, fingers held) → RETREAT opens far from cup.
- Probe v9b (16×2500): 1 episode — from env 13, which was permanently
  wedged in every earlier probe. Mechanics now reach all envs, but the
  RATE is stuck at ~1/run (~0.02/ep).
- Suspicion: episode-horizon truncation — the scripted tail needs ~300
  steps after solid lift; attempts starting after ~step 600 of the
  900-step episode can never finish (saved lens 378-518 all imply
  early starts).

### G16 — funnel counters + scale — MEASURED
- Probe v10 (32×5000, ≈175 ep-eq): takeover 98 → carry_done 15 (15%) →
  released 10 → retreat 10 → setdown 5 → cycle 3 saved; 33 truncated
  in-script; 3 stuck-recoveries. Bottleneck = takeover→carry_done.

### G17 — surface-pick staging (ZONE 0.8) — NO EFFECT ON RATE
- Hypothesis was late-start truncation; staging pins the cup on the
  table under the settling hand (real from-surface grasp preserved).
- Probe v11 (32×5000): takeover 90 → carry_done 16 (18%) → cycle 2-3,
  trunc 39. Same profile as v10 — truncation was a SYMPTOM. The binding
  constraint is the static hold sagging below the +0.02 lift threshold
  before 150 consecutive steps accumulate. (Staging kept: it makes
  demos denser per episode and costs nothing.)
- Banked demos so far: v10 3 + v11 3 raw episodes (plus v6/v9b singles
  recorded pre-G13 — action fix means those are NOT usable).

### G18 — integral droop compensation in WANDER — MARGINAL
- Probe v12 (32×5000): takeover 97 → carry_done 19 (up from 15-16) →
  released 12 → retreat 4 (down — the chain now completes later in the
  episode and truncates mid-RELEASE) → cycle 1. Net demos/run flat.

## PROTOCOL STOP — root-cause diagnosis (after probes v6-v12)

Outcome metric (saved demos/run) has been flat at 1-3 across seven
probes despite each probe measurably fixing its target stage (funnel:
carry_done 0 → 19/run; every stage of the chain has now completed at
least once).

Root cause: the demo expert must thread SIX sequential fragile stages —
policy grasp → solid lift → 150 CONSECUTIVE-step aloft hold (vs 5 Nm
effort-saturation droop) → lower → release (vs the kinematic attach's
re-glue race and the cup's 36 mm inner-mouth finger trap) → retreat/
home — inside a 900-step horizon, starting from HIGH-VARIANCE policy
grasp configurations. Per-stage success ~0.5-0.8 multiplies to ~1-2%
end-to-end. Each scripted fix moves the surviving population to the
next fragile boundary; no single remaining defect dominates. This is a
variance problem, not a bug.

Decision (recommended next step, adopted):
1. BANK WITH WALL-CLOCK: long unattended recording runs at the measured
   ~0.017/ep — 32 envs × 40k steps ≈ 1400 ep-eq ≈ 20-25 demos/run;
   repeat to reach ~100. No new mechanism risk.
2. START STAGED VALIDATIONS NOW: gates 2-6 (dataset load, forward pass,
   smoke train, checkpoint save/reload, offline inference) need A
   dataset, not 100 demos — run them on the 7 banked post-G13 episodes
   (v10: 3, v11: 3, v12: 1) to de-risk downstream while data accrues.
3. LATER: if the RL session's E9+ staged-curriculum checkpoints install
   the place skill, switch the demo source to pure-policy rollouts and
   re-record at quality.

## Blockers

- B1: ~~GR00T not importable~~ — RESOLVED (G2).
- B2: ~~no camera sensors~~ — RESOLVED (G4).
- B3: no demonstrations — record ~100 Ludo demos via hybrid expert once
  B2 resolves. Caveat: expert quality plateaued at ~0.02-0.03
  completions/episode in RL-session probes; may need a better expert or
  acceptance-filtering of successful episodes only.
- B4: ~~scaffold drift~~ — RESOLVED for train path (G3); inference.py
  rework deferred to gate 7.

## Next actions

1. Build camera recorder env variant (B2) — wrist + overhead RGB.
2. Record + validate demos (B3): success-filtered episodes, LeRobot v2
   + meta/modality.json, then gates 2-3 (dataset load, per-episode
   validation).
3. Gates 4-8: forward pass → smoke train → checkpoint reload → offline
   inference (rework inference.py) → predicted-vs-recorded comparison.

## G31 — MuJoCo cross-check: the grip wall is NOT PhysX-specific

Minimal rigs in MuJoCo 3.11 (script: isaac/scripts/mujoco_grip_crosscheck.py):
box pads and the vendor finger STLs on prismatic slides, 40 mm / 50 g
cylinder, close-then-lift, μ=1.2, multiple close depths, rigid AND soft
(solref/solimp) contact:
- moderate close: squeeze-EJECT (cup shot ~1 m) — same as PhysX probes;
- deep close, rigid: cup slips out during lift despite ~14 N capacity;
- deep close, soft: cup stays centred but pads slide off vertically;
- vendor meshes: self-collide before caging (and vendor STLs are in
  metres — repo importers beware).

CONCLUSION: naive rigid-pad grasping of rigid objects fails in BOTH
engines; grasp-capable simulation requires a dedicated, tuned gripper
model (cf. Robotiq 2F-85 in MuJoCo Menagerie). Synria never produced
one (their own MJCF is placeholder physics, G30). Ranked path forward:
1. Adopt a validated gripper asset (Robotiq 2F-85) for sim-honest
   grasping experiments (ledger track 2).
2. Keep the kinematic-hold bridge as the Synria-gripper stand-in — it
   is a reasonable model of what compliant pads do.
3. Real hardware remains the definitive validation for the Synria
   gripper itself.

## R2 — VALIDATED GRIPPER GRASPS THE CUP (positive control)

Isaac Lab's Franka + Robotiq 2F-85 variant (NVIDIA-tuned closed-loop
drives), our exact cup (40 mm / 50 g / G30 material): feedback descent
to pad height (xy-offset 9 mm), close — drive commanded 0.79, STALLED
at 0.488 ON the cup (true force closure) — lift: cup 0.020 → 0.186 m
tracking the hand. Script: isaac/scripts/diag_robotiq_grasp.py.

CONCLUSION: PhysX force closure works with a properly modelled gripper.
The Synria asset (vendor-uncalibrated, G30) is the wall — confirmed by
positive control. Proceeding R3: mount the 2F-85 on the Synria arm.

## R-track status (Robotiq swap, probes R4 v1-v9)

PROVEN: R2 — the validated Franka+2F-85 lifts our cup (drive stalls ON
the cup at 0.488, cup rises 16.6 cm). PhysX force closure works with a
properly modelled gripper.

Hybrid attempt (Synria arm + flattened 2F-85): weld to link6 holds,
articulation boots with the correct 12-joint set, empirical jacobian
row discovered (PhysX merges fixed-jointed bodies: gripper base rides
link6's row 5, not the -1 convention's 7). BUT the flatten/CopySpec
surgery produced a DEGENERATE gripper: all internal finger bodies
coincide with the gripper base (v9: pad-mid == base exactly) — the
instanced link transforms did not survive the copy. Every "free close"
in v5-v9 traces to this.

CONCLUSION: hand-rolled USD articulation surgery on an instanced,
variant-gated asset is the wrong tool. Two correct paths:
1. Isaac Sim's Robot Assembler extension (purpose-built for attaching
   grippers to arms — the AssemblerFixedJoint in the Franka asset is
   its artifact). One focused attempt.
2. Surrogate arm: use the intact Franka+2F-85 for the honest-grasp task
   line (works TODAY per R2); Synria remains the identity for demos/
   real-hardware, Franka+2F-85 becomes the sim-grasp workhorse.

### INCIDENT (resolved): isaacsim-robot-setup contaminated the Isaac venv
Installing `isaacsim-robot-setup==5.1.0.0` silently replaced the shared
Isaac venv's torch with 2.7.0+cu126 (no Blackwell sm_120 kernels) —
"no kernel image" CUDA failures on the 5090. Remediated immediately:
force-reinstalled torch/vision/audio 2.7.0/0.22.0/2.7.0 from the cu128
index (--no-deps); CUDA ops verified. Lesson: NEVER pip-install into
the shared Isaac venv without --no-deps + pin verification.

## R-track FINAL (Assembler path, ~15 probes)

Works: Robot Assembler composes the hybrid mechanically (fixed joint,
single articulation, 12 joints, transforms track, boot-pose search +
grasp-height stance protocol solid). Venv contamination incident from
isaacsim-robot-setup deps resolved (cu128 restored, JIT verified).

Blocker, precisely characterized: a SIMULATION-READY standalone 2F-85
does not exist in the asset catalog — `Robotiq_2F_85_edit.usd` is an
authoring layer (fingers close through the cup: no effective collision/
physics), `configuration/Robotiq_2F_85_config.usd` has broken upstream
sublayer URLs, and the only complete physics composition lives inside
franka.usd's variant machinery (proven working in R2). Both extraction
attempts (flatten+CopySpec; direct reference) lose the physics layers.
Note: pad-frames-at-base-origin is NORMAL for this asset (earlier
"degenerate transforms" conclusion was partly wrong; the persistent
symptom — free closes — is the missing collision composition).

RECOMMENDATION (standing): use the surrogate Franka+2F-85 (R2-proven)
as the sim-grasp workhorse via a task-env robot variant — unblocks the
honest-grasp pipeline (no-hold picks, caveat-free demos, GR00T v4)
immediately. Revisit the Synria hybrid if/when a fixed standalone
2F-85 asset ships.

### CORRECTION NOTE (S1 diagnosis): orphan-GPU confound in late R-track
The "no active physics scene / physics_sim_view None" boot failures in
S1 v1-v4 were PhysX GPU OOM: 14 orphaned probe processes (kills hit the
bash wrapper, python children survived) held 28.5 GB. Cleaned. This
confound may also have degraded late R-track probes (v5+ ran alongside
accumulating orphans) — the edit-asset-lacks-physics conclusion still
stands on structural evidence, but hybrid re-testing on a clean GPU is
cheap if ever revisited. Ops rule added: kill the PYTHON child (pgrep
-f script name), and verify nvidia-smi after every kill.

### S1 v10 (2026-07-30): lift telemetry — the "grasps" are phantoms
Lift-stage telemetry (cup z, hand z, finger stall every 20 steps) settled it:
- The cup NEVER moves (z = 0.020 constant, not even a millimeter of
  disturbance) while the hand rises smoothly and the finger reading holds
  0.44. This is not a slip — the pads never had the cup.
- Smoking gun: env 1 entered LIFT with hand z = 0.313, i.e. pads ~15 cm
  ABOVE the cup, yet still registered "stall 0.44". The 0.42-0.44 readings
  are therefore NOT contact stalls.
- Hypothesis (v11 tests it): the 2F-85 drive is heavily damped, so at the
  CLOSE gate (t_in > 90 env steps = 180 physics steps) the finger is still
  MID-TRAVEL toward 0.79 — 0.44 is a position en route, not an
  equilibrium. R2 waited 300 physics steps before reading 0.488.
- v11 adds a boot-time free-air close calibration (settle curve closing on
  nothing) + grasp-geometry print (hand-to-cup ground truth at detection)
  + CLOSE gate lengthened to 240. Grasp band will be recalibrated against
  the free-air settle value.
- Ops note: v9/v10 processes hang after printing RESULT (env teardown);
  kill the python child by PID after RESULT and verify nvidia-smi.

### S1 v11-v13 (2026-07-30): the failure onion, peeled layer by layer
Three instrumented probes isolated FOUR stacked defects:
1. v11 free-air calibration: the drive closes fully (0.798) in <30 steps on
   nothing — so 0.42-0.44 IS an obstruction stall. Combined with
   grasp-geometry ground truth (hand 3.7-16.6 cm from the cup at
   "grasp"), the obstruction is the GROUND, not the cup.
2. v11 vision audit: 3.3 cm vision error during approach — the gripper
   occludes the overhead camera and shifts the HSV centroid. Fixed in v12:
   perceive only while parked at home + mask-area occlusion gate
   (m00 >= 8000). Result: 2-4 mm vision error, confirmed by acquire-vs-gt.
3. v12 grasp-geometry: with vision fixed, the 3D-norm descend gate let the
   hand enter CLOSE up to 12 mm low; pad tips at ~9 mm jam into the ground
   at stall 0.44 BEFORE cup contact (~0.49), and the jam reaction shoves
   the arm centimeters (v10's "grasp at hand z 0.313" pop explained).
   Fixed in v13: separate xy (<8 mm) and z (<5 mm) gates + grasp z
   0.181->0.187 + ground band 0.42-0.445 excluded from grasp detection.
4. v13 close telemetry, the deepest layer: with near-perfect entry
   (dxy 3-5 mm, z 0.18-0.19 — BETTER centering than R2's 9 mm), the
   fingers still snap to 0.798 = fully closed on NOTHING. Stall read
   0.558 two control steps after CLOSE began -> finger speed ~30 rad/s:
   the step target 0->0.79 golf-clubs the cup out of the jaws. R2's
   single-shot success with the same snap was wedge luck, not robustness.
Fix in v14: ramp the close target at 0.013 rad/control step (~1 s stroke,
matching the real 2F-85's 0.5-1 s), open at 2x that rate. This is also
more honest sim-to-real: no real gripper steps its full stroke in 16 ms.

### S1 v15-v16 (2026-07-30): SUCCESS — first honest physical pick-move-place
Final two defects and their fixes:
5. v15 boot print: the servo's attitude lock (boot hand orientation) held
   the gripper at 45.4 DEGREES off vertical the entire time — every
   earlier close approached with a tilted pad plane (wedge-ejects the cup,
   spears one fingertip into the ground; also why hand-origin dxy misled:
   the TCP projects centimeters from the hand origin under tilt). Fixed:
   rotate lock_quat by the shortest arc taking the hand z-axis to the
   nearest vertical (45.4 deg correction applied at boot).
6. v15 vertical-gripper telemetry: at grasp height the fingertip arc dips
   into the ground EARLY in the close sweep (stalls 0.33-0.41, reaction
   levers the hand up 3 cm, pads then close 0.798 above the cup top).
   Fixed in v16: PRE-SHAPE the fingers to 0.35 rad (~55 mm opening)
   during ALIGN/DESCEND — past the arc's low region, tips hoisted, still
   clears the 40 mm cup with the +/-8 mm alignment budget.
RESULT (v16, 4 envs, 4000 steps, --headless): 29 close attempts,
3 genuine on-cup drive stalls (0.498 / 0.497 / 0.532 — matching R2's
0.488 signature), 2 physical friction-held lifts, 2 COMPLETE physical
pick-move-places (third grasp truncated by run end). Zero kinematic
holds, zero teleports: vision (HSV overhead, 2-4 mm), DLS servo,
ramped pre-shaped close, PhysX friction the whole way.
Honest caveats: per-attempt grasp rate is still low (~10% of close
entries; misses are residual ground jams and alignment outliers) —
throughput tuning is S2 work. Success criteria per mission rules: this
is measured task performance in sim, not loss curves.
S1 STATUS: COMPLETE. Next: S2 — record LeRobot demos from this expert
(only successful episodes kept), then S3 GR00T v4 fine-tune + honest
fixed-harness comparison.

### S2 (2026-07-30/31): COMPLETE — 60 honest surrogate demos recorded + converted
Recorder embedded in franka_vision_pick.py (--record): success-only
episodes, states/actions (T,8 = 7 arm + finger_joint, applied clipped
targets per the G13 convention), wrist+overhead 224px mp4 at 30 Hz.
Operational finding: marathon runs physically degrade — after ~100k
steps of accumulated failed-close ground contact the arms wedge and stop
converting (batch 1 stalled at 27 eps by step 95k, hands reading below
the floor). Fix: fresh-boot batches (~30 eps / <=90k steps) with
episode-index resume from disk. Batches: 27 + 23 + 10 = 60 episodes;
batch 3 converted 10-for-10 picks to placements.
Converted with convert_synria_lerobot.py --embodiment franka ->
reports/training/franka_cup_lerobot_demos: 60 episodes, 18,902 frames,
modality arm 0-7 / gripper 7-8, robot_type franka_panda_robotiq_2f85.
Validated: parquet/video counts, dims, action ranges (gripper 0-0.79).
Next: S3 — staged GR00T v4 fine-tune on this dataset.

### S3 (2026-07-31): GR00T v4 fine-tune LAUNCHED (staged)
Smoke (20 steps): PASS — dataset load, forward/backward, checkpoint-20
saved; loss 1.33 at start, ~1 step/s. Full run: N1.7-3B, new_embodiment,
frozen DiT/LLM/visual + tuned projector/VLLN (v3 recipe, config.yaml
verified from the v3 checkpoint), batch 1, lr 1e-4, 5000 steps,
save at 2500/5000. Dataset: franka_cup_ds (60 eps / 18,902 frames).
Output: scratchpad/gr00t_franka_v4; log gr00t_finetune_v4.log.
Next: open-loop eval (MAE vs zeros baseline) then in-sim fixed harness.

### S3 (2026-07-31): v4 fine-tune COMPLETE + open-loop eval — BEST RESULT YET
Training: 5000 steps in 467 s (10.7 steps/s), final train loss 0.552,
checkpoints 2500/5000 saved with processor + experiment_cfg.
Open-loop eval (checkpoint-5000, trajs 0/10/25/40/55, 200 steps each):
  MAE per traj: 0.202 / 0.231 / 0.237 / 0.226 / 0.246
  Average MAE 0.228, MSE 0.114
  Zeros-baseline MAE on the same windows: 0.5995
  -> 62% below the trivial baseline (v3: 0.36 vs 0.66 = 45% below;
     v2 sat AT its baseline). GR00T v4 predicts the surrogate expert's
     demonstrated behavior more faithfully than any prior fine-tune.
Honest scope note: open-loop MAE measures imitation fidelity, not task
success. The decisive test is the in-sim closed-loop harness (serve
checkpoint into franka_cup_env, count physical picks/places) — next.

### S3 FINAL (2026-07-31): closed-loop eval — 0/24, root cause diagnosed
Closed-loop harness (policy-served, fixed 900-step windows, honest
pick/place counts, cup spawn +/-8 cm like the demos): 0 picks, 0 places
over 24 episodes (cut early — binomial p < 1e-4 vs even a third of the
expert's window rate; continuing added no information).
Diagnosis (telemetry + frame dumps of a served window):
- The policy REPRODUCES the expert's action schedule: pre-shapes the
  fingers to 0.349 (expert: 0.35), sweeps a demo-like arc toward the
  workspace, then closes fully — but hovering at z ~0.41 beside the cup.
  It never visually servos onto the actual cup position.
- Interpretation: open-loop MAE 0.228 (62% below zeros) shows action
  imitation works; the closed-loop failure is missing VISUAL GROUNDING —
  the model averages over demo trajectories instead of conditioning the
  descent on the observed cup location. With the 32 GB frozen-DiT
  constraint (NVIDIA's full fine-tune floor is 40 GB), only the
  projector/VLLN adapt, which appears insufficient to couple vision to
  the action head for precise spatial targeting; 60 demos over a
  +/-8 cm spawn range is also thin for spatial generalization.
Honest mission answer at this stage: under local hardware constraints,
GR00T v4 does NOT complete the task closed-loop (0/24) despite the best
open-loop imitation of the whole effort. The scripted vision expert
remains the only policy that physically completes pick-move-place.
Recommended escalations (in order of cost):
1. Control experiment: fixed cup spawn fine-tune + eval — isolates the
   spatial-grounding hypothesis cheaply.
2. Scale demos to 200+ with spawn curriculum (recorder is ready; ~3 h).
3. Unfrozen-DiT fine-tune on a 40 GB+ cloud GPU (the decisive test of
   the frozen-DiT hypothesis).

### CONTROL EXPERIMENT (2026-07-31): v5 fixed-spawn — DIAGNOSIS CONFIRMED,
### FIRST GR00T CLOSED-LOOP TASK COMPLETIONS
Setup: 22 fixed-spawn demos (cup always (0.5,0); recorder --fixed_spawn;
batch ended 22/40 at its step cap — sufficient for a control), converted
(7,010 frames), v5 fine-tune with the EXACT v4 recipe (frozen DiT,
batch 1, lr 1e-4, 5000 steps; 464 s, train loss 0.565), closed-loop
eval with --fixed_spawn (8 windows x 4 envs = 32 episodes).
RESULT: 4 physical picks, 3 COMPLETE physical pick-move-places over 32
episodes (12.5% / 9.4%) — the FIRST GR00T-driven closed-loop task
completions of the entire project. v4 (randomized spawn): 0/24.
Open-loop v5: MAE 0.212 (vs v4's 0.228) — imitation fidelity comparable;
the closed-loop difference is the task geometry.
VERDICT: the spatial-grounding diagnosis is CONFIRMED. With frozen
DiT + 60 demos, GR00T learns the SKILL (approach, pre-shape, descend,
grasp, lift, carry, place — executable end-to-end) but not the VISUAL
CONDITIONING needed to retarget it to a randomized cup position.
Escalations now have confident expected value:
1. Demo scaling (200+ randomized, spawn curriculum) — attacks coverage.
2. Unfrozen-DiT cloud fine-tune (40 GB+) — attacks the visuomotor
   coupling limit. Highest expected value: both together.
Honest note: 9.4% fixed-spawn success is far below the scripted
expert (~69% of windows converted in the fixed-spawn recording run);
GR00T's execution is also less reliable than the expert even without
spatial variation — expected with 22 demos and frozen weights.

### ESCALATION (2026-07-31): 200-demo randomized corpus COMPLETE
Overnight fresh-boot batch driver (timeout-guarded; recorder hard-exits
after RESULT since the teardown-hang fix): 60 -> 200 episodes across 6
batches, zero manual intervention after the hard-exit fix landed.
Converted: 200 episodes / 63,340 frames ->
reports/training/franka_cup_lerobot_demos_v2 (franka embodiment split).
Next: v6 fine-tune (frozen DiT, demo-scaling arm) + evals. The cloud
unfrozen-DiT package (docs/CLOUD_FINETUNE_FRANKA.md) can now ship with
this corpus.

### v6 (2026-07-31): 200-demo frozen-DiT at 5000 steps — TRAINS TO BASELINE
Open-loop MAE 0.5701 vs zeros-baseline 0.5702 on the same trajectories —
the model learned nothing beyond the mean action. Cause identified before
concluding anything about demo scaling: exposure confound. At batch 1,
5000 steps = 0.08 epochs of the 63,340-frame corpus (v4 got 0.26 epochs
of its 18,902 frames). Closed-loop skipped for this checkpoint (v2
precedent: open-loop AT baseline = nothing to evaluate).
v6b launched: identical except 15,000 steps (~0.24 epochs — exposure
matched to v4). ~23 min. If v6b recovers v4-level open-loop AND gains
closed-loop picks, demo scaling helps; if open-loop recovers but
closed-loop stays 0, frozen-DiT visual grounding remains the binding
constraint and the cloud arm is decisive.

### v6b (2026-07-31): 15k steps made it WORSE — batch-noise hypothesis
v6b (exposure-matched 15,000 steps) open-loop MAE 0.808 — ABOVE the
zeros baseline (0.570). Longer frozen-DiT training on the diverse corpus
actively degrades. New hypothesis: batch-1 gradient noise. v4's 60-demo
corpus is comparatively homogeneous (noise tolerable, MAE 0.228); the
200-demo randomized corpus produces conflicting per-sample gradients
that a batch of 1 cannot average. v6c launched: effective batch 8 via
gradient accumulation (identical VRAM), 5000 optimizer steps = 40k
samples (~0.6 epoch), ~1 h. If v6c stays at/above baseline, the
demo-scaling arm CLOSES under frozen DiT and the cloud unfrozen-DiT run
(which also unlocks true batch 8-16) is the decisive remaining test.

### v6c FINAL (2026-07-31): batch-8 accumulation — first RANDOMIZED
### closed-loop completion; demo-scaling arm concluded
v6c (200 demos, effective batch 8 via gradient accumulation, 5000 opt
steps = 40k samples, 76 min): train loss 0.098, open-loop MAE 0.0599 —
10x below the zeros baseline (0.5702) and 4x better than v4 (0.228).
Closed-loop (randomized spawn, 32 episodes): 1 pick, 1 COMPLETE
pick-move-place (3.1%) — the first randomized-task completion by any
GR00T checkpoint in this project (v4: 0/24).

COMPLETE LOCAL RESULTS TABLE (frozen DiT, RTX 5090 32 GB):
  ckpt | demos     | batch | open-loop MAE (vs 0.57-0.60 zeros) | closed-loop
  v4   | 60 rand   | 1     | 0.228                              | 0/24 rand
  v5   | 22 fixed  | 1     | 0.212                              | 4 picks 3 places /32 fixed
  v6   | 200 rand  | 1     | 0.570 (AT baseline)                | skipped
  v6b  | 200 rand  | 1     | 0.808 (WORSE than baseline)        | skipped
  v6c  | 200 rand  | 8 acc | 0.0599                             | 1 pick 1 place /32 rand

Conclusions:
1. Effective batch size is CRITICAL for diverse corpora — batch 1 trains
   to (or past) the trivial baseline; batch 8 accumulation yields the
   best imitation of the whole project at identical VRAM.
2. Demo scaling + batch fix moved randomized closed-loop from 0 to
   nonzero: the local recipe CAN produce spatially-grounded behavior.
3. But 3.1% vs the scripted expert's ~65% window rate: the frozen-DiT
   visuomotor ceiling remains dominant. The open/closed gap (MAE 0.06
   yet 1/32) is compounding drift the frozen backbone cannot correct.
Recommended: cloud unfrozen-DiT run (docs/CLOUD_FINETUNE_FRANKA.md,
updated: use batch >= 8 — now empirically critical) on the 200-demo
corpus. Local option: longer batch-8 training (2-3 epochs) may still
squeeze more; diminishing odds it closes the gap alone.

### GRASP AUDIT (2026-07-31): THE GRIP WALL ROOT CAUSE — FOUND
Mission-directed geometric audit (reports/synria_grasp_audit.md). The
months-long Synria "grip wall" is fully explained by asset defects:
(1) arm links have NO colliders; (2) finger mesh colliders are DEAD
(corrupt extents — fingers close straight through the cup);
(3) the only live colliders are two STATIC pad boxes that never move
with the joints — fingers jam after 0.4/0.3 mm travel against their
faces (matched to 0.1 mm, reproduced in free air with no cup).
No training run ever closed the gripper onto anything. All prior
friction/effort/decomposition work (G25-G31) was tuning around a
gripper that mechanically could not close.
Geometry itself: 50.0 mm measured opening vs 40 mm cup = 5 mm/side
(marginal-acceptable); recommended cup 25-30 mm regardless.
Deterministic gate script committed (synria_grasp_feasibility.py,
0/10 on production). Synria policy training BLOCKED until a fixed
asset passes the gate; padfix re-authoring in progress
(make_padfix_usd.py; box orientation needs USD-composed transforms).

### GRIPPER REPAIR MISSION (2026-07-31 evening) — phases 1-5, INCOMPLETE
Every attempt recorded (no hidden failures), newest evidence wins:
- P1 composition map: pad_collider cubes ARE parented to the moving
  finger links (local (0,∓30,∓13.75) mm); joints axis Z, limits
  [0,25]/[-25,0] mm, no mimic/scale; arm links have NO colliders; finger
  MESH collision prims carry degenerate extents (±3.4e38).
- P4 travel (10 cycles, 0/25/50/75/100%/0):
  production @floor FAIL; production @base+0.5 m **PASS (25.00 mm)**;
  nopad @floor PASS; padfix/padfix_l/padfix2 @floor FAIL, @base+0.5 m
  PASS; padfix_l with no ground and no cup @floor PASS.
  => ROOT CAUSE OF ALL "JAMS": the harness drove the arm at the URDF's
  real gains (5 Nm / k=80), which cannot hold the pose against gravity;
  the arm sagged and the fingers closed on the GROUND. The earlier
  "static phantom pad" reading (audit, morning) is REFUTED — corrected
  in reports/synria_grasp_audit.md AMENDMENT section.
- Failed repair attempts, kept for the record: synria_6dof_arm_nopad.usd
  (all finger colliders off), _padfix.usd / _padfix_l.usd / _padfix2.usd
  (re-authored 4 mm pads, single-side, explicit contact offsets),
  _repaired.usd (mesh colliders off) — none changed the outcome once the
  floor-contact confound was removed.
- Harness bugs found and fixed along the way: descend overshoot buried
  the 60 mm fingers 35 mm underground; per-step pose-pinning of test
  objects defeated contact impulses; hover "held" criterion counted a
  fall to the ground as success; arm drifted 92 mm in x during a close.
- P5 blocks (elevated base, free-standing 20/28/30/40 mm blocks verified
  between the pads within 1 mm): production 0/12, g30 0/12, padfix2
  0/12 — fingers travel the FULL 25 mm/side and close to 0.01 mm
  through every block. **NEW OPEN DEFECT: no finger collider generates
  contact with dynamic rigid bodies.** Hypothesis (untested): collision
  prims are not effective in the spawned articulation (cooking of scaled
  Cube colliders under links, missing approximation attrs, or Isaac
  Lab's spawn path not applying the collision API through the reference).
- Phases 6-9 (arm colliders, env wiring, 28 mm object, grasp gate,
  automated gate + launcher guard) NOT started.
Synria policy training remains BLOCKED. Gate script:
isaac/scripts/synria_grasp_feasibility.py (exit 0 pass / 3 travel fail /
4 block fail).

### GRIPPER REPAIR — CORRECTED FINDINGS (2026-07-31, late)
Runtime probe of the SPAWNED articulation (scratchpad probe, results in
reports/synria_grasp_audit.md "SECOND AMENDMENT"):
- pad_collider prims: enabled=True in the live scene (they work).
- finger mesh colliders: enabled=False (dead, but redundant).
- The pad collider spans ~-15..+58 mm in z RELATIVE TO THE FINGER FRAME
  (the finger extends diagonally). Every earlier block/hover/grasp test
  placed its object at "frame - 30 mm" = BELOW the collider, which is
  why they all read "closes straight through".
- With placement corrected to the true pad centre (pad_center_world()),
  fingers STALL ON CONTACT: travel 0-6 mm instead of the full 25 mm.
- Phase 5 still not passed: measured widths remain untrustworthy because
  the arm drifts under contact load (block offset from the pads walked
  from -0.5 mm to -21 mm in x across 12 trials). Needs a rigid fixture
  (object pose fixed relative to the PADS, re-measured immediately
  before the close), not more asset edits.
Net: two of this session's own harness bugs (arm sag onto the floor;
objects placed outside the collider) produced two confident but WRONG
"the asset is broken" conclusions. Confirmed real defects remaining:
arm links have no collision geometry; dead mesh colliders; and the
likely control-side bug — grasp poses targeting the finger FRAME rather
than the PAD CENTRE (audit both RL and GR00T envs).

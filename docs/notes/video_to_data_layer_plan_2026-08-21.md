# Video→Data as an Evaluated Layer — Integration & Analysis Plan (2026-08-21)

Add NVIDIA Isaac **Video→Data** (Real→Sim→Real) as a new data/learning layer in
this project, and — the point of the exercise — **rigorously evaluate its
effectiveness** against our existing scripted-corpus → GR00T path, with the same
audit discipline we used for the GR00T target-observability experiment.

Sources: repo <https://github.com/nvidia-isaac/video_to_data>, project
<https://nvidia-isaac.github.io/video_to_data/>, CHORD
<https://arxiv.org/abs/2607.00033> (82.12% over 1,831 tasks; 90.77% whole-body;
open-loop + closed-loop real transfer).

---

## 1. What it is (grounded in the release)

Three stages, **host-orchestrated with containerized inference** — thin Python
wrappers `docker run` each module; **no CUDA/PyTorch on the host** (all ML deps
live in the images). Modules exchange strongly-typed `v2d_common` dataclasses
(`DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Mask`) and
write intermediate artifacts to disk (depth PNGs, pose JSONs, mask PNGs) →
independently runnable, cacheable, composed via `v2d_pipelines`.

| Stage | In → Out | Status |
|---|---|---|
| **1. Video Ingestion Agent** | raw human video → action segments + entity scene graph + SigLIP-2 frame embeddings (LangGraph, NL retrieval, optional Gradio UI) | **released** |
| **2. Reconstruction** | clip → depth, masks, textured object meshes, 6-DoF object pose trajectories, human hand/body mesh + motion | **released** |
| **3. Robotic Grounding** | reconstructed motion → retarget to robot embodiment → **RL in Isaac Lab (PPO) with CHORD contact-wrench guidance** → policy | **RL/retarget code "later release"; docs "coming soon"** |

Implication: we can run **Stages 1–2 now** (video → sim-ready assets +
trajectories); Stage 3 needs either the pending release or our own interim
retargeting (we already have Isaac Lab + the Synria arm).

---

## 2. Why it fits THIS project (the thesis under test)

Our current data path is **scripted expert corpus → GR00T-1.7 LoRA** (Ludo).
Documented result: **5% closed-loop** (see `groot17_eval01_findings`), and the
scripted expert itself is a hand-tuned ~600-line controller (and I'm currently
hand-scripting the dice-cup manipulation — see `ludo_game_milestones`).

Video→Data offers a **different data path**: *human demonstration video →
reconstructed 6-DoF trajectories → retarget to the arm → RL (CHORD)*. The
evaluable claim: **does video-derived data + contact-wrench RL produce working
manipulation policies where our scripted-corpus + imitation path did not — and
at what cost?**

The **dice-in-cup / roll / dump** task (B2b) is contact-rich dexterous
manipulation — precisely CHORD's target. It is the natural first case: instead
of hand-tuning the grasp/shake/invert controller, learn it from a human demo of
rolling dice in a cup.

Real→Sim→Real closes on hardware we already run: policy → **M3 Pro (Jetson
Orin)**.

---

## 3. Evaluation design (the "proper analysis")

### Research questions
- **RQ1 — Efficacy:** does a video-derived policy beat our current learned
  baseline (GR00T 5%) on the same task/harness?
- **RQ2 — Data efficiency:** demos (human videos) and sim episodes to reach a
  success threshold, vs the scripted-corpus route (150 eps → 5%).
- **RQ3 — Contact-rich reach:** can it do the dice-cup manipulation that the
  scripted controller struggles with (B2b)?
- **RQ4 — Sim-to-real gap:** success on the M3 Pro vs in sim.
- **RQ5 — Cost:** wall-clock + GPU-hours + disk per usable policy.

### Controlled comparison matrix (isolate one variable at a time)
| Data source | Learning method | Policy | Eval |
|---|---|---|---|
| Scripted corpus (existing) | GR00T-1.7 imitation | groot17_v* | closed-loop harness |
| **Video-derived (V2D)** | GR00T-1.7 imitation (interim) | v2d_imit | **same harness** |
| **Video-derived (V2D)** | **CHORD RL (Isaac Lab)** | v2d_chord | **same harness** |
| Scripted expert | — (upper bound) | — | 97.2% reference |

Same task, same seeds, same scorer ⇒ differences attributable to the data/method.

### Metrics (reuse our honest-scoring stack)
- **Task success** (first-try + with-retry), **placement error mm**, grasp/lift/
  release funnel — via `ludo_stats.py` + `session_summary.json` + the Inspect
  Robots `m3pro_pickplace` scorer (operator/VLM grader; **no privileged oracle**).
- **Data efficiency:** #human-demo-videos and #sim-episodes to threshold.
- **Compute cost:** GPU-hours, wall-clock, peak VRAM, disk (Docker images).
- **Sim-to-real:** matched-task success sim vs M3 Pro.

### Task suite (escalating, reuses existing assets)
- **T1** tabletop pick-carry-place (the Ludo token move — we have the scripted
  97.2% and GR00T 5% points already).
- **T2** dice-in-cup (B2a cup asset + B1 die exist and are verified).
- **T3** roll-in-cup → invert-dump → read (B1 reader) — the full dexterous case.

### Provenance & honesty (matches the audited theme)
Every run stamps `provenance.json` (git SHA, dirty flag, seed, host, GPU, CUDA,
model paths, argv, uncommitted-patch hash) as the executor already does; results
scored by the honest rule (success ≠ reward; failures logged as data). One
append-only **ledger** row per experiment. Negative results are results.

---

## 4. Phased integration (incremental; verify each phase before the next)

- **Phase 0 — Feasibility & setup (small, do first).** Clone the repo; inventory
  the Docker modules + their image sizes and GPU/VRAM needs; confirm which stages
  run on the RTX 5090 (32 GB) and/or Thor; check Docker disk headroom (Thor is at
  81% after the prune — reconstruction images may be large). Deliverable: a
  feasibility note + which stages are green.
- **Phase 1 — Ingestion + Reconstruction on one real demo.** Capture a short
  human video (operator: pick-place a token; then rolling dice in a cup). Run
  Stages 1–2 → action segments + object meshes + 6-DoF trajectories + hand motion.
  Validate the artifacts (typed `v2d_common` outputs on disk).
- **Phase 2 — Retarget to the Synria arm.** Use the released retargeting when it
  lands; interim, map the reconstructed 6-DoF object/hand trajectory to a Synria
  end-effector trajectory via our existing IK (`dls_step`) and import into the
  Isaac Lab env we already run. Deliverable: the arm reproducing the demo motion
  in sim.
- **Phase 3 — Train + evaluate.** CHORD RL in Isaac Lab when released; interim,
  imitation-train (LeRobot/GR00T) on the retargeted trajectories. Eval closed-loop
  on the **same harness** as GR00T eval01; fill the comparison matrix.
- **Phase 4 — Sim-to-real.** Deploy the best policy to the M3 Pro; measure RQ4.

Each phase → one ledger entry + honest scoring; stop/branch on what the data says.

---

## 5. Where it plugs into the repo
- New layer dir `video_to_data/` (git submodule or vendored) + `isaac/scripts/v2d_*`
  thin wrappers that `docker run` the modules and drop artifacts under
  `reports/v2d/<run>/`.
- Bridge: a converter from V2D 6-DoF trajectories → our LeRobot dataset schema
  (so the existing GR00T training + Inspect Robots eval consume it unchanged).
- Reuse `provenance.json`, `ludo_stats.py`, `session_summary.json`, and the
  Inspect Robots scorer verbatim → apples-to-apples with prior results.

---

## 6. Risks & unknowns (flag, then retire in Phase 0/1)
- **Stage 3 not released** → interim retarget + imitation; revisit when CHORD RL
  ships.
- **Embodiment fit:** CHORD is bimanual-dexterous/hand-focused; ours is a single
  6-DoF arm + parallel gripper. Retargeting a human *hand* trajectory to a
  parallel jaw is lossy — evaluate applicability honestly (may suit T1 pick-place
  better than T3 in-hand dice work). This is itself a finding.
- **GPU/VRAM/disk:** reconstruction (depth/mesh/pose models) may be heavy; verify
  on 5090 first, Thor second (watch Docker disk).
- **Capture rig:** need a calibrated-enough RGB(-D?) human demo; confirm what the
  reconstruction expects (mono RGB vs depth).

---

## 7. First concrete step
**Phase 0 feasibility only** (cheap, reversible): clone
`nvidia-isaac/video_to_data`, read its README/module manifest, list the Docker
images + requirements, and run a GPU/disk check on the 5090. Output: a one-page
"what runs here today" note that turns the unknowns above into facts before any
heavy work. Awaiting go-ahead to run Phase 0.

---

## Phase 0 — Feasibility results (2026-08-21) — DONE

Cloned `nvidia-isaac/video_to_data` (963 MB w/ LFS) and inventoried it against
our hardware.

**5090 box facts:** RTX 5090 **compute cap 12.0 (Blackwell / sm_120)**, driver
580.178.04, CUDA 13.0, Docker 29.1.3 + nvidia runtime + Container Toolkit, `uv`
present, Python 3.12, **573 GB free**.

**Verdict — what runs here today:**

| Stage | Runtime | On our HW? |
|---|---|---|
| **1. Video Ingestion Agent** | Python venv + vLLM server | **Likely YES** on the 5090 (Blackwell runs vLLM). Test: fits a VLM in 32 GB. |
| **2. Reconstruction** (HOI: cuVSLAM+TensorRT `v2d_cusfm`) | Docker, ≥24 GB VRAM, ref host 2× A6000 | **NO — hard-blocked.** README: sm_120 "not supported… the runtime check remains **mandatory**." Both our GPUs (5090, Thor) are Blackwell. |
| **3. Robotic Grounding** (retarget IK + RSL-RL PPO, Isaac Lab 2.3.1) | Docker; driver 580/CUDA 13 (**matches ours**) | **Likely YES** on the 5090. Ships real code (`run_retarget_local.sh`, `scripts/rsl_rl/`). Targets the **Sharpa** hand embodiment, not Synria. |

**Key structural finding (unblocks the evaluation):** the stages are decoupled.
Stage 3 runs off **public mocap datasets** (Arctic/TACO + MANO), NOT our own
reconstruction — `run_load_local.sh` (raw → MANO-FK parquet) → `run_retarget_local.sh`
(IK → processed) → Isaac Lab play. So the **reconstruction Blackwell-block does
NOT gate the core "demo-data → RL policy" claim** — we can evaluate Stage 3 on
the 5090 with the provided example sequences.

**What the block DOES cost us:** reconstructing *our own* novel videos (our robot
/ dice-cup demos) into 3D data is not possible on 5090/Thor. Options: (a) a cloud
A6000/L40S for reconstruction only; (b) wait for Blackwell `sm_120` support; (c)
use the released HF dataset `nvidia/video-to-data` (pre-reconstructed).

**Embodiment gap (as predicted):** grounding targets the Sharpa dexterous hand;
mapping to our 6-DoF parallel-jaw Synria is custom work — a documented finding,
not a quick win.

### Recommended methodical next unit — Phase 1 (revised)
Run **Stage 3 on the provided example sequences** end-to-end on the 5090
(build the Docker image → LOAD → retarget/IK → Isaac Lab play) to (a) prove the
grounding pipeline runs on our hardware and (b) see a retargeted policy play in
sim — the first tangible V2D result here. Reconstruction of our own videos and
Synria retargeting are deferred behind their (documented) blockers. In parallel,
a bounded test of Stage 1 ingestion on one of our own robot videos.

# Synria + NVIDIA Isaac GR00T — Fine-Tune Recipe

Interface correction, 2026-09-07: examples below use the recorded G3 migration
(`new_embodiment`, wrist/overhead, step-based training). Step counts are explicit
example budgets, not conversions from epochs. Historical environment, resource
and performance claims still require revalidation; see the [current handoff](handoff-2026-09-07.md).

End-to-end recipe for post-training **NVIDIA Isaac GR00T N1.7** on the Synria
6DOF arm for board-game pick-and-place. Eight phases from environment setup
through sim-to-real transfer planning, all runnable on the Linux RTX 5090
workstation.

Pairs with:

- [`isaac/isaaclab_tasks/synria_pickplace/gr00t/`](../isaac/isaaclab_tasks/synria_pickplace/gr00t/) — embodiment, dataset, finetune, inference scaffold
- [`isaac/isaaclab_tasks/synria_pickplace/`](../isaac/isaaclab_tasks/synria_pickplace/) — Isaac Lab MDP env (observations, rewards, terminations, trial state)
- [`docs/SYNRIA_PICK_AND_PLACE_TASK.md`](SYNRIA_PICK_AND_PLACE_TASK.md) — V1 task specification

**Why GR00T instead of PPO-from-scratch:** the LeRobot SO-101 single-arm
post-training recipe is published and well-documented. The Synria 6DOF arm is
structurally the same class of robot (6 revolute joints + parallel-jaw gripper
+ wrist camera), so the recipe carries over with embodiment-name changes.
Fine-tuning a foundation model converges with far less data than PPO-from-scratch
and produces a stronger portfolio signal.

---

## Reference documents

| Resource | URL |
|---|---|
| Post-Training GR00T N1.5 for SO-101 | [huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) |
| GR00T N1.7 Model Card | [huggingface.co/nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Isaac-GR00T repo (Apache-2.0) | [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| GR00T-Mimic synthetic motion pipeline | [developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning](https://developer.nvidia.com/blog/building-a-synthetic-motion-generation-pipeline-for-humanoid-robot-learning/) |
| Synthetic Manipulation Blueprint | [build.nvidia.com/nvidia/isaac-gr00t-synthetic-manipulation](https://build.nvidia.com/nvidia/isaac-gr00t-synthetic-manipulation/blueprintcard) |

---

## Hardware budget

| Resource | Requirement |
|---|---|
| GPU | RTX 5090 (32 GB VRAM) — sufficient for LoRA fine-tune at batch 8–16. Full fine-tune likely needs cloud GPU (A100/H100). |
| System RAM | ≥ 64 GB recommended (GR00T-Mimic amplification step loads large trajectory buffers) |
| Disk | Budget ~200 GB for amplified datasets (100K trajectories × frames) |
| Python version | **3.10 strict** — upstream `gr00t` pyproject pins `requires-python = "==3.10.*"`. Must be isolated from this repo's Python 3.11+ venv. |

---

## Phase 1 — Environment Setup

The GR00T venv must be strictly isolated from the Isaac Sim venv (Python version
conflict). Use two separate venvs:

| Venv | Python | Purpose |
|---|---|---|
| `~/code/Isaac-GR00T/.venv` | 3.10 (uv-managed) | GR00T fine-tuning + inference |
| `~/.venv/isaacsim5` | 3.12 | Isaac Sim + Isaac Lab + scene builders |

```bash
# 1. Clone the upstream repo (Apache-2.0)
git clone https://github.com/NVIDIA/Isaac-GR00T ~/code/Isaac-GR00T
cd ~/code/Isaac-GR00T

# 2. Create a Python 3.10 venv and install gr00t + all deps
uv sync

# 3. Smoke test
uv run python -c "import gr00t; print('GR00T import OK:', gr00t.__version__)"

# 4. In a separate shell, keep the Isaac Sim venv for env/scene work
source ~/.venv/isaacsim5/bin/activate
```

The wrapper scripts (`finetune.py`, `inference.py`) subprocess-invoke GR00T's
CLI entry points, so they can be called from the Isaac Sim venv as long as
`--gr00t-repo ~/code/Isaac-GR00T` is passed (which tells the wrapper to invoke
`uv run python gr00t/experiment/launch_finetune.py` from that directory).

---

## Phase 2 — Verify Synria Embodiment Values

The scaffold's
[`embodiment.py`](../isaac/isaaclab_tasks/synria_pickplace/gr00t/embodiment.py)
owns the Synria embodiment constants. These have been verified against the
SolidWorks-export URDF in `isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf`.

**Current values (verified):**

| Constant | Value |
|---|---|
| `SYNRIA_EMBODIMENT_TAG` | `"new_embodiment"` |
| `SYNRIA_ARM_JOINT_NAMES` | `["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]` |
| `SYNRIA_GRIPPER_JOINT_NAMES` | `["left_finger", "right_finger"]` |
| `SYNRIA_ACTION_DIM` | `8` (6 arm + 2 finger) |
| `SYNRIA_CAMERA_KEYS` | `["wrist", "overhead"]` |
| `SYNRIA_CAMERA_RES` | `(224, 224)` |

**Re-verify against the URDF before any fine-tune run** (the SolidWorks export
may change if the vendor updates the URDF):

```bash
python3 -c "
import xml.etree.ElementTree as ET
root = ET.parse('isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf').getroot()
joints = [(j.attrib['name'], j.attrib.get('type')) for j in root.findall('joint')]
for name, jtype in joints:
    print(f'{jtype:12s}  {name}')
"
```

Expected output:

```
revolute      Joint1
revolute      Joint2
revolute      Joint3
revolute      Joint4
revolute      Joint5
revolute      Joint6
prismatic     left_finger
prismatic     right_finger
fixed         tool0_fixed_joint
```

If the joint names differ from `SYNRIA_ARM_JOINT_NAMES` in `embodiment.py`,
update that file before collecting demonstrations. The names must match the
joint names GR00T will observe in both training data and inference.

**Inspect the current modality config (no GPU required):**

```bash
physical-ai-lab gr00t-embodiment
physical-ai-lab gr00t-embodiment --json
```

---

## Phase 3 — Seed Dataset Collection

Goal: **~100 LeRobot-format demonstrations per game** (chess, checkers, ludo).

Output paths (defined in `dataset.py`):

```text
reports/training/synria_chess_lerobot_demos/
reports/training/synria_checkers_lerobot_demos/
reports/training/synria_ludo_lerobot_demos/
```

### Option A — Teleoperated demos (highest quality)

Use a leader arm (Synria leader variant) or a 3D mouse / VR controller to
teleoperate the physical Synria arm. Record joint state + actions + wrist camera
at 30 Hz. LeRobot's CLI handles recording:

```bash
# One session per game (~100 episodes × ~30 s = ~50 min per game)
lerobot-record \
    --robot synria \
    --task synria_chess_pick \
    --num-episodes 100 \
    --output reports/training/synria_chess_lerobot_demos
```

This requires the Synria arm hardware and a leader arm or controller.
Produces the highest-quality seed data.

### Option B — Scripted demos from Isaac Lab (faster, lower quality)

Write a deterministic pick-and-place policy using joint waypoints inside the
Isaac Lab task (no learning required — just hardcoded keyframes or MoveIt-like
IK). Roll out 100 episodes with randomized piece + zone selections:

```bash
source ~/.venv/isaacsim5/bin/activate

python isaac/scripts/collect_synria_demos.py \
    --task Synria-Chess-PickPlace-v0 \
    --num-episodes 100 \
    --output reports/training/synria_chess_lerobot_demos \
    --headless
```

The scripted policy captures state + actions in LeRobot format. Lower-quality
seed data (no generalisation from human motion), but fast to collect and
avoids hardware dependency.

**Budget:** ~100 demos per game is the SO-101 recipe baseline. More seed demos
produce better fine-tuned policies; the GR00T-Mimic step in Phase 4 multiplies
coverage regardless.

---

## Phase 4 — GR00T-Mimic Amplification (optional, recommended)

GR00T-Mimic uses NVIDIA Cosmos world-foundation models to amplify a small seed
set into a much larger synthetic dataset. NVIDIA published ~780K synthetic
trajectories from a small seed in ~11 hours on an H100 cluster with a ~40%
downstream performance lift.

```bash
# Single game
gr00t-mimic amplify \
    --input  reports/training/synria_chess_lerobot_demos \
    --output reports/training/synria_chess_mimic_amplified \
    --target-trajectories 100000 \
    --cosmos-model nvidia/Cosmos-1.0

# All three games in sequence
for game in chess checkers ludo; do
    gr00t-mimic amplify \
        --input  reports/training/synria_${game}_lerobot_demos \
        --output reports/training/synria_${game}_mimic_amplified \
        --target-trajectories 100000
done
```

Expected output paths (match `dataset.py` `SYNRIA_MIMIC_AMPLIFIED_ROOTS`):

```text
reports/training/synria_chess_mimic_amplified/
reports/training/synria_checkers_mimic_amplified/
reports/training/synria_ludo_mimic_amplified/
```

**Runtime estimate on RTX 5090:** 8–16 hours per game. The published H100
benchmark was ~780K in 11 h; RTX 5090 throughput will be lower.

**Skip this phase** if you want to validate the fine-tune pipeline first
(Path A — direct fine-tune on seed demos only). Add amplification after the
pipeline is proven end-to-end.

---

## Phase 5 — Fine-Tune

Two invocation styles. Option B (the repo wrapper) is preferred because it
computes dataset paths from `--game`, injects the Synria embodiment tag, and
writes a `wrapper_manifest.json` for reproducibility.

### Inspect the command first (no GPU, no demos required)

```bash
# Dry-run: print the full launch command without invoking anything
physical-ai-lab gr00t-finetune-dry-run --game chess
physical-ai-lab gr00t-finetune-dry-run --game chess --amplified

# Or via the module directly:
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \
    --game chess --dry-run
```

### Option A — Direct invocation (matches the official upstream README)

From inside the `Isaac-GR00T` repo with the GR00T uv venv:

```bash
cd ~/code/Isaac-GR00T

uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/physical-ai-jetson-robotics/reports/training/synria_chess_mimic_amplified \
    --output-dir /path/to/physical-ai-jetson-robotics/runs/gr00t_synria_chess_v0 \
    --embodiment-tag new_embodiment \
    --modality-config-path /path/to/physical-ai-jetson-robotics/isaac/isaaclab_tasks/synria_pickplace/gr00t/modality_config.py \
    --max-steps 10000 \
    --global-batch-size 16 \
    --lora-rank 16
```

### Option B — Repo wrapper (recommended)

The wrapper resolves dataset paths automatically from `--game` + `--amplified`,
injects `--embodiment-tag new_embodiment`, and writes
`runs/gr00t_synria_<game>_v0/wrapper_manifest.json` with the full resolved
command for reproducibility. Anything after `--extra` is forwarded verbatim to
GR00T's `launch_finetune.py`.

```bash
# Chess, amplified dataset, LoRA rank 16
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \
    --game chess \
    --amplified \
    --base-model nvidia/GR00T-N1.7-3B \
    --output-dir runs/gr00t_synria_chess_v0 \
    --gr00t-repo ~/code/Isaac-GR00T \
    --max-steps 10000 \
    --global-batch-size 16 \
    --extra \
    --lora-rank 16 \
    --max-steps 50000

# Checkers, seed demos only (skip amplification)
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \
    --game checkers \
    --base-model nvidia/GR00T-N1.7-3B \
    --output-dir runs/gr00t_synria_checkers_v0 \
    --gr00t-repo ~/code/Isaac-GR00T \
    --max-steps 10000 \
    --global-batch-size 16 \
    --extra \
    --lora-rank 16

# Ludo, amplified dataset
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \
    --game ludo \
    --amplified \
    --base-model nvidia/GR00T-N1.7-3B \
    --output-dir runs/gr00t_synria_ludo_v0 \
    --gr00t-repo ~/code/Isaac-GR00T \
    --max-steps 10000 \
    --global-batch-size 16 \
    --extra \
    --lora-rank 16
```

The wrapper writes `runs/gr00t_synria_<game>_v0/wrapper_manifest.json`:

```json
{
  "wrapper": "isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune",
  "game": "chess",
  "amplified": true,
  "base_model": "nvidia/GR00T-N1.7-3B",
  "command": ["python", "/path/to/launch_finetune.py", ...]
}
```

**LoRA is essential on a single RTX 5090.** Full fine-tune of N1.7-3B at FP16
likely exhausts 32 GB VRAM. Start with `--lora-rank 16`; drop to 8 if OOM.

**Outputs** (under `runs/` which is gitignored):

```text
runs/gr00t_synria_chess_v0/
├── wrapper_manifest.json    ← commit this (metadata only, no weights)
├── checkpoint-*/            ← LoRA adapters (.safetensors) — do NOT commit
└── logs/                    ← training curves — commit summary JSON
```

---

## Phase 6 — Evaluate Inside Isaac Lab

Roll out the fine-tuned policy against the registered Synria task IDs and
capture V1 success rates. The `inference.py` module loads the GR00T checkpoint
and steps the Isaac Lab env.

```bash
source ~/.venv/isaacsim5/bin/activate  # needs Isaac Lab + gymnasium

# Chess
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.inference \
    --task Synria-Chess-PickPlace-v0 \
    --checkpoint runs/gr00t_synria_chess_v0 \
    --base-model nvidia/GR00T-N1.7-3B \
    --num-episodes 20 \
    --instruction "pick the white pawn and move it to the right staging zone, then return it" \
    --report reports/training/gr00t_synria_chess_eval.json

# Checkers
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.inference \
    --task Synria-Checkers-PickPlace-v0 \
    --checkpoint runs/gr00t_synria_checkers_v0 \
    --base-model nvidia/GR00T-N1.7-3B \
    --num-episodes 20 \
    --instruction "pick the red checker and move it to the left staging zone, then return it" \
    --report reports/training/gr00t_synria_checkers_eval.json

# Ludo
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.inference \
    --task Synria-Ludo-PickPlace-v0 \
    --checkpoint runs/gr00t_synria_ludo_v0 \
    --base-model nvidia/GR00T-N1.7-3B \
    --num-episodes 20 \
    --instruction "pick the blue token and move it to the top staging zone, then return it" \
    --report reports/training/gr00t_synria_ludo_eval.json
```

**V1 success criteria** (from `docs/SYNRIA_PICK_AND_PLACE_TASK.md`):

- Piece centred within ±10 mm of its original board square after the full
  PICK → PLACE → PICK → RETURN sequence
- Piece did not drop below `TABLE_SURFACE_Z − 0.05 m` at any point
- No arm collision force > 0.1 N on non-target links

Eval report format (`gr00t_synria_<game>_eval.json`):

```json
{
  "task": "Synria-Chess-PickPlace-v0",
  "checkpoint": "runs/gr00t_synria_chess_v0",
  "num_episodes": 20,
  "success_rate": 0.0,
  "episodes": [
    { "episode": 0, "success": false, "phase_reached": "PICK_FROM_BOARD", "steps": 412 }
  ]
}
```

Push the eval JSON to `reports/training/` as evidence regardless of the
success rate — failed runs are valid evidence of the pipeline working.

---

## Phase 7 — Iterate

Typical iteration loop after first eval:

### 7a — Embodiment mismatch

If the policy produces random joint motion, the joint ordering in training data
may not match `SYNRIA_ARM_JOINT_NAMES`. Diagnose:

```bash
# Inspect the recorded action ordering and GR00T modality slices.
python -c "
import json
from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_DEMO_ROOTS
meta = SYNRIA_DEMO_ROOTS['chess'] / 'meta'
info = json.loads((meta / 'info.json').read_text())
modality = json.loads((meta / 'modality.json').read_text())
print('Recorded action feature:', info['features']['action'])
print('GR00T action slices:', modality['action'])
"
```

If the ordering differs, fix `SYNRIA_ARM_JOINT_NAMES` in `embodiment.py`
(and re-run the CLI tests: `pytest tests/test_cli_gr00t.py -v`), re-order
the action columns in the LeRobot dataset, then re-fine-tune.

### 7b — Insufficient generalisation

If success rate stalls at ~50%:

1. Collect more seed demos specifically for failure cases (corner pieces,
   far staging zones, edge-of-reach positions)
2. Re-run GR00T-Mimic amplification on the expanded seed
3. Re-fine-tune with the larger amplified dataset

### 7c — Imitation ceiling + RL lift

If pure imitation fine-tuning stalls below the 85% target, layer Isaac Lab RL
on top: use the fine-tuned GR00T weights as the policy initialization and run
PPO from there. The Isaac Lab MDP reward structure
(`isaac/isaaclab_tasks/synria_pickplace/mdp_logic.py`) was designed for exactly
this use case.

```bash
# Initialize PPO from fine-tuned GR00T weights (conceptual — adjust to
# your Isaac Lab version's RL trainer interface)
python -m isaaclab.scripts.train \
    --task Synria-Chess-PickPlace-v0 \
    --headless \
    --num_envs 512 \
    --init-policy runs/gr00t_synria_chess_v0
```

---

## Phase 8 — Sim-to-Real Transfer (post-training milestone)

Once Phase 6 reports ≥ 85% V1 success in sim across all three games, the next
milestone is transfer to the physical Synria arm on Jetson AGX Thor.

This phase is out of scope for this doc. Key references:

- [`docs/HARDWARE.md`](HARDWARE.md) — Thor hardware bring-up plan
- [`docs/JETSON_DEPLOYMENT.md`](JETSON_DEPLOYMENT.md) — edge deployment workflow
- [`docs/VENDOR_INTEGRATION_MAP.md`](VENDOR_INTEGRATION_MAP.md) — URDF-limits-vs-firmware-limits rule

**Gate**: do not send policy commands to the physical arm until:

1. URDF joint limits have been verified against the firmware SDK limits
2. A contact-force safety monitor is active (stops motion if force > threshold)
3. GR00T's policy server pattern (Pattern B in `inference.py`) has been validated
   in sim so the policy and the Isaac Lab env run in separate processes — the
   same architecture used on real hardware

---

## What to commit after each phase

Large checkpoints (`.safetensors`, LoRA adapters > 100 MB) belong outside git
under `runs/` (gitignored). Push them to HuggingFace Hub if you want them
downloadable. Everything else belongs in the repo as evidence:

| Artifact | Path | When |
|---|---|---|
| Verified URDF joint one-liner output | `reports/training/synria_urdf_joint_verify.txt` | Phase 2 |
| Seed demo count + summary | `reports/training/gr00t_<game>_seed_summary.json` | Phase 3 |
| Mimic amplification manifest (counts only, no payload) | `reports/training/gr00t_mimic_<game>_manifest.json` | Phase 4 |
| Wrapper manifest | `runs/gr00t_synria_<game>_v0/wrapper_manifest.json` | Phase 5 |
| Fine-tune training log (loss curve summary) | `reports/training/gr00t_<game>_finetune_log.json` | Phase 5 |
| Per-episode eval results | `reports/training/gr00t_synria_<game>_eval.json` | Phase 6 |
| Rollout screenshot / GIF | `reports/training/<game>_v1_rollout.{png,gif}` | Phase 6 |
| Updated `embodiment.py` (if joint names changed) | `isaac/isaaclab_tasks/synria_pickplace/gr00t/embodiment.py` | Phase 7a |

---

## Quick CLI reference (no GPU required)

```bash
# Show current embodiment config table
physical-ai-lab gr00t-embodiment

# Show as JSON (machine-readable)
physical-ai-lab gr00t-embodiment --json

# Print the resolved fine-tune launch command without invoking it
physical-ai-lab gr00t-finetune-dry-run --game chess
physical-ai-lab gr00t-finetune-dry-run --game chess --amplified
physical-ai-lab gr00t-finetune-dry-run --game chess --amplified --max-steps 20000 --global-batch-size 8

# Run all CLI tests (no GPU required)
pytest tests/test_cli_gr00t.py -v          # 36 tests
pytest tests/test_gr00t_scaffold_import.py -v  #  5 tests
```

---

## Honest caveats

- **GR00T API churn.** N1 → N1.5 → N1.7 has changed module paths. Pin a
  specific git SHA (`git pin ~/code/Isaac-GR00T`) before a fine-tune run.
  `gr00t.policy.gr00t_policy.Gr00tPolicy` is the N1.7 name; earlier versions
  used different paths. `inference.py` has a `TODO(RTX)` marker where the exact
  `from_pretrained` signature needs to be confirmed against your installed
  version.

- **Embodiment registration friction.** GR00T's dataset loader validates action
  key names against the registered embodiment. If the key names in
  `synria_modality_config()` don't match what the dataset loader expects, you
  get a silent shape mismatch or a validation error. Run the embodiment
  verification (Phase 2) every time you update the GR00T version.

- **Demo collection is real work.** Path A (direct fine-tune on ~100 seed demos)
  is only as good as those demos. Scripted demos (Option B) produce consistent
  but narrow coverage; teleop demos (Option A) produce diverse but slow-to-collect
  training signal. Budget accordingly.

- **VRAM is tight.** 32 GB handles LoRA-16 fine-tune of N1.7-3B at FP16 with
  batch size 8–16. If you see OOM: lower `--global-batch-size` first, then lower
  `--lora-rank`. Full fine-tune of 3B almost certainly needs an H100.

- **Two-venv friction.** Python 3.10 (GR00T) and Python 3.12 (Isaac Sim) cannot
  co-exist in a single venv. The wrapper scripts use `--gr00t-repo` + subprocess
  to bridge them, but that means the GR00T process cannot directly access Isaac
  Lab's GPU context. For Pattern B (policy server), this is the intended
  architecture; for Pattern A (in-process), both packages must be in one venv
  which means pinning Isaac Lab to a Python 3.10 build — non-trivial.

- **Synria URDF source of truth.** Use the URDF in
  `isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf` for all embodiment
  values. Do not mix in the Yahboom M3 Pro URDF (which has a different arm with
  5 joints named `arm1..arm5`). The URDF one-liner in Phase 2 confirms you are
  reading the right file.

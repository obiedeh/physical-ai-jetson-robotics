#!/usr/bin/env bash
# Train the Synria pick-and-place policy with RSL-RL PPO inside Isaac Lab.
#
# Prerequisites (run in order):
#   1. bash scripts/linux_rtx/verify_rtx_env.sh          — check env
#   2. bash scripts/linux_rtx/import_synria_urdf.sh       — build arm USD
#   3. bash scripts/linux_rtx/build_synria_chess_scene.sh — build scene USD
#   4. bash scripts/linux_rtx/smoke_isaaclab_env.sh       — validate env stack
#   5. this script                                        — full training run
#
# Usage:
#   bash scripts/linux_rtx/train_isaaclab_synria.sh
#
# Environment variable overrides (set before calling the script):
#   ISAAC_PYTHON   — path to Isaac Sim 5.1 Python (default: ~/.venv/isaacsim5/bin/python)
#   TRAIN_TASK     — gym task ID             (default: Synria-Chess-PickPlace-v0)
#   TRAIN_NUM_ENVS — parallel environments   (default: 4096)
#   TRAIN_ITERS    — PPO iterations          (default: 1500)
#   TRAIN_DEVICE   — cuda or cpu             (default: cuda)
#   TRAIN_SEED     — RNG seed                (default: 42)
#
# Smoke-test override (fast validation, ~2 min):
#   TRAIN_NUM_ENVS=16 TRAIN_ITERS=5 bash scripts/linux_rtx/train_isaaclab_synria.sh
#
# Multi-game sweep (run all three tasks back-to-back):
#   for TASK in Synria-Chess-PickPlace-v0 Synria-Checkers-PickPlace-v0 Synria-Ludo-PickPlace-v0; do
#     TRAIN_TASK=$TASK bash scripts/linux_rtx/train_isaaclab_synria.sh
#   done

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"

# Isaac Sim 5.1 local install env vars (required for physics/extension loading)
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${HOME}/isaacsim}"
export ACCEPT_EULA=Y
export CARB_APP_PATH="$ISAACSIM_ROOT/kit"
export ISAAC_PATH="$ISAACSIM_ROOT"
export EXP_PATH="$ISAACSIM_ROOT/apps"
export LD_PRELOAD="$ISAACSIM_ROOT/kit/libcarb.so"
TRAIN_TASK="${TRAIN_TASK:-Synria-Chess-PickPlace-v0}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-1500}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TRAIN_SEED="${TRAIN_SEED:-42}"

TRAIN_SCRIPT="$REPO_ROOT/isaac/scripts/train_synria_pickplace.py"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "ERROR: Isaac Sim Python not found: $ISAAC_PYTHON" >&2
  echo "       Run: bash scripts/linux_rtx/verify_rtx_env.sh" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

ARM_USD="$REPO_ROOT/isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.usda"
if [[ ! -f "$ARM_USD" ]]; then
  echo "ERROR: Synria arm USD not found. Run: bash scripts/linux_rtx/import_synria_urdf.sh" >&2
  exit 1
fi

# Derive scene USD from task name
case "$TRAIN_TASK" in
  *Chess*)    SCENE_USD="$REPO_ROOT/isaac/usd/scenes/synria_chess/synria_chess_v0.usda" ;;
  *Checkers*) SCENE_USD="$REPO_ROOT/isaac/usd/scenes/synria_checkers/synria_checkers_v0.usda" ;;
  *Ludo*)     SCENE_USD="$REPO_ROOT/isaac/usd/scenes/synria_ludo/synria_ludo_v0.usda" ;;
  *)          SCENE_USD="" ;;
esac

if [[ -n "$SCENE_USD" && ! -f "$SCENE_USD" ]]; then
  echo "ERROR: Scene USD not found: $SCENE_USD" >&2
  echo "       Run the matching build_synria_*_scene.sh first." >&2
  exit 1
fi

# ── Launch ───────────────────────────────────────────────────────────────────

echo "========================================"
echo " Synria Pick-and-Place RL Training"
echo " Task      : $TRAIN_TASK"
echo " Num envs  : $TRAIN_NUM_ENVS"
echo " Iterations: $TRAIN_ITERS"
echo " Device    : $TRAIN_DEVICE"
echo " Seed      : $TRAIN_SEED"
echo " Output    : $REPO_ROOT/reports/training/"
echo "========================================"

# Vulkan ICD for headless GPU rendering (Isaac Sim requires this on some systems)
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ -f "$NVIDIA_ICD" ]]; then
  export VK_ICD_FILENAMES="$NVIDIA_ICD"
fi

"$ISAAC_PYTHON" "$TRAIN_SCRIPT" \
  --task        "$TRAIN_TASK" \
  --num_envs    "$TRAIN_NUM_ENVS" \
  --max_iterations "$TRAIN_ITERS" \
  --device      "$TRAIN_DEVICE" \
  --seed        "$TRAIN_SEED" \
  --headless

echo "========================================"
echo " Training complete."
echo " Checkpoints: $REPO_ROOT/reports/training/"
echo " Results    : $REPO_ROOT/reports/training/*/training_results.json"
echo ""
echo " Next steps:"
echo "   - Open TensorBoard: tensorboard --logdir reports/training/"
echo "   - Commit artifacts: git add reports/training/ && git commit"
echo "========================================"

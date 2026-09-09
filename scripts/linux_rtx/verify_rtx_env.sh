#!/usr/bin/env bash
# Verify the RTX 5090 workstation has everything needed to run Isaac Lab training.
#
# Run this first — before building USD scenes or launching training.
# All checks print PASS / FAIL; the script exits 0 only if every check passes.
#
# Usage:
#   bash scripts/linux_rtx/verify_rtx_env.sh
#   # Override Isaac Sim Python:
#   ISAAC_PYTHON=/path/to/isaacsim5/bin/python bash scripts/linux_rtx/verify_rtx_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS="${GREEN}PASS${NC}"; FAIL="${RED}FAIL${NC}"; WARN="${YELLOW}WARN${NC}"

errors=0

check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf "  %-45s %b\n" "$label" "$PASS"
  else
    printf "  %-45s %b\n" "$label" "$FAIL"
    errors=$((errors + 1))
  fi
}

check_warn() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf "  %-45s %b\n" "$label" "$PASS"
  else
    printf "  %-45s %b  (non-fatal)\n" "$label" "$WARN"
  fi
}

echo "========================================"
echo " RTX 5090 Environment Verification"
echo " Repo: $REPO_ROOT"
echo " Isaac Sim Python: $ISAAC_PYTHON"
echo "========================================"

# ── GPU ──────────────────────────────────────────────────────────────────────
echo
echo "── GPU ──"
check "nvidia-smi available"                command -v nvidia-smi
check "GPU listed by nvidia-smi"            bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "rtx\|a100\|h100\|l40\|geforce"'
check "CUDA available (torch)"              bash -c 'python3 -c "import torch; assert torch.cuda.is_available()"' 2>/dev/null || \
  check "CUDA available (nvcc)"             bash -c 'nvcc --version'

# ── Isaac Sim Python ─────────────────────────────────────────────────────────
echo
echo "── Isaac Sim Python ──"
check "Isaac Sim Python exists"             test -x "$ISAAC_PYTHON"
check "isaacsim importable"                 "$ISAAC_PYTHON" -c "import isaacsim"
check "omni.isaac.core importable"          "$ISAAC_PYTHON" -c "import omni.isaac.core"
check "pxr (USD) importable"               "$ISAAC_PYTHON" -c "from pxr import Usd"

# ── Isaac Lab ────────────────────────────────────────────────────────────────
echo
echo "── Isaac Lab ──"
check "isaaclab importable"                 "$ISAAC_PYTHON" -c "import isaaclab"
check "isaaclab.envs importable"            "$ISAAC_PYTHON" -c "from isaaclab.envs import ManagerBasedRLEnv"
check "isaaclab.assets importable"          "$ISAAC_PYTHON" -c "from isaaclab.assets import ArticulationCfg"
check "isaaclab_rl importable"              "$ISAAC_PYTHON" -c "from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner"
check "gymnasium importable"               "$ISAAC_PYTHON" -c "import gymnasium"
check "torch importable"                   "$ISAAC_PYTHON" -c "import torch"

# ── Synria task package ───────────────────────────────────────────────────────
echo
echo "── Synria task package ──"
check "Repo on PYTHONPATH (isaaclab_tasks)" \
  "$ISAAC_PYTHON" -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
import isaac.isaaclab_tasks.synria_pickplace
"
check "Synria tasks registered in gym" \
  "$ISAAC_PYTHON" -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
import isaac.isaaclab_tasks.synria_pickplace
import gymnasium as gym
assert 'Synria-Chess-PickPlace-v0' in gym.envs.registry
"
check "train_cfg importable" \
  "$ISAAC_PYTHON" -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
assert len(TASK_TRAIN_CFG) == 3
"

# ── USD assets ───────────────────────────────────────────────────────────────
echo
echo "── USD assets ──"
check "Synria arm USDA exists" \
  test -f "$REPO_ROOT/isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.usda"
check_warn "Chess scene USDA exists" \
  test -f "$REPO_ROOT/isaac/usd/scenes/synria_chess/synria_chess_v0.usda"
check_warn "Checkers scene USDA exists" \
  test -f "$REPO_ROOT/isaac/usd/scenes/synria_checkers/synria_checkers_v0.usda"
check_warn "Ludo scene USDA exists" \
  test -f "$REPO_ROOT/isaac/usd/scenes/synria_ludo/synria_ludo_v0.usda"

# ── Training script ───────────────────────────────────────────────────────────
echo
echo "── Training entry point ──"
check "train_synria_pickplace.py exists" \
  test -f "$REPO_ROOT/isaac/scripts/train_synria_pickplace.py"
check "training output dir writable" \
  bash -c "mkdir -p '$REPO_ROOT/reports/training' && touch '$REPO_ROOT/reports/training/.write_test' && rm '$REPO_ROOT/reports/training/.write_test'"

# ── Result ────────────────────────────────────────────────────────────────────
echo
echo "========================================"
if [[ $errors -eq 0 ]]; then
  printf " %b  All checks passed — ready to train.\n" "$PASS"
  echo ""
  echo " Next steps:"
  if [[ ! -f "$REPO_ROOT/isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.usda" ]]; then
    echo "   1. Import URDF:   bash scripts/linux_rtx/import_synria_urdf.sh"
    echo "   2. Build scenes:  bash scripts/linux_rtx/build_synria_chess_scene.sh"
    echo "   3. Smoke test:    bash scripts/linux_rtx/smoke_isaaclab_env.sh"
    echo "   4. Full training: bash scripts/linux_rtx/train_isaaclab_synria.sh"
  else
    echo "   1. Build scenes:  bash scripts/linux_rtx/build_synria_chess_scene.sh"
    echo "   2. Smoke test:    bash scripts/linux_rtx/smoke_isaaclab_env.sh"
    echo "   3. Full training: bash scripts/linux_rtx/train_isaaclab_synria.sh"
  fi
else
  printf " %b  %d check(s) failed — fix the above before training.\n" "$FAIL" "$errors"
  echo ""
  echo " If Isaac Sim is not installed, see docs/SETUP_RTX.md."
fi
echo "========================================"

exit $errors

#!/usr/bin/env bash
# Isaac Lab baseline diagnostic — is the INSTALL broken, or is OUR env config?
#
# Context: Synria training never reaches a PPO iteration. It dies at
# `sim.reset()` — either deadlocking in SimulationContext's timeline-STOP
# handler (busy-loop at 100% CPU) or exiting 0 with a misleading
# "Training complete". Root cause analysis: docs/handoff-2026-07-27-1937-CDT.md.
#
# This script runs a STOCK Isaac Lab task (Isaac-Cartpole-v0) through Isaac
# Lab's OWN train.py, using the same venv, env vars and headless flags as
# scripts/linux_rtx/train_isaaclab_synria.sh. Nothing of ours is involved
# except the environment.
#
#   Cartpole TRAINS  -> the install is fine; the bug is in our env_cfg /
#                       train script. Debug Synria-*-PickPlace-v0.
#   Cartpole FAILS   -> the Isaac Lab / Isaac Sim 5.1 install is broken.
#                       Fix the install; our task config is not the problem.
#
# It also runs with `--/app/fastShutdown=False` so a failing teardown raises a
# real traceback instead of a silent `exit 0` (handoff step 1).
#
# Usage:
#   bash scripts/linux_rtx/diagnose_isaaclab_baseline.sh
#
# Environment variable overrides:
#   ISAAC_PYTHON    — Isaac Sim 5.1 Python  (default: ~/.venv/isaacsim5/bin/python)
#   ISAACLAB_ROOT   — Isaac Lab checkout    (default: ~/IsaacLab)
#   DIAG_TASK       — task to baseline      (default: Isaac-Cartpole-v0)
#   DIAG_NUM_ENVS   — parallel envs         (default: 32)
#   DIAG_ITERS      — PPO iterations        (default: 3)
#   DIAG_TIMEOUT    — hard wall-clock cap   (default: 600 seconds)
#   DIAG_HEADLESS   — 1 headless, 0 GUI     (default: 1)
#
# The timeout matters: the known failure mode is an infinite busy-loop, so an
# unbounded run would hang this script forever.

set -uo pipefail   # NOTE: no -e; we inspect the exit code ourselves.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
DIAG_TASK="${DIAG_TASK:-Isaac-Cartpole-v0}"
DIAG_NUM_ENVS="${DIAG_NUM_ENVS:-32}"
DIAG_ITERS="${DIAG_ITERS:-3}"
DIAG_TIMEOUT="${DIAG_TIMEOUT:-600}"
DIAG_HEADLESS="${DIAG_HEADLESS:-1}"

STOCK_TRAIN="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py"
LOG_DIR="${REPO_ROOT}/reports/diagnostics"
LOG_FILE="${LOG_DIR}/isaaclab_baseline_$(date +%Y%m%d_%H%M%S).log"

# ── Pre-flight: GPU ──────────────────────────────────────────────────────────
# A driver userspace/kernel mismatch (e.g. after an unattended nvidia upgrade
# without a reboot) makes CUDA fail with "Error 804: forward compatibility was
# attempted on non supported HW". Isaac Sim then fails in confusing ways that
# look exactly like an Isaac Lab bug. Check this FIRST — it costs one second
# and it has already burned one session.

echo "== Pre-flight =============================================="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARN: nvidia-smi not found; skipping driver check." >&2
elif ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed — the NVIDIA driver is not usable." >&2
  echo >&2
  nvidia-smi 2>&1 | sed 's/^/       /' >&2
  echo >&2
  KERNEL_VER="$(sed -n 's/^NVRM version:.*Module *for *[^ ]* *\([0-9.]*\).*/\1/p' \
                /proc/driver/nvidia/version 2>/dev/null)"
  echo "       Running kernel module : ${KERNEL_VER:-unknown}" >&2
  echo "       Installed userspace   : $(dpkg -l 2>/dev/null | awk '/nvidia-driver-[0-9]/{print $3; exit}')" >&2
  echo >&2
  echo "       If those differ, the driver was upgraded without a reboot." >&2
  echo "       Fix: reboot, then re-run this script." >&2
  exit 2
fi

CUDA_CHECK="$("$ISAAC_PYTHON" -c 'import torch; print("OK" if torch.cuda.is_available() else "NO_CUDA")' 2>/dev/null | tail -1)"
if [[ "$CUDA_CHECK" != "OK" ]]; then
  echo "ERROR: torch.cuda.is_available() is False in the Isaac venv." >&2
  echo "       Isaac Sim cannot run. Diagnose before going further:" >&2
  echo "         $ISAAC_PYTHON -c 'import torch; torch.cuda.device_count()'" >&2
  echo "       Most common cause: driver upgraded without a reboot." >&2
  exit 2
fi
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CUDA       : available"

# ── Pre-flight: paths ────────────────────────────────────────────────────────

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "ERROR: Isaac Sim Python not found: $ISAAC_PYTHON" >&2
  exit 1
fi

if [[ ! -f "$STOCK_TRAIN" ]]; then
  echo "ERROR: Isaac Lab stock trainer not found: $STOCK_TRAIN" >&2
  echo "       Set ISAACLAB_ROOT to your Isaac Lab checkout." >&2
  exit 1
fi

# ── Isaac Sim 5.1 local-install env vars (must match train_isaaclab_synria.sh) ──

ISAACSIM_ROOT="${ISAACSIM_ROOT:-${HOME}/isaacsim}"
export ACCEPT_EULA=Y
export CARB_APP_PATH="$ISAACSIM_ROOT/kit"
export ISAAC_PATH="$ISAACSIM_ROOT"
export EXP_PATH="$ISAACSIM_ROOT/apps"
export LD_PRELOAD="$ISAACSIM_ROOT/kit/libcarb.so"

NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ -f "$NVIDIA_ICD" ]]; then
  export VK_ICD_FILENAMES="$NVIDIA_ICD"
fi

mkdir -p "$LOG_DIR"

HEADLESS_FLAG=()
if [[ "$DIAG_HEADLESS" == "1" ]]; then
  HEADLESS_FLAG=(--headless)
fi

echo
echo "== Isaac Lab baseline diagnostic ==========================="
echo "Task       : $DIAG_TASK   (stock Isaac Lab task — none of our code)"
echo "Trainer    : $STOCK_TRAIN"
echo "Num envs   : $DIAG_NUM_ENVS"
echo "Iterations : $DIAG_ITERS"
echo "Headless   : $DIAG_HEADLESS"
echo "Timeout    : ${DIAG_TIMEOUT}s"
echo "Log        : $LOG_FILE"
echo "============================================================"
echo

# fastShutdown=False: make a failing teardown raise a real traceback instead of
# exiting 0 silently and printing a misleading "complete".
timeout --signal=INT --kill-after=30 "$DIAG_TIMEOUT" \
  "$ISAAC_PYTHON" "$STOCK_TRAIN" \
    --task "$DIAG_TASK" \
    --num_envs "$DIAG_NUM_ENVS" \
    --max_iterations "$DIAG_ITERS" \
    "${HEADLESS_FLAG[@]}" \
    --kit_args="--/app/fastShutdown=False" \
  2>&1 | tee "$LOG_FILE"

STATUS="${PIPESTATUS[0]}"

# ── Verdict ──────────────────────────────────────────────────────────────────
# Exit 0 is NOT sufficient evidence of success: the whole point of this bug is
# that the run exits 0 without training. Require proof of a real iteration.

echo
echo "== Verdict ================================================="

ITER_HITS="$(grep -cE "Learning iteration|Mean reward|Computation:" "$LOG_FILE" 2>/dev/null || true)"

if [[ "$STATUS" -eq 124 || "$STATUS" -eq 137 ]]; then
  echo "RESULT: TIMEOUT after ${DIAG_TIMEOUT}s — the stock task HANGS too."
  echo
  echo "  => The Isaac Lab / Isaac Sim 5.1 INSTALL is broken, not our config."
  echo "     This is the same deadlock signature as Synria training."
  echo "     Fix the install before touching env_cfg.py again."
  VERDICT=1
elif [[ "$ITER_HITS" -gt 0 ]]; then
  echo "RESULT: PASS — the stock task reached $ITER_HITS training iteration marker(s)."
  echo
  echo "  => The install is HEALTHY. The bug is in OUR task config."
  echo "     Next: diff our env_cfg.py against the cartpole env cfg, and"
  echo "     re-run Synria with --kit_args \"--/app/fastShutdown=False\"."
  VERDICT=0
else
  echo "RESULT: FAIL — exit code $STATUS with NO training iteration in the log."
  echo
  if [[ "$STATUS" -eq 0 ]]; then
    echo "  => Exited 0 without training: the same false-complete signature."
    echo "     A STOCK task reproducing this means the INSTALL is broken."
  else
    echo "  => Check the log for the traceback (fastShutdown=False should have"
    echo "     surfaced a real error rather than a silent exit)."
  fi
  echo "     Log: $LOG_FILE"
  VERDICT=1
fi

echo "============================================================"
exit "$VERDICT"

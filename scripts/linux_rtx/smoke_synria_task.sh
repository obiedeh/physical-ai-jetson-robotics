#!/usr/bin/env bash
# Runtime smoke gate for the Synria pick-and-place task config.
#
# WHY THIS EXISTS: the task package was once authored blind against Isaac
# Lab stubs, and the unit suite (636 tests) can only exercise those stubs
# and the pure-math kernels. Eight runtime bugs stacked up invisibly behind
# green tests — reward sign inversions, impossible grasp thresholds, frame
# mismatches, dt-crushed milestones (see docs/handoff-2026-07-27-1937-CDT.md).
# Unit tests CANNOT catch that class of bug; three real training iterations
# can. Run this on the RTX box before landing ANY change to
# isaac/isaaclab_tasks/synria_pickplace/ or the train script.
#
# Verdict logic (exit 0 = PASS):
#   1. The run must produce real PPO iteration markers (exit 0 alone is
#      meaningless — Isaac Sim can exit 0 without training).
#   2. Mean episode length at the last logged iteration must be > 1.0 —
#      the instant-termination signature of a broken reward/termination
#      config (this exact bug shipped once).
#
# Usage:
#   bash scripts/linux_rtx/smoke_synria_task.sh
#
# Environment variable overrides:
#   ISAAC_PYTHON  — Isaac Sim python  (default: ~/.venv/isaacsim5/bin/python)
#   SMOKE_TASK    — task id           (default: Synria-Chess-PickPlace-v0)
#   SMOKE_ENVS    — parallel envs     (default: 16)
#   SMOKE_ITERS   — PPO iterations    (default: 3)
#   SMOKE_TIMEOUT — wall-clock cap    (default: 600 seconds)

set -uo pipefail   # no -e: we inspect exit codes ourselves

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"
SMOKE_TASK="${SMOKE_TASK:-Synria-Chess-PickPlace-v0}"
SMOKE_ENVS="${SMOKE_ENVS:-16}"
SMOKE_ITERS="${SMOKE_ITERS:-3}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-600}"

ISAACSIM_ROOT="${ISAACSIM_ROOT:-${HOME}/isaacsim}"
export ACCEPT_EULA=Y
export CARB_APP_PATH="$ISAACSIM_ROOT/kit"
export ISAAC_PATH="$ISAACSIM_ROOT"
export EXP_PATH="$ISAACSIM_ROOT/apps"
export LD_PRELOAD="$ISAACSIM_ROOT/kit/libcarb.so"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
[[ -f "$NVIDIA_ICD" ]] && export VK_ICD_FILENAMES="$NVIDIA_ICD"

LOG_DIR="${REPO_ROOT}/reports/diagnostics"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/smoke_synria_$(date +%Y%m%d_%H%M%S).log"

echo "== Synria task smoke gate ==================================="
echo "Task    : $SMOKE_TASK  ($SMOKE_ENVS envs, $SMOKE_ITERS iters)"
echo "Log     : $LOG_FILE"
echo "=============================================================="

timeout --signal=INT --kill-after=30 "$SMOKE_TIMEOUT" \
  "$ISAAC_PYTHON" "$REPO_ROOT/isaac/scripts/train_synria_pickplace.py" \
    --task "$SMOKE_TASK" \
    --num_envs "$SMOKE_ENVS" \
    --max_iterations "$SMOKE_ITERS" \
    --device cuda \
    --seed 42 \
    --headless \
    --kit_args="--/app/fastShutdown=False" \
  >"$LOG_FILE" 2>&1
STATUS=$?

ITER_HITS="$(grep -cE "Learning iteration" "$LOG_FILE" || true)"
LAST_EP_LEN="$(grep -E "Mean episode length" "$LOG_FILE" | tail -1 | awk '{print $NF}')"

echo
echo "== Verdict ==================================================="
echo "exit=$STATUS  iterations=$ITER_HITS  last_mean_episode_len=${LAST_EP_LEN:-n/a}"

if [[ "$ITER_HITS" -lt "$SMOKE_ITERS" ]]; then
  echo "RESULT: FAIL — fewer than $SMOKE_ITERS PPO iterations in the log."
  echo "        Check $LOG_FILE for the traceback."
  exit 1
fi

# Episode length 1.0 = every episode dies on its first step: the
# instant-termination signature (broken termination term or reward config).
if [[ -n "${LAST_EP_LEN:-}" ]] && awk "BEGIN{exit !($LAST_EP_LEN <= 1.0)}"; then
  echo "RESULT: FAIL — mean episode length ${LAST_EP_LEN} (instant termination)."
  exit 1
fi

echo "RESULT: PASS — task trains ($ITER_HITS iterations, ep len ${LAST_EP_LEN:-n/a})."
# NOTE: a teardown segfault AFTER training completes is a known Isaac Sim
# wart; the verdict above deliberately ignores the process exit code.
exit 0

#!/usr/bin/env bash
# E29 (2026-08-06): does the grasp_height reward raise upright grasps?
# Pre-registered in docs/SYNRIA_EXPERIMENT_LEDGER.md before launch.
#
# Three phases, strictly sequential — never concurrent. Two 4096-env jobs
# on one GPU halve each other's throughput AND break comparability between
# arms, which is how the first grasp-fix attempt had to be thrown away.
#
#   1. Re-evaluate the 3 CONTROL checkpoints (v5_exploitfix) with the
#      current harness, so control and treatment are measured by identical
#      code including grasp_upright and the tightened align.
#   2. Train 3 seeds fresh with the z-term.
#   3. Evaluate the 3 new checkpoints.
set -u

PY="$HOME/.venv/isaacsim5/bin/python"
REPO="$HOME/github/physical-ai-jetson-robotics"
LOG="$REPO/reports/logs/2026-08-06"
mkdir -p "$LOG"
cd "$REPO" || exit 1

echo "[e29] start $(date -Is)"

echo "[e29] PHASE 1: re-evaluating control checkpoints"
for s in 42 43 44; do
  CK="reports/training/synria_chess_pickplace_v5_exploitfix_s$s/model_final.pt"
  [ -f "$CK" ] || { echo "[e29] MISSING control $CK"; continue; }
  "$PY" isaac/scripts/eval_synria_sequence.py --headless \
      --checkpoint "$CK" --num_envs 256 --steps 1700 --seed 123 \
      --out "reports/eval/e29_control_s$s.json" \
      > "$LOG/e29_control_eval_s$s.log" 2>&1
  echo "[e29] control s$s eval exited $? at $(date -Is)"
done

echo "[e29] PHASE 2: training 3 seeds with the grasp_height reward"
for s in 42 43 44; do
  "$PY" isaac/scripts/train_synria_pickplace.py \
      --task Synria-Chess-PickPlace-v0 \
      --num_envs 4096 --max_iterations 5000 --seed "$s" \
      --experiment "synria_chess_pickplace_v7_zterm_s$s" --headless \
      > "$LOG/e29_train_s$s.log" 2>&1
  echo "[e29] train s$s exited $? at $(date -Is)"
done

echo "[e29] PHASE 3: evaluating the trained checkpoints"
for s in 42 43 44; do
  CK="reports/training/synria_chess_pickplace_v7_zterm_s$s/model_final.pt"
  [ -f "$CK" ] || { echo "[e29] MISSING treatment $CK"; continue; }
  "$PY" isaac/scripts/eval_synria_sequence.py --headless \
      --checkpoint "$CK" --num_envs 256 --steps 1700 --seed 123 \
      --out "reports/eval/e29_zterm_s$s.json" \
      > "$LOG/e29_zterm_eval_s$s.log" 2>&1
  echo "[e29] zterm s$s eval exited $? at $(date -Is)"
done

# The training script rewrites reports/training/synria_reach_policy.json on
# every run, including short ones; it is a tracked canonical pointer and a
# probe run already clobbered it once. Flag it rather than silently leave it.
echo "[e29] NOTE: synria_reach_policy.json now points at the last run here;"
echo "[e29]       check whether that is intended before committing it."
echo "[e29] ALLDONE $(date -Is)"

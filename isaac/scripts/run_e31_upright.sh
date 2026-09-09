#!/usr/bin/env bash
# E31 (2026-08-06): does upright-on-descent reduce tipping?
# Pre-registered in docs/SYNRIA_EXPERIMENT_LEDGER.md before launch.
#
# Strictly sequential — never two GPU jobs at once.
#   1. Descent-probe the 3 CONTROL checkpoints, so the primary mechanism
#      metric exists for both arms (E29 could not: the probe post-dates it).
#   2. Train 3 seeds fresh with upright_maintenance.
#   3. Eval + descent-probe the 3 new checkpoints.
#
# Control STAGE evals (e29_control_s*.json) are reused: same harness build,
# unchanged since, so re-running them would only burn GPU time.
set -u

PY="$HOME/.venv/isaacsim5/bin/python"
REPO="$HOME/github/physical-ai-jetson-robotics"
LOG="$REPO/reports/logs/2026-08-06"
mkdir -p "$LOG"
cd "$REPO" || exit 1

echo "[e31] start $(date -Is)"

echo "[e31] PHASE 1: descent-probing the control checkpoints"
for s in 42 43 44; do
  CK="reports/training/synria_chess_pickplace_v5_exploitfix_s$s/model_final.pt"
  [ -f "$CK" ] || { echo "[e31] MISSING control $CK"; continue; }
  "$PY" isaac/scripts/probe_descent_tilt.py --headless \
      --checkpoint "$CK" --num_envs 256 --steps 1700 --seed 123 \
      --out "reports/eval/e31_probe_control_s$s.json" \
      > "$LOG/e31_probe_control_s$s.log" 2>&1
  echo "[e31] control probe s$s exited $? at $(date -Is)"
done

echo "[e31] PHASE 2: training 3 seeds with upright_maintenance"
for s in 42 43 44; do
  "$PY" isaac/scripts/train_synria_pickplace.py \
      --task Synria-Chess-PickPlace-v0 \
      --num_envs 4096 --max_iterations 5000 --seed "$s" \
      --experiment "synria_chess_pickplace_v8_upright_s$s" --headless \
      > "$LOG/e31_train_s$s.log" 2>&1
  echo "[e31] train s$s exited $? at $(date -Is)"
done

echo "[e31] PHASE 3: eval + descent-probe the trained checkpoints"
for s in 42 43 44; do
  CK="reports/training/synria_chess_pickplace_v8_upright_s$s/model_final.pt"
  [ -f "$CK" ] || { echo "[e31] MISSING treatment $CK"; continue; }
  "$PY" isaac/scripts/eval_synria_sequence.py --headless \
      --checkpoint "$CK" --num_envs 256 --steps 1700 --seed 123 \
      --out "reports/eval/e31_upright_s$s.json" \
      > "$LOG/e31_eval_s$s.log" 2>&1
  echo "[e31] eval s$s exited $? at $(date -Is)"
  "$PY" isaac/scripts/probe_descent_tilt.py --headless \
      --checkpoint "$CK" --num_envs 256 --steps 1700 --seed 123 \
      --out "reports/eval/e31_probe_upright_s$s.json" \
      > "$LOG/e31_probe_upright_s$s.log" 2>&1
  echo "[e31] probe s$s exited $? at $(date -Is)"
done

# Tracked canonical pointer that the trainer rewrites on every run.
echo "[e31] NOTE: check reports/training/synria_reach_policy.json before committing."
echo "[e31] ALLDONE $(date -Is)"

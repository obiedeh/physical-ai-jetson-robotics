#!/usr/bin/env bash
# E26 — final RL experiment: does training scale still pay past 15k?
#
# Resumes the best checkpoint on record (E23, eval grasp 0.706 / full 0.126)
# three times with different TRAINING seeds, 10k iterations each. 10k, not
# less: that is the extension size that produced the E22 -> E23 gain, and a
# shorter budget reintroduces the "too young to tell" ambiguity that made the
# first seed sweep (3000 iters) uninterpretable.
#
# The decision rule is PRE-REGISTERED in docs/SYNRIA_EXPERIMENT_LEDGER.md
# (entry E26) before this script was first run. The aggregator at the end
# prints the verdict against E23's committed eval report; the rule is on the
# 3-run mean vs the 3-run spread, primary stage grasp.
#
# Runs are independent: a crash in one does not stop the rest.
set -u

PY="$HOME/.venv/isaacsim5/bin/python"
REPO="$HOME/github/physical-ai-jetson-robotics"
LOGDIR="$REPO/reports/logs/2026-08-05-e26"
RESUME_CKPT="$REPO/reports/training/synria_chess_pickplace_v3_e23/model_final.pt"
BASELINE_REPORT="$REPO/reports/eval/synria_chess_pickplace_v3_e23.json"
ITERS=10000
ENVS=4096
SEEDS=(42 43 44)

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

for f in "$RESUME_CKPT" "$BASELINE_REPORT"; do
  [ -f "$f" ] || { echo "[e26] FATAL: missing $f"; exit 1; }
done

echo "[e26] HEAD $(git rev-parse --short HEAD) | exploit fix 8c3ed7f is $(git merge-base --is-ancestor 8c3ed7f HEAD && echo PRESENT || echo MISSING)"
echo "[e26] resume: $RESUME_CKPT"
echo "[e26] budget ${ITERS} iters x ${ENVS} envs, seeds ${SEEDS[*]}; started $(date -Is)"

for SEED in "${SEEDS[@]}"; do
  NAME="synria_chess_pickplace_v6_e26_s${SEED}"
  CKPT="$REPO/reports/training/$NAME/model_final.pt"

  echo "[e26] === seed $SEED training, $(date -Is) ==="
  "$PY" isaac/scripts/train_synria_pickplace.py \
      --task Synria-Chess-PickPlace-v0 \
      --num_envs "$ENVS" \
      --max_iterations "$ITERS" \
      --seed "$SEED" \
      --experiment "$NAME" \
      --resume --checkpoint "$RESUME_CKPT" \
      --headless \
      > "$LOGDIR/train_s${SEED}.log" 2>&1
  echo "[e26] seed $SEED training exited $? at $(date -Is)"

  if [ -f "$CKPT" ]; then
    echo "[e26] === seed $SEED eval, $(date -Is) ==="
    PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_synria_sequence.py \
        --headless --checkpoint "$CKPT" \
        > "$LOGDIR/eval_s${SEED}.log" 2>&1
    echo "[e26] seed $SEED eval exited $? at $(date -Is)"
    grep -E "^\[eval\] (completed|stage)" "$LOGDIR/eval_s${SEED}.log" || \
      echo "[e26] WARNING: seed $SEED produced no eval table"
  else
    echo "[e26] WARNING: seed $SEED left no checkpoint at $CKPT; eval skipped"
  fi
done

echo "[e26] === cross-seed verdict vs E23 ==="
"$PY" isaac/scripts/aggregate_evals.py \
    reports/eval/synria_chess_pickplace_v6_e26_s*.json \
    --baseline "$BASELINE_REPORT" 2>&1 || \
  echo "[e26] WARNING: aggregation failed"

echo "[e26] ALL COMPLETE at $(date -Is)"

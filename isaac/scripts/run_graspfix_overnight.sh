#!/usr/bin/env bash
# Overnight experiment (2026-08-04/05): does the corrected 30 mm grasp centre
# make genuine pick-place cycles emerge?
#
# Runs seeds 42, 43 and 44 at an identical budget so the result rests on
# three independent seeds rather than one -- a single seed clearing a
# threshold is an anecdote, three is a finding.
#
# IMPORTANT: this waits for a nominated PID (the v3_e23 resume run) to exit
# first. An earlier attempt ran seed 42 CONCURRENTLY with that job on the
# same GPU; both halved in speed, and worse, seed 42 would have executed
# under GPU contention while seeds 43/44 ran clean -- three seeds under
# different conditions are not the controlled comparison the experiment
# needs. That attempt was aborted at iteration 316 and is re-run here from
# scratch under identical conditions.
#
# Each run is independent: a crash in one does not stop the others.
set -u

PY="$HOME/.venv/isaacsim5/bin/python"
REPO="$HOME/github/physical-ai-jetson-robotics"
LOGDIR="$REPO/reports/logs/2026-08-05"
WAIT_PID="${1:-}"

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

if [ -n "$WAIT_PID" ]; then
  echo "[overnight] waiting for pid $WAIT_PID (v3_e23) to finish; started $(date -Is)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[overnight] pid $WAIT_PID finished at $(date -Is); GPU is now free"
  sleep 30   # let VRAM actually release before claiming 4096 envs
fi

for SEED in 42 43 44; do
  NAME="synria_chess_pickplace_v4_graspfix_s${SEED}"
  echo "[overnight] launching seed $SEED at $(date -Is)"
  "$PY" isaac/scripts/train_synria_pickplace.py \
      --task Synria-Chess-PickPlace-v0 \
      --num_envs 4096 \
      --max_iterations 3000 \
      --seed "$SEED" \
      --experiment "$NAME" \
      --headless \
      > "$LOGDIR/train_v4_graspfix_s${SEED}.log" 2>&1
  echo "[overnight] seed $SEED exited $? at $(date -Is)"
done

echo "[overnight] ALL RUNS COMPLETE at $(date -Is)"

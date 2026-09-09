#!/usr/bin/env bash
# Self-healing launcher for the integrated Ludo physical-rolls game run.
# Retries on Isaac crash (carb mutex / physx re-entrancy), clearing stale shm.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-reports/ludo_live_02}"
ROLLS="${ROLLS:-3}"; CMDS="${CMDS:-6}"; SEED="${SEED:-0}"; MAX="${MAX:-3}"
cd "$REPO"
for i in $(seq 1 "$MAX"); do
  rm -f /dev/shm/sem.carbonite-sharedmemory /dev/shm/carb* 2>/dev/null
  echo "[wrap] attempt $i/$MAX $(date -Is)"
  SYNRIA_FRAME_STRIDE=2 SYNRIA_CAMERA_RES=640 PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" \
    ~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py --headless \
    --endgame --physical-rolls "$ROLLS" --commands "$CMDS" --seed "$SEED" --out "$OUT"
  rc=$?
  echo "[wrap] attempt $i exited rc=$rc $(date -Is)"
  if grep -q 'game session done' "$OUT/run.log" 2>/dev/null; then echo "[wrap] SUCCESS"; exit 0; fi
  sleep 5
done
echo "[wrap] FAILED after $MAX attempts"; exit 1

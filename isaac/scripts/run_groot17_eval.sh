#!/usr/bin/env bash
# GR00T-1.7 (ludo_groot17_v1, LoRA on ludo_corpus01) closed-loop eval chain:
#   1. serve_groot17.py (lerobot17 venv) on :5591
#   2. ludo_turn_executor.py --policy-port 5591 : N random square-pair
#      episodes, scored by the SAME rule as the scripted expert (97.2%
#      turn-level, seeds 10-12 post far-place fix)
#   3. ludo_stats.py over the eval session
#   4. corpus batch-2 top-up (ludo_corpus02 stopped at 23/150 when the GPU
#      was handed to training) into a FRESH dir — corpus mode restarts its
#      episode index at 0 and appends to raw_manifest, so topping up in
#      place would clobber episode_000000.. and duplicate manifest rows.
# Preflight: CUDA must work (the 2026-08-20 10:24 aptdaemon run left
# libnvidia-compute-580 at 580.173.02 under a 580.178.04 kernel module).
# Log: reports/logs/<date>/groot17_eval_chain.log  Markers: reports/groot17_eval.{DONE,FAIL}
set -u
cd ~/github/physical-ai-jetson-robotics
EPISODES=${EPISODES:-20}
SEED=${SEED:-300}
TOPUP=${TOPUP:-127}
OUT=${OUT:-reports/ludo_groot17_eval01}
CKPT=reports/training/ludo_groot17_v1/checkpoints/last/pretrained_model
LOG=reports/logs/$(date +%F); mkdir -p "$LOG"
rm -f reports/groot17_eval.DONE reports/groot17_eval.FAIL
exec >> "$LOG/groot17_eval_chain.log" 2>&1
echo "=== groot17 eval chain start $(date -Is) episodes=$EPISODES seed=$SEED out=$OUT"

if ! ~/.venv/isaacsim5/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "PREFLIGHT FAIL: torch.cuda unavailable (driver userland/kernel mismatch?)"
  touch reports/groot17_eval.FAIL; exit 2
fi

# 1. server (run from OUTSIDE the repo: the repo-local lerobot/ shadows the lib)
( cd /tmp && env -u HF_TOKEN \
  LD_LIBRARY_PATH=$HOME/.venv/lerobot17/lib/python3.12/site-packages/nvidia/cu13/lib \
  $HOME/.venv/lerobot17/bin/python \
  $HOME/github/physical-ai-jetson-robotics/isaac/scripts/serve_groot17.py \
  --checkpoint $HOME/github/physical-ai-jetson-robotics/$CKPT --port 5591 \
  > $HOME/github/physical-ai-jetson-robotics/$LOG/serve17.log 2>&1 ) &
SERVER_PID=$!
for i in $(seq 1 180); do
  grep -q "listening on :5591" "$LOG/serve17.log" 2>/dev/null && break
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died:"; tail -20 "$LOG/serve17.log"; touch reports/groot17_eval.FAIL; exit 3; fi
  sleep 2
done
grep -q "listening on :5591" "$LOG/serve17.log" || { echo "server never listened"; kill $SERVER_PID; touch reports/groot17_eval.FAIL; exit 3; }
echo "server up ($(date -Is))"

# 2. closed-loop eval
~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py --headless \
  --policy-port 5591 --commands "$EPISODES" --seed "$SEED" --out "$OUT" \
  > "$LOG/groot17_eval.log" 2>&1
RC=$?
echo "eval exited $RC ($(date -Is))"
kill $SERVER_PID 2>/dev/null; pkill -f serve_groot17.py 2>/dev/null

# 3. stats
python3 isaac/scripts/ludo_stats.py --glob "$OUT" > "$LOG/groot17_eval_stats.log" 2>&1
echo "stats exited $?"
[ -f "$OUT/session_summary.json" ] && python3 -c "
import json; s=json.load(open('$OUT/session_summary.json'))
print('first_try', s['first_try_ok'], '/', s['commands'], s['first_try_rate'])
print('funnel', s.get('policy_eval',{}).get('funnel'))"

# 4. corpus batch-2 top-up (fresh dir, fresh seed)
if [ "$TOPUP" -gt 0 ]; then
  echo "corpus top-up: $TOPUP episodes -> reports/ludo_corpus02b ($(date -Is))"
  ~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py --headless \
    --corpus "$TOPUP" --seed 201 --out reports/ludo_corpus02b \
    > "$LOG/ludo_corpus02b.log" 2>&1
  echo "corpus top-up exited $? ($(date -Is))"
fi

echo "=== chain end $(date -Is)"
if [ $RC -eq 0 ]; then touch reports/groot17_eval.DONE; else touch reports/groot17_eval.FAIL; fi

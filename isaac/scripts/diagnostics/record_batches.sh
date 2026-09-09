#!/bin/bash
# Fresh-boot recording batches until 200 randomized-spawn episodes.
# Fresh boots dodge the physical-degradation stall (arms wedge after
# ~100k steps of failed-close ground contact); timeout handles the
# post-RESULT teardown hang.
RAW=reports/training/franka_cup_lerobot_raw
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
while true; do
  N=$(ls "$RAW" 2>/dev/null | grep -c episode_)
  echo "[driver] $N episodes at $(date '+%H:%M:%S')"
  if [ "$N" -ge 200 ]; then
    echo "[driver] target reached: $N episodes"
    break
  fi
  timeout 7200 "$HOME/.venv/isaacsim5/bin/python" isaac/scripts/franka_vision_pick.py \
    --headless --num_envs 8 --steps 90000 \
    --record "$RAW" --max_episodes 200
  echo "[driver] batch exit code $?"
  pkill -9 -f franka_vision_pick
  sleep 15
done
echo "[driver] DONE"

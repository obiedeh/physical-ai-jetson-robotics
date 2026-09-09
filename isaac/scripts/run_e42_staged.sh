#!/usr/bin/env bash
# E41 staged evals: best screen checkpoints under the demo distribution
# (80% zone-surface starts) — like-for-like with the corpus before any
# cross-arm comparison. Waits for the screen's DONE marker first.
set -u
R=~/github/physical-ai-jetson-robotics
LOG=$R/reports/logs/2026-08-06/e42_staged.log
PY="$HOME/.venv/isaacsim5/bin/python"
THOR=oedeh@192.168.1.170
CKROOT=/home/oedeh/models/gr00t_synria/gr00t_synria_thor_v2_visonly/gr00t_synria_thor_v2_visonly
rm -f $R/reports/e42_staged.DONE
exec >> "$LOG" 2>&1
echo "=== staged wait for screen $(date -Is)"
for i in $(seq 1 240); do [ -f $R/reports/e42_screen.DONE ] && break; sleep 15; done
echo "=== staged start $(date -Is)"
for N in 8000 10000; do
  echo "--- staged ckpt $N: starting server $(date -Is)"
  ssh -o BatchMode=yes -o ConnectTimeout=15 $THOR "
    cd ~/github/Isaac-GR00T && source scripts/activate_thor.sh > /dev/null 2>&1
    setsid nohup ~/.venv/gr00t/bin/python gr00t/eval/run_gr00t_server.py \
      --model-path $CKROOT/checkpoint-$N \
      --modality-config-path /home/oedeh/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
      --port 5594 > ~/gr00t_serve_e41.log 2>&1 < /dev/null &
    echo \$! > ~/gr00t_serve_e41.pid"
  UP=0
  for i in $(seq 1 40); do
    sleep 10
    ssh -o BatchMode=yes -o ConnectTimeout=10 $THOR 'ss -tln 2>/dev/null | grep -q 5594' && { UP=1; break; }
  done
  [ $UP -ne 1 ] && { echo "server failed ckpt $N"; continue; }
  ( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_sequence.py \
      --host 192.168.1.170 --port 5594 --num_envs 16 --steps 3600 --staged --headless \
      > $R/reports/logs/2026-08-06/e42_staged_ckpt$N.log 2>&1 )
  grep -E "grasp:|carry5s:|dock:|cycle:" $R/reports/logs/2026-08-06/e42_staged_ckpt$N.log | tail -4
  ssh -o BatchMode=yes $THOR 'kill $(cat ~/gr00t_serve_e41.pid) 2>/dev/null; true'
  echo "--- staged ckpt $N done $(date -Is)"
  sleep 15
done
echo "=== staged complete $(date -Is)"
touch $R/reports/e42_staged.DONE

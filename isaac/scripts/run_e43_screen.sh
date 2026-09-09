#!/usr/bin/env bash
# E41 checkpoint screen: serve each GR00T-full-Thor checkpoint on Thor,
# eval with the funnel harness (16 envs, scratch starts) in Isaac Sim on
# the 5090. Spread-first order; refine around the peak afterwards.
# Master log: reports/logs/2026-08-06/e43_screen.log; DONE/FAIL markers.
set -u
R=~/github/physical-ai-jetson-robotics
LOG=$R/reports/logs/2026-08-06/e43_screen.log
PY="$HOME/.venv/isaacsim5/bin/python"
THOR=oedeh@192.168.1.170
CKROOT='~/models/gr00t_synria/gr00t_synria_thor_v3_visonly_cont/gr00t_synria_thor_v3_visonly_cont'
rm -f $R/reports/e43_screen.DONE $R/reports/e43_screen.FAIL
exec >> "$LOG" 2>&1
echo "=== E41 screen start $(date -Is)"

CKPTS="2000 4000 6000 8000 10000 1000 500"
for N in $CKPTS; do
  echo "--- ckpt $N: starting server $(date -Is)"
  ssh -o BatchMode=yes -o ConnectTimeout=15 $THOR "
    cd ~/github/Isaac-GR00T && source scripts/activate_thor.sh > /dev/null 2>&1
    setsid nohup ~/.venv/gr00t/bin/python gr00t/eval/run_gr00t_server.py \
      --model-path $CKROOT/checkpoint-$N \
      --modality-config-path ~/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
      --port 5594 > ~/gr00t_serve_e41.log 2>&1 < /dev/null &
    echo \$! > ~/gr00t_serve_e41.pid" || { echo "ssh start failed ckpt $N"; continue; }
  UP=0
  for i in $(seq 1 40); do
    sleep 10
    ssh -o BatchMode=yes -o ConnectTimeout=10 $THOR 'ss -tln 2>/dev/null | grep -q 5594' && { UP=1; break; }
  done
  if [ $UP -ne 1 ]; then
    echo "server never opened :5594 for ckpt $N; tail:"
    ssh -o BatchMode=yes $THOR 'tail -5 ~/gr00t_serve_e41.log' || true
    ssh -o BatchMode=yes $THOR 'kill $(cat ~/gr00t_serve_e41.pid) 2>/dev/null' || true
    continue
  fi
  echo "--- ckpt $N: server up, eval starting $(date -Is)"
  ( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_sequence.py \
      --host 192.168.1.170 --port 5594 --num_envs 16 --steps 3600 --headless \
      > $R/reports/logs/2026-08-06/e43_eval_ckpt$N.log 2>&1 )
  RC=$?
  tail -6 $R/reports/logs/2026-08-06/e43_eval_ckpt$N.log | grep -iE "grasp|carry|dock|cycle|total" || echo "ckpt $N eval rc=$RC (no summary line)"
  ssh -o BatchMode=yes $THOR 'kill $(cat ~/gr00t_serve_e41.pid) 2>/dev/null; sleep 2; kill -9 $(cat ~/gr00t_serve_e41.pid) 2>/dev/null; true'
  echo "--- ckpt $N done rc=$RC $(date -Is)"
  sleep 15
done
echo "=== E41 screen complete $(date -Is)"
touch $R/reports/e43_screen.DONE

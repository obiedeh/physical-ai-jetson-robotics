#!/usr/bin/env bash
# E45 n>=150 confirmations: ckpt36k scratch (dock-rate estimate) and
# ckpt40k staged (table confirmation). 8500 steps x 16 envs ~ 151 ep-equiv.
set -u
R=~/github/physical-ai-jetson-robotics
LOG=$R/reports/logs/2026-08-06
PY="$HOME/.venv/isaacsim5/bin/python"
THOR=oedeh@192.168.1.170
CKROOT=/home/oedeh/models/gr00t_synria/gr00t_synria_thor_v5_vis40k/gr00t_synria_thor_v5_vis40k
rm -f $R/reports/e45_n150.DONE
exec >> $LOG/e45_n150.log 2>&1
echo "=== n150 start $(date -Is)"
run_one() {  # ckpt, staged_flag, tag
  ssh -o BatchMode=yes -o ConnectTimeout=15 $THOR "
    cd ~/github/Isaac-GR00T && source scripts/activate_thor.sh > /dev/null 2>&1
    setsid nohup ~/.venv/gr00t/bin/python gr00t/eval/run_gr00t_server.py \
      --model-path $CKROOT/checkpoint-$1 \
      --modality-config-path /home/oedeh/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
      --port 5594 > ~/gr00t_serve_e45n.log 2>&1 < /dev/null &
    echo \$! > ~/gr00t_serve_e45n.pid"
  UP=0
  for i in $(seq 1 40); do sleep 10
    ssh -o BatchMode=yes -o ConnectTimeout=10 $THOR 'ss -tln 2>/dev/null | grep -q 5594' && { UP=1; break; }
  done
  [ $UP -ne 1 ] && { echo "server failed $3"; return 1; }
  ( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_sequence.py \
      --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 $2 --headless \
      > $LOG/e45_n150_$3.log 2>&1 )
  grep -E "grasp:|carry5s:|dock:|cycle:|episode-equiv" $LOG/e45_n150_$3.log | tail -5
  ssh -o BatchMode=yes $THOR 'kill $(cat ~/gr00t_serve_e45n.pid) 2>/dev/null; true'
  sleep 15
}
run_one 36000 "" ck36_scratch
run_one 40000 "--staged" ck40_staged
echo "=== n150 complete $(date -Is)"
touch $R/reports/e45_n150.DONE

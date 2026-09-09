#!/usr/bin/env bash
# Overnight endgame: E48 ckpt-35k at n>=150 (scratch + staged), then the
# E49 system rows S1 and S2 (staged n>=150). The staged bare run IS S0.
set -u
R=~/github/physical-ai-jetson-robotics
LOG=$R/reports/logs/2026-08-06
PY="$HOME/.venv/isaacsim5/bin/python"
THOR=oedeh@192.168.1.170
CK=/home/oedeh/models/gr00t_synria/gr00t_synria_thor_v6_merged40k/gr00t_synria_thor_v6_merged40k/checkpoint-35000
rm -f $R/reports/e48_endgame.DONE
exec >> $LOG/e48_endgame.log 2>&1
echo "=== endgame start $(date -Is)"

start_server() {
  ssh -o ConnectTimeout=20 $THOR "
    cd ~/github/Isaac-GR00T && source scripts/activate_thor.sh > /dev/null 2>&1
    setsid nohup ~/.venv/gr00t/bin/python gr00t/eval/run_gr00t_server.py \
      --model-path $CK \
      --modality-config-path /home/oedeh/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
      --port 5594 > ~/gr00t_serve_endgame.log 2>&1 < /dev/null &
    echo \$! > ~/gr00t_serve_endgame.pid"
  for i in $(seq 1 40); do sleep 10
    ssh -o ConnectTimeout=15 $THOR 'ss -tln 2>/dev/null | grep -q 5594' && return 0
  done
  return 1
}
stop_server() {
  ssh -o ConnectTimeout=20 $THOR 'kill $(cat ~/gr00t_serve_endgame.pid) 2>/dev/null; sleep 2; kill -9 $(cat ~/gr00t_serve_endgame.pid) 2>/dev/null; true'
  sleep 10
}

start_server || { echo "server failed"; exit 1; }
echo "--- n150 scratch $(date -Is)"
( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_sequence.py \
    --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 --headless \
    > $LOG/e48_n150_scratch.log 2>&1 )
grep -E "grasp:|carry5s:|dock:|cycle:" $LOG/e48_n150_scratch.log | tail -4

echo "--- n150 staged (= S0) $(date -Is)"
( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_sequence.py \
    --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 --staged --headless \
    > $LOG/e48_n150_staged_S0.log 2>&1 )
grep -E "grasp:|carry5s:|dock:|cycle:" $LOG/e48_n150_staged_S0.log | tail -4

echo "--- S1 supervisor $(date -Is)"
( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_system.py \
    --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 --staged \
    --supervisor --headless > $LOG/e49_S1.log 2>&1 )
grep -E "grasp:|carry5s:|dock:|cycle:|retries" $LOG/e49_S1.log | tail -6

echo "--- S2 supervisor+gemini $(date -Is)"
( cd $R && PYTHONUNBUFFERED=1 "$PY" -u isaac/scripts/eval_gr00t_system.py \
    --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 --staged \
    --supervisor --gemini --headless > $LOG/e49_S2.log 2>&1 )
grep -E "grasp:|carry5s:|dock:|cycle:|retries|gates" $LOG/e49_S2.log | tail -7

stop_server
echo "=== endgame complete $(date -Is)"
touch $R/reports/e48_endgame.DONE

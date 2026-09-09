#!/usr/bin/env bash
# Progress-based watchdog (2026-08-10). Born from two multi-hour losses:
# a training process that HUNG AT TEARDOWN while pgrep called it alive, and
# an eval chain whose pgrep -f poll matched its own ssh shell forever.
# Lesson encoded here: NEVER infer health from process existence — measure
# PROGRESS (log growth) and completion MARKERS, every cycle.
#
# Emits ONE line per cycle:  OK ...   or   STALL/DIED ... (actionable)
set -u
cd ~/github/physical-ai-jetson-robotics
STATE=/tmp/claude-1000/watchdog_state
mkdir -p "$STATE"
LOGDIR=reports/logs/2026-08-06
DONE_RE='DONE|ALLDONE|EVALSDONE|exited [0-9]|End of training|======== RESULT|target reached'

while true; do
  ISSUES=""; OKBITS=""
  # --- 5090: any log touched in the last 3h is a tracked job -------------
  for f in $(find $LOGDIR -name "*.log" ! -name "*chain*" ! -name "*serve*" -mmin -180 2>/dev/null); do
    b=$(basename "$f" .log)
    sz=$(stat -c %s "$f"); prev=$(cat "$STATE/$b" 2>/dev/null || echo 0)
    echo "$sz" > "$STATE/$b"
    age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
    if [ "$age" -lt 32 ] && [ "$sz" -gt "$prev" ]; then
      OKBITS="$OKBITS ${b}:+$(( (sz - prev) / 1024 ))K"
    elif tail -c 4000 "$f" | grep -qE "$DONE_RE"; then
      : # finished cleanly — not a stall
    elif [ "$age" -ge 32 ] && fuser "$f" >/dev/null 2>&1; then
      # quiet AND a process still holds the log open = genuine stall;
      # quiet with no writer = the stage simply finished (its exit status
      # lives in the chain log, not here)
      ISSUES="$ISSUES STALL:${b}(quiet ${age}m, writer attached)"
    fi
  done
  # --- 5090 GPU cross-check ----------------------------------------------
  GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  # --- Thor: training log + policy server --------------------------------
  THOR=$(ssh -o BatchMode=yes -o ConnectTimeout=10 oedeh@192.168.1.170 '
    for f in /home/oedeh/gr00t_e*.log; do
      [ -f "$f" ] || continue
      b="${f%.log}"
      # finished or failed jobs are not stall candidates
      [ -f "$b.DONE" ] && continue
      [ -f "$b.FAIL" ] && continue
      age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
      if [ "$age" -lt 32 ]; then printf "%s:active " "$(basename $f .log)";
      elif tail -c 3000 "$f" | grep -qE "End of training|listening"; then :;
      else printf "STALL:%s(quiet %sm) " "$(basename $f .log)" "$age"; fi
    done
    ss -tln 2>/dev/null | grep -q 5599 && printf "serve:UP" || printf "serve:down"
  ' 2>/dev/null) || {
    sleep 30   # debounce: one transient ssh timeout is not an outage
    THOR=$(ssh -o BatchMode=yes -o ConnectTimeout=10 oedeh@192.168.1.170 'echo retry-ok; ss -tln 2>/dev/null | grep -q 5599 && printf "serve:UP" || printf "serve:down"' 2>/dev/null) || THOR="THOR-UNREACHABLE"
  }
  case "$THOR" in *STALL*|*UNREACHABLE*) ISSUES="$ISSUES $THOR";; esac
  # --- one line per cycle -------------------------------------------------
  if [ -n "$ISSUES" ]; then
    echo "$(date +%H:%M) PROBLEM:$ISSUES | gpu=${GPU}% | thor: $THOR"
  else
    # OK heartbeats go to stderr: logged in the monitor output file for
    # liveness verification, but never raise a notification (operator asked
    # for internal-only probing; only PROBLEM lines should wake anyone)
    echo "$(date +%H:%M) OK${OKBITS:- (all quiet, all marked done)} | gpu=${GPU}% | thor: $THOR" >&2
  fi
  sleep 1800
done

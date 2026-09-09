#!/usr/bin/env bash
# G2 scorer runner: self-restarting on quota stalls, resumable via
# g2_progress.jsonl checkpoint. Writes G2FINAL marker when the scorer
# exits 0 (full report written).
set -u
R="$HOME/github/physical-ai-jetson-robotics"
G="$HOME/github/physical-ai-jetson-robotics-gemini-er-2"
LOG="$R/reports/logs/2026-08-06/g2_score.log"
MARKER="$R/reports/perception_bench_g2/G2FINAL"
rm -f "$MARKER"
for attempt in 1 2 3 4 5 6; do
  echo "[loop] attempt $attempt $(date -Is)" >> "$LOG"
  if python3 "$G/gemini_er2_bridge/scripts/bench_g2.py" \
      --bench "$R/reports/perception_bench_g2" --limit 40 >> "$LOG" 2>&1; then
    touch "$MARKER"
    echo "[loop] G2FINAL $(date -Is)" >> "$LOG"
    exit 0
  fi
  echo "[loop] scorer exited nonzero; cooling off 15 min" >> "$LOG"
  sleep 900
done
echo "[loop] gave up after 6 attempts" >> "$LOG"
exit 1

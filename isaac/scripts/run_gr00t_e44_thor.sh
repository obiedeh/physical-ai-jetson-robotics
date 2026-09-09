#!/usr/bin/env bash
# E44: fresh visual-only, one 25k-step schedule (time-parity campaign).
# Smoke-validated config: batch 8 (~80 GB), eager attention, unfrozen
# LLM + visual backbone, 2.38 s/step -> 10k steps ~ 6.7 h.
# Log: ~/gr00t_e44.log  Markers: ~/gr00t_e44.{DONE,FAIL}
set -u
LOG=~/gr00t_e44.log
rm -f ~/gr00t_e44.DONE ~/gr00t_e44.FAIL
exec >> "$LOG" 2>&1
echo "=== E41 start $(date -Is)"
cd ~/github/Isaac-GR00T
source scripts/activate_thor.sh
export PATH="$HOME/.venv/gr00t/bin:$PATH"
CORPUS=~/github/physical-ai-jetson-robotics/reports/training/synria_e33v3_lerobot_v21
OUT=~/models/gr00t_synria/gr00t_synria_thor_v4_vis25k
mkdir -p "$OUT"
~/.venv/gr00t/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$CORPUS" \
  --embodiment-tag new_embodiment \
  --modality-config-path ~/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
  --output-dir "$OUT" \
  --experiment-name gr00t_synria_thor_v4_vis25k \
  --tune-visual \
  --global-batch-size 8 \
  --max-steps 25000 \
  --save-steps 1000 \
  --save-total-limit 6 \
  --save-only-model \
  --dataloader-num-workers 2 \
  --no-use-wandb
RC=$?
echo "=== E41 exit $RC $(date -Is)"
if [ $RC -eq 0 ]; then touch ~/gr00t_e44.DONE; else touch ~/gr00t_e44.FAIL; fi

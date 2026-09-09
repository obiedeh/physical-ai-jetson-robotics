#!/usr/bin/env bash
# E43: continue E42 visual-only arm from ckpt-10000 weights, +10k steps.
# Smoke-validated config: batch 8 (~80 GB), eager attention, unfrozen
# LLM + visual backbone, 2.38 s/step -> 10k steps ~ 6.7 h.
# Log: ~/gr00t_e43.log  Markers: ~/gr00t_e43.{DONE,FAIL}
set -u
LOG=~/gr00t_e43.log
rm -f ~/gr00t_e43.DONE ~/gr00t_e43.FAIL
exec >> "$LOG" 2>&1
echo "=== E41 start $(date -Is)"
cd ~/github/Isaac-GR00T
source scripts/activate_thor.sh
export PATH="$HOME/.venv/gr00t/bin:$PATH"
CORPUS=~/github/physical-ai-jetson-robotics/reports/training/synria_e33v3_lerobot_v21
OUT=~/models/gr00t_synria/gr00t_synria_thor_v3_visonly_cont
mkdir -p "$OUT"
~/.venv/gr00t/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/oedeh/models/gr00t_synria/gr00t_synria_thor_v2_visonly/gr00t_synria_thor_v2_visonly/checkpoint-10000 \
  --dataset-path "$CORPUS" \
  --embodiment-tag new_embodiment \
  --modality-config-path ~/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
  --output-dir "$OUT" \
  --experiment-name gr00t_synria_thor_v3_visonly_cont \
  --tune-visual \
  --global-batch-size 8 \
  --max-steps 10000 \
  --save-steps 500 \
  --save-total-limit 20 \
  --save-only-model \
  --dataloader-num-workers 2 \
  --no-use-wandb
RC=$?
echo "=== E41 exit $RC $(date -Is)"
if [ $RC -eq 0 ]; then touch ~/gr00t_e43.DONE; else touch ~/gr00t_e43.FAIL; fi

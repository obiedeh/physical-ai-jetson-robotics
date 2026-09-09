#!/usr/bin/env bash
# E41: GR00T-full-Thor fine-tune (gr00t_synria_thor_v1).
# Smoke-validated config: batch 8 (~80 GB), eager attention, unfrozen
# LLM + visual backbone, 2.38 s/step -> 10k steps ~ 6.7 h.
# Log: ~/gr00t_e41.log  Markers: ~/gr00t_e41.{DONE,FAIL}
set -u
LOG=~/gr00t_e41.log
rm -f ~/gr00t_e41.DONE ~/gr00t_e41.FAIL
exec >> "$LOG" 2>&1
echo "=== E41 start $(date -Is)"
cd ~/github/Isaac-GR00T
source scripts/activate_thor.sh
export PATH="$HOME/.venv/gr00t/bin:$PATH"
CORPUS=~/github/physical-ai-jetson-robotics/reports/training/synria_e33v3_lerobot_v21
OUT=~/models/gr00t_synria/gr00t_synria_thor_v1
mkdir -p "$OUT"
~/.venv/gr00t/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$CORPUS" \
  --embodiment-tag new_embodiment \
  --modality-config-path ~/github/physical-ai-jetson-robotics/isaac/scripts/gr00t_synria_config.py \
  --output-dir "$OUT" \
  --experiment-name gr00t_synria_thor_v1 \
  --tune-llm --tune-visual \
  --global-batch-size 8 \
  --max-steps 10000 \
  --save-steps 500 \
  --save-total-limit 20 \
  --save-only-model \
  --dataloader-num-workers 2 \
  --no-use-wandb
RC=$?
echo "=== E41 exit $RC $(date -Is)"
if [ $RC -eq 0 ]; then touch ~/gr00t_e41.DONE; else touch ~/gr00t_e41.FAIL; fi

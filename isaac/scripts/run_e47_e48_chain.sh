#!/usr/bin/env bash
# Overnight chain: when the E46 corpus DONE marker lands —
#   1. rsync the v2.1 (GR00T-format) e46 corpus to Thor
#   2. launch E48 on Thor (merged-corpus 40k schedule)
#   3. launch E47 on the 5090 (ACT on e46 corpus, baseline recipe)
# Log: reports/logs/2026-08-06/e47e48_chain.log
set -u
R=~/github/physical-ai-jetson-robotics
LOG=$R/reports/logs/2026-08-06/e47e48_chain.log
exec >> "$LOG" 2>&1
echo "=== chain armed $(date -Is)"
for i in $(seq 1 120); do
  [ -f $R/reports/e46_corpus.DONE ] && break
  sleep 300
done
[ -f $R/reports/e46_corpus.DONE ] || { echo "corpus never landed"; exit 1; }
echo "=== corpus landed $(date -Is)"

# the migrator leaves v2.1 at ${DS}_old; GR00T consumes v2.1
V21=$R/reports/training/synria_e46_lerobot_old
[ -d "$V21" ] || V21=$R/reports/training/synria_e46_lerobot   # not migrated case
echo "v21 source: $V21"
rsync -a "$V21/" oedeh@192.168.1.170:~/github/physical-ai-jetson-robotics/reports/training/synria_e46_lerobot_v21/
echo "rsync to Thor exited $? ($(date -Is))"

scp -q $R/isaac/scripts/run_gr00t_e48_thor.sh oedeh@192.168.1.170:~/gr00t_e48.sh
ssh -o BatchMode=yes oedeh@192.168.1.170 'chmod +x ~/gr00t_e48.sh; setsid nohup ~/gr00t_e48.sh > /dev/null 2>&1 < /dev/null & echo E48 launched'

echo "=== launching E47 (ACT on e46, baseline recipe) $(date -Is)"
DS=$R/reports/training/synria_e46_lerobot
cd ~ && HF_HUB_OFFLINE=1 ~/.venv/lerobot/bin/python ~/.venv/lerobot/bin/lerobot-train \
  --dataset.repo_id=synria_e33v2 \
  --dataset.root=$DS \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=$R/reports/training/synria_act_v4_e46 \
  --batch_size=8 --steps=100000 --wandb.enable=false \
  > $R/reports/logs/2026-08-06/e47_train.log 2>&1
echo "E47 train exited $? ($(date -Is))"
touch $R/reports/e47_train.DONE

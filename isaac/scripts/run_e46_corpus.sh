#!/usr/bin/env bash
# E46 corpus (dock-targeted expansion): 1520 episodes with the PROVEN corpus-v1
# hybrid expert, then the full pipeline: convert -> stats -> v3.0 migrate ->
# ACT retrain -> checkpoint-sweep eval -> final 40-episode eval.
# 20 raw episodes are held back from training as an open-loop probe set.
set -u
cd ~/github/physical-ai-jetson-robotics
LOG=reports/logs/2026-08-06
PY_ISAAC=~/.venv/isaacsim5/bin/python
PY_GROOT=~/.venv/gr00t/bin/python
PY_LEROBOT=~/.venv/lerobot/bin/python
RAW=reports/training/synria_e46_lerobot_raw
DS=reports/training/synria_e46_lerobot

echo "[e46] STAGE 1: record 320 episodes ($(date -Is))"
$PY_ISAAC isaac/scripts/record_e33_demos.py --headless --expert hybrid \
  --checkpoint reports/training/synria_chess_pickplace_v6_e26_s43/model_final.pt \
  --num_envs 16 --steps 3000000 --max_episodes 1520 --seed 46 \
  --out_root $RAW > $LOG/e46_record.log 2>&1
echo "[e46] record exited $? ($(date -Is))"

echo "[e46] STAGE 2: hold out last 20 episodes"
mkdir -p ${RAW}_holdout
for i in $(seq 1500 1519); do
  d=$(printf "episode_%06d" $i)
  [ -d "$RAW/$d" ] && mv "$RAW/$d" "${RAW}_holdout/"
done
python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("$RAW/raw_manifest.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
keep = [r for r in rows if r["episode_index"] < 1500]
p.write_text("\n".join(json.dumps(r) for r in keep) + "\n")
print("manifest trimmed to", len(keep))
PYEOF

echo "[e46] STAGE 3: convert + stats + migrate"
$PY_GROOT isaac/scripts/convert_synria_lerobot.py --raw $RAW --out $DS \
  --embodiment synria > $LOG/e46_convert.log 2>&1
echo "[e46] convert exited $?"
cd ~ && $PY_LEROBOT ~/github/physical-ai-jetson-robotics/isaac/scripts/gen_episodes_stats_v21.py \
  --root ~/github/physical-ai-jetson-robotics/$DS > ~/github/physical-ai-jetson-robotics/$LOG/e46_stats.log 2>&1
echo "[e46] stats exited $?"
$PY_LEROBOT -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id synria_e33v2 \
  --root ~/github/physical-ai-jetson-robotics/$DS --push-to-hub false \
  > ~/github/physical-ai-jetson-robotics/$LOG/e46_migrate.log 2>&1
echo "[e46] migrate exited $?"
cd ~/github/physical-ai-jetson-robotics

echo "[e46] dataset build complete ($(date -Is))"
touch reports/e46_corpus.DONE

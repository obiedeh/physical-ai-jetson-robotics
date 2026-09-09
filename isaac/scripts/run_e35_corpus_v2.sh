#!/usr/bin/env bash
# E35 corpus v2 (data-quantity lever): 320 episodes with the PROVEN corpus-v1
# hybrid expert, then the full pipeline: convert -> stats -> v3.0 migrate ->
# ACT retrain -> checkpoint-sweep eval -> final 40-episode eval.
# 20 raw episodes are held back from training as an open-loop probe set.
set -u
cd ~/github/physical-ai-jetson-robotics
LOG=reports/logs/2026-08-06
PY_ISAAC=~/.venv/isaacsim5/bin/python
PY_GROOT=~/.venv/gr00t/bin/python
PY_LEROBOT=~/.venv/lerobot/bin/python
RAW=reports/training/synria_e33v2_lerobot_raw
DS=reports/training/synria_e33v2_lerobot

echo "[v2] STAGE 1: record 320 episodes ($(date -Is))"
$PY_ISAAC isaac/scripts/record_e33_demos.py --headless --expert hybrid \
  --checkpoint reports/training/synria_chess_pickplace_v6_e26_s43/model_final.pt \
  --num_envs 16 --steps 600000 --max_episodes 320 --seed 11 \
  --out_root $RAW > $LOG/e35_record.log 2>&1
echo "[v2] record exited $? ($(date -Is))"

echo "[v2] STAGE 2: hold out last 20 episodes"
mkdir -p ${RAW}_holdout
for i in $(seq 300 319); do
  d=$(printf "episode_%06d" $i)
  [ -d "$RAW/$d" ] && mv "$RAW/$d" "${RAW}_holdout/"
done
python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("$RAW/raw_manifest.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
keep = [r for r in rows if r["episode_index"] < 300]
p.write_text("\n".join(json.dumps(r) for r in keep) + "\n")
print("manifest trimmed to", len(keep))
PYEOF

echo "[v2] STAGE 3: convert + stats + migrate"
$PY_GROOT isaac/scripts/convert_synria_lerobot.py --raw $RAW --out $DS \
  --embodiment synria > $LOG/e35_convert.log 2>&1
echo "[v2] convert exited $?"
cd ~ && $PY_LEROBOT ~/github/physical-ai-jetson-robotics/isaac/scripts/gen_episodes_stats_v21.py \
  --root ~/github/physical-ai-jetson-robotics/$DS > ~/github/physical-ai-jetson-robotics/$LOG/e35_stats.log 2>&1
echo "[v2] stats exited $?"
$PY_LEROBOT -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id synria_e33v2 \
  --root ~/github/physical-ai-jetson-robotics/$DS --push-to-hub false \
  > ~/github/physical-ai-jetson-robotics/$LOG/e35_migrate.log 2>&1
echo "[v2] migrate exited $?"
cd ~/github/physical-ai-jetson-robotics

echo "[v2] STAGE 4: train ACT v2 ($(date -Is))"
cd ~ && HF_HUB_OFFLINE=1 $PY_LEROBOT ~/.venv/lerobot/bin/lerobot-train \
  --dataset.repo_id=synria_e33v2 \
  --dataset.root=/home/oedeh/github/physical-ai-jetson-robotics/$DS \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/home/oedeh/github/physical-ai-jetson-robotics/reports/training/synria_act_v2 \
  --batch_size=8 --steps=100000 --wandb.enable=false \
  > ~/github/physical-ai-jetson-robotics/$LOG/e35_train.log 2>&1
echo "[v2] train exited $? ($(date -Is))"
cd ~/github/physical-ai-jetson-robotics

serve() {
  OLD=$(ss -tlnp 2>/dev/null | grep 5599 | grep -oP "pid=\K[0-9]+" | head -1)
  [ -n "$OLD" ] && kill "$OLD" 2>/dev/null; sleep 5
  setsid nohup env HF_HUB_OFFLINE=1 $PY_LEROBOT \
    "$PWD/isaac/scripts/serve_act_policy.py" --checkpoint "$PWD/$1" \
    --port 5599 > $LOG/e35_serve.log 2>&1 < /dev/null &
  for i in $(seq 1 24); do ss -tln | grep -q 5599 && return 0; sleep 5; done
  return 1
}

echo "[v2] STAGE 5: checkpoint sweep + final eval"
for CK in 020000 060000 100000; do
  D=reports/training/synria_act_v2/checkpoints/$CK/pretrained_model
  [ -d "$D" ] || continue
  serve "$D" || continue
  $PY_ISAAC isaac/scripts/eval_act_synria.py --headless --episodes 10 \
    --out reports/eval/act_v2_sweep_$CK.json > $LOG/e35_eval_$CK.log 2>&1
  echo "[v2] sweep $CK exited $?"
done
serve reports/training/synria_act_v2/checkpoints/last/pretrained_model && \
$PY_ISAAC isaac/scripts/eval_act_synria.py --headless --episodes 40 \
  --out reports/eval/act_v2_final.json > $LOG/e35_eval_final.log 2>&1
echo "[v2] final eval exited $?"
echo "[v2] ALLDONE ($(date -Is))"

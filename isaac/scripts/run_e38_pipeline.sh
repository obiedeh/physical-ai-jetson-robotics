#!/usr/bin/env bash
# E38: corpus v3 (1000 eps, E36 picker) -> ACT v3 on THOR, capped ~5 epochs.
# All completion checks use MARKER FILES or log markers — never pgrep (4x trap).
set -u
cd ~/github/physical-ai-jetson-robotics
LOG=reports/logs/2026-08-06
RAW=reports/training/synria_e33v3_lerobot_raw
DS=reports/training/synria_e33v3_lerobot

until grep -q E38RECDONE $LOG/e38_record.log 2>/dev/null; do
  echo "[e38] waiting on recorder $(date +%H:%M) ($(grep -cE "episode .* saved" $LOG/e38_record.log 2>/dev/null || echo 0) eps)"
  sleep 300
done
echo "[e38] recording done $(date -Is)"

echo "[e38] holdout 20"
mkdir -p ${RAW}_holdout
N=$(ls -d $RAW/episode_* | wc -l)
for i in $(seq $((N-20)) $((N-1))); do
  d=$(printf "episode_%06d" $i); [ -d "$RAW/$d" ] && mv "$RAW/$d" "${RAW}_holdout/"
done
python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("$RAW/raw_manifest.jsonl")
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
keep=[r for r in rows if r["episode_index"] < $N-20]
p.write_text("\n".join(json.dumps(r) for r in keep)+"\n"); print("kept",len(keep))
PYEOF

~/.venv/gr00t/bin/python isaac/scripts/convert_synria_lerobot.py --raw $RAW --out $DS --embodiment synria > $LOG/e38_convert.log 2>&1
echo "[e38] convert exited $?"
( cd ~ && ~/.venv/lerobot/bin/python ~/github/physical-ai-jetson-robotics/isaac/scripts/gen_episodes_stats_v21.py --root ~/github/physical-ai-jetson-robotics/$DS > ~/github/physical-ai-jetson-robotics/$LOG/e38_stats.log 2>&1 )
echo "[e38] stats exited $?"
( cd ~ && ~/.venv/lerobot/bin/python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id synria_e33v3 --root ~/github/physical-ai-jetson-robotics/$DS --push-to-hub false > ~/github/physical-ai-jetson-robotics/$LOG/e38_migrate.log 2>&1 )
echo "[e38] migrate exited $?"

FRAMES=$(python3 -c "import json;print(json.load(open('$DS/meta/info.json'))['total_frames'])")
STEPS=$(python3 -c "print(min(320000, int($FRAMES/8*5)))")   # ~5 epochs, capped
echo "[e38] frames=$FRAMES -> steps=$STEPS (~5 epochs)"

echo "[e38] rsync dataset to Thor"
rsync -a $DS oedeh@192.168.1.170:github/physical-ai-jetson-robotics/reports/training/
echo "[e38] launching training on Thor ($(date -Is))"
ssh -o BatchMode=yes oedeh@192.168.1.170 "
cd ~/github/physical-ai-jetson-robotics
rm -f /tmp/e38_train_done
setsid nohup env HF_HUB_OFFLINE=1 bash -c '~/.venv/lerobot/bin/lerobot-train \
  --dataset.repo_id=synria_e33v3 \
  --dataset.root=/home/oedeh/github/physical-ai-jetson-robotics/$DS \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/home/oedeh/github/physical-ai-jetson-robotics/reports/training/synria_act_v3 \
  --batch_size=8 --steps=$STEPS --wandb.enable=false \
  > /tmp/e38_train_thor.log 2>&1; touch /tmp/e38_train_done' > /dev/null 2>&1 < /dev/null &
echo thor-launch-ok"
until ssh -o BatchMode=yes oedeh@192.168.1.170 'test -f /tmp/e38_train_done' 2>/dev/null; do
  echo "[e38] waiting on Thor training $(date +%H:%M)"
  sleep 600
done
echo "[e38] Thor training done $(date -Is)"

echo "[e38] serve + sweep + final eval"
for CK in 100000 200000 last; do
  ssh -o BatchMode=yes oedeh@192.168.1.170 "
  OLD=\$(ss -tlnp 2>/dev/null | grep 5599 | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n \"\$OLD\" ] && kill \$OLD 2>/dev/null; sleep 5
  D=/home/oedeh/github/physical-ai-jetson-robotics/reports/training/synria_act_v3/checkpoints/$CK/pretrained_model
  [ -d \"\$D\" ] || exit 1
  cd ~/github/physical-ai-jetson-robotics
  setsid nohup env HF_HUB_OFFLINE=1 ~/.venv/lerobot/bin/python isaac/scripts/serve_act_policy.py \
    --checkpoint \"\$D\" --bind 0.0.0.0 --port 5599 > /tmp/e38_serve.log 2>&1 < /dev/null &
  sleep 60; ss -tln | grep -q 5599" 2>/dev/null || continue
  EPS=10; OUT=reports/eval/e38_sweep_$CK.json
  [ "$CK" = "last" ] && EPS=40 && OUT=reports/eval/e38_act_v3_final.json
  ~/.venv/isaacsim5/bin/python isaac/scripts/eval_act_synria.py --headless --episodes $EPS \
    --host 192.168.1.170 --out $OUT > $LOG/e38_eval_$CK.log 2>&1
  echo "[e38] eval $CK exited $?"
done
echo "E38DONE"

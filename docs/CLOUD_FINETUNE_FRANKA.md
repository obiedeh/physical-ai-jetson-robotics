# Cloud fine-tune: GR00T N1.7 unfrozen-DiT on the Franka surrogate demos

Purpose: the decisive escalation identified by the control experiment
(ledger, 2026-07-31). Local 32 GB forces `--no-tune-diffusion-model`
(NVIDIA's full fine-tune floor is 40 GB); the frozen-DiT policy learned
the pick-move-place skill but not the visual conditioning to retarget it
to a randomized cup position (v4: open-loop MAE 0.228, closed-loop 0/24;
v5 fixed-spawn control: 3 places / 32 episodes). Unfreezing the DiT
attacks the visuomotor-coupling limit directly.

## What to upload

- Dataset: `reports/training/franka_cup_lerobot_demos_v2` (200+ episodes,
  produced by the batch recorder + `convert_synria_lerobot.py
  --embodiment franka`). LeRobot v2.1: parquet + wrist/overhead mp4 +
  meta (modality.json: arm 0-7, gripper 7-8).
- Modality config: `isaac/isaaclab_tasks/synria_pickplace/gr00t/modality_config.py`
  (registers `new_embodiment`; dimension-agnostic — reused unchanged).
- Isaac-GR00T checkout (github.com/NVIDIA/Isaac-GR00T, Apache-2.0) with
  its Python 3.10-3.12 venv per the repo README.

## GPU requirements

- 40 GB+ VRAM (A100 40/80 GB, H100, L40S 48 GB). Batch 1 fits in 40 GB
  with the DiT unfrozen; larger batches (8-16) on 80 GB will train
  faster and likely better — the local runs were batch 1 only because
  of the 32 GB ceiling, not by choice.
- HF_TOKEN env var required for the `nvidia/GR00T-N1.7-3B` download.

## Launch command (unfrozen DiT — note NO --no-tune-diffusion-model)

```bash
HF_TOKEN=... python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/franka_cup_lerobot_demos_v2 \
  --embodiment-tag new_embodiment \
  --modality-config-path /path/to/modality_config.py \
  --output-dir /path/to/gr00t_franka_v7_cloud \
  --global-batch-size 16 \
  --max-steps 10000 \
  --save-steps 2500
```

Reference timings: locally (RTX 5090, frozen DiT, batch 1) 5,000 steps
took ~8 min. Unfrozen at batch 8 on an A100-80GB expect roughly
30-60 min for 10,000 steps — small either way; the cost is dominated by
setup, not training.

## Bring-back + evaluation

Download the final checkpoint directory (includes `processor/` and
`experiment_cfg/` — keep the whole `checkpoint-NNNN` folder), then:

1. Open-loop: `gr00t/eval/open_loop_eval.py --model-path <ckpt>
   --dataset-path <dataset> --embodiment-tag new_embodiment
   --steps 200 --traj-ids 0 10 25 40 55` (v4 reference: MAE 0.228;
   zeros baseline 0.60).
2. Closed-loop (the metric that matters): serve with
   `gr00t/eval/run_gr00t_server.py --model-path <ckpt>
   --embodiment-tag new_embodiment --port 5591`, then
   `isaac/scripts/eval_gr00t_franka.py --headless --enable_cameras
   --port 5591 --num_envs 4 --steps 7200` (randomized spawn;
   v4 reference: 0 picks / 24 episodes; fixed-spawn v5 control:
   4 picks, 3 places / 32).

Success criterion per the mission rules: measured closed-loop
picks/places on the randomized task — never loss curves alone.

# Artifact manifest — where everything lives

Latest resumption note: [September 7 handoff](handoff-2026-09-07.md), including
corrected results, vendor-asset prerequisites and validation limits.

Historical inventory notice, 2026-09-07: the local paths and sizes below record
the July 2026 migration, not a current inventory or completeness guarantee.
The July handoff is a dated resumption note. Current status is in the
[README](../README.md); vendor-source geometry is now locally obtained and
excluded from the current tracked tree. Evidence and metadata remain unchanged.

Nothing that matters lives in `/tmp` any more. Scratch space is wiped on
reboot, so all checkpoints, datasets and reference material were moved to
`~/models/` (same filesystem, so the move was instant and cost no extra
disk). Text evidence — logs, scripts, reports — is versioned in this
repository.

## In this repository (versioned, browsable on GitHub)

| Path | What it is |
|---|---|
| `isaac/scripts/synria_grasp_feasibility.py` | The Synria grasp **gate**. Modes: default trials, `--travel_test`, `--block_test`, `--hover_test`; switches `--free_air`, `--no_world`, `--no_self_collision`, `--raise_base`, `--usd <variant>`. Exit 0 pass / 3 travel fail / 4 block fail. |
| `isaac/scripts/evaluate_stage_pipeline.py` | Closed-loop stage evaluator: per-episode JSONL, failure taxonomy, skill isolation (`--takeover_stage`), diagnostic shims. |
| `isaac/scripts/analyze_failures.py` | Failure distributions + Wilson 95% CIs → `reports/stage_metrics.csv`. |
| `isaac/scripts/eval_gr00t_franka.py` | Fixed closed-loop harness for the Franka surrogate. |
| `isaac/scripts/franka_vision_pick.py` | Scripted vision expert / demo recorder (`--record`, `--dwell`, `--fixed_spawn`). |
| `isaac/scripts/probe_gripper_runtime.py` | Queries the **spawned** articulation for live collider state — the probe that overturned two wrong conclusions. |
| `isaac/scripts/diagnostics/` | 17 one-shot investigation scripts (USD composition, collider inventory, finger limits, pad-variant generators, corpus merge, MuJoCo cross-check, batch driver). |
| `isaac/isaaclab_tasks/synria_pickplace/franka_cup_env.py` | The Franka + 2F-85 surrogate environment. |
| `isaac/usd/robots/synria_6dof_arm_v2/*_{nopad,padfix,padfix_l,padfix2,repaired}.usd` | Gripper repair override layers (vendor asset untouched). |
| `reports/logs/2026-07-31/` | 212 raw run logs — every training, evaluation, probe and trial in this session. |
| `reports/*.jsonl`, `reports/synria_*.json` | Per-episode ledgers and trial results. |
| `reports/training/franka_cup_lerobot_demos_v2` | The 200-episode / 63,340-frame corpus used for v6c–v7. |
| `reports/synria_grasp_audit.md` | Grasp audit **plus both amendments** (self-corrections kept, not deleted). |
| `reports/failure_analysis.md`, `docs/experiment_ledger.md` | Reliability program findings and the per-experiment record. |
| `docs/SYNRIA_GR00T_LEDGER.md` | Full narrative ledger, G0 → repair mission. |
| `docs/handoff-2026-07-31-2023-CDT.md` | Historical July resumption note; use the September 7 handoff above for current prerequisites. |
| `docs/CLOUD_FINETUNE_FRANKA.md` | Unfrozen-DiT cloud run instructions. |

## Outside the repository (persistent on disk, too large for git)

| Path | Size | What it is |
|---|---|---|
| `~/models/gr00t_franka/gr00t_franka_v6c/checkpoint-5000` | 16 G | **Best open-loop model** (MAE 0.0599 vs 0.57 baseline); first randomized closed-loop completion. |
| `~/models/gr00t_franka/gr00t_franka_v6d/checkpoint-16000` | 16 G | Best action fit (MAE 0.0355), 0/40 closed-loop — the overfitting datapoint. |
| `~/models/gr00t_franka/gr00t_franka_v7/checkpoint-6000` | 16 G | Dwell-corpus model. |
| `~/models/gr00t_franka/gr00t_franka_v5_fixed/checkpoint-5000` | 16 G | Fixed-spawn control — first GR00T task completions. |
| `~/models/gr00t_franka/gr00t_franka_v4`, `v6`, `v6b` | 44–59 G each | Earlier surrogate runs (v6/v6b are the batch-1 failures). |
| `~/models/gr00t_franka/gr00t_franka_v4_smoke` | 28 G | Smoke run — **disposable**. |
| `~/models/gr00t_synria/gr00t_synria_v1,v2,v3` | 220 G total | Original Synria-embodiment fine-tunes (v3 = MAE 0.36 result). |
| `~/models/datasets/franka_ds_v2`, `franka_ds_v3` | 179 M, 211 M | Training copies of the 200- and 226-episode corpora. |
| `~/models/datasets/synria_cup_*`, `vision_raw_all` | ~590 M | Synria-track corpora. |
| `~/models/reference/vendor_descriptions/` | 721 M | Upstream Synria URDF / MJCF / STL meshes — the source for every gripper measurement. |

Serving any checkpoint:

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token) ~/.venv/gr00t/bin/python \
  ~/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
  --model-path ~/models/gr00t_franka/gr00t_franka_v6c/checkpoint-5000 \
  --embodiment-tag new_embodiment --port 5555
```

The cloud bundle (`franka_cloud_package.tar.gz`) is **not** kept: it
exceeded GitHub's 100 MB file limit and every input is already tracked
here. Rebuild it in one command when needed:

```bash
tar czf /tmp/franka_cloud_package.tar.gz \
  reports/training/franka_cup_lerobot_demos_v2 \
  isaac/isaaclab_tasks/synria_pickplace/gr00t/modality_config.py \
  docs/CLOUD_FINETUNE_FRANKA.md
```

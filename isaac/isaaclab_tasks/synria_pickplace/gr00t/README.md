# Synria + NVIDIA Isaac GR00T

NVIDIA Isaac GR00T integration for the Synria pick-and-place task. Targets **GR00T N1.7** (the latest open VLA model as of 2026-05). Follows NVIDIA's published [LeRobot SO-101 post-training recipe](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) with embodiment adjustments for the Synria 6DOF + parallel-jaw gripper + C10 wrist camera.

## File layout

```
isaac/isaaclab_tasks/synria_pickplace/gr00t/
├── __init__.py       # Windows-import-safe surface
├── embodiment.py     # Synria embodiment tag, joint names, modality config
├── dataset.py        # LeRobot dataset adapter + seed/amplified path conventions
├── finetune.py       # python -m ...gr00t.finetune (LoRA fine-tune entry)
├── inference.py      # python -m ...gr00t.inference (eval rollout entry)
└── README.md         # this file
```

Full end-to-end recipe + RTX checklist: [`docs/SYNRIA_GR00T_FINETUNE.md`](../../../../docs/SYNRIA_GR00T_FINETUNE.md).

## Two integration paths

- **Path A — direct fine-tune**: GR00T N1.7 fine-tuned on Synria demonstrations → deployed as the manipulation policy.
- **Path C — GR00T-Mimic data amplification**: small seed set (~100 demos per game) amplified by GR00T-Mimic into 100K+ synthetic trajectories → larger fine-tune dataset.

The two combine cleanly: collect a small seed set, amplify via GR00T-Mimic, fine-tune GR00T on the amplified set, evaluate inside Isaac Lab.

## What this scaffold does NOT do

- Doesn't ship a fine-tuned checkpoint.
- Doesn't ship demonstrations — those are collected on RTX (teleop or scripted).
- Doesn't lock down the exact GR00T API version — N1 → N1.5 → N1.7 has changed module paths; the scaffold uses the N1.7 API names and will likely need light edits on whichever version you install.

## On Windows

All Python files import cleanly (try / except gates around GR00T / gymnasium / huggingface_hub). Invoking `finetune.main()` or `inference.main()` raises a clear `SystemExit` pointing at the RTX requirement.

Tests:

```bash
pytest tests/test_gr00t_scaffold_import.py -v
```

## On RTX 5090

```bash
# 1. Install GR00T (current public package name; verify on install)
pip install isaac-gr00t

# 2. Confirm Synria joint names from the actual URDF
#    (edit embodiment.py SYNRIA_ARM_JOINT_NAMES if they differ from joint_1..joint_6)
python -c "import xml.etree.ElementTree as ET; root = ET.parse('isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf').getroot(); print([j.attrib['name'] for j in root.findall('joint')])"

# 3. Collect seed demonstrations (~100 per game, LeRobot format)
#    Either teleop with a leader arm or scripted demonstrations from the
#    Isaac Lab task config with PPO-style random exploration.
#    Output paths: reports/training/synria_<game>_lerobot_demos/

# 4. Amplify with GR00T-Mimic (optional, Path C)
#    Output paths: reports/training/synria_<game>_mimic_amplified/

# 5. Fine-tune
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \
    --game chess \
    --amplified \
    --base-model nvidia/GR00T-N1.7-2B \
    --output-dir runs/gr00t_synria_chess_v0 \
    --num-epochs 10 \
    --batch-size 16 \
    --lora-rank 16

# 6. Evaluate inside the Isaac Lab task
python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.inference \
    --task Synria-Chess-PickPlace-v0 \
    --checkpoint runs/gr00t_synria_chess_v0 \
    --num-episodes 20 \
    --report reports/training/gr00t_synria_chess_eval.json
```

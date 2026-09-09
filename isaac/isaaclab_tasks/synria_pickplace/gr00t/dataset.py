"""Dataset adapter: LeRobot demonstrations ↔ GR00T input format.

GR00T N1.7's fine-tuning entry (``gr00t/experiment/launch_finetune.py``)
consumes LeRobot-v2-format dataset roots passed via ``--dataset-path`` and
slices their flat state/action vectors using the dataset's
``meta/modality.json``. This module owns:

    1. The on-disk path conventions for Synria demos (per-game).
    2. The ``meta/modality.json`` payload + writer that binds the
       recorder's column layout to the keys in ``modality_config.py``.
    3. The GR00T-Mimic seed-set convention — small set of human / scripted
       demos that GR00T-Mimic amplifies into the larger training set.

The actual demo collection (teleop, scripted, or via Isaac Lab rollouts)
happens on the RTX side; this scaffold codifies where each game's demos
land and how to wire them into the GR00T training run.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Per-game demo paths. Stored under reports/training/ because LeRobot
# demonstrations are validation/evidence artifacts, not source code.
SYNRIA_DEMO_ROOTS: dict[str, Path] = {
    "ludo": _REPO_ROOT / "reports" / "training" / "synria_ludo_lerobot_demos",
    "chess": _REPO_ROOT / "reports" / "training" / "synria_chess_lerobot_demos",
    "checkers": _REPO_ROOT / "reports" / "training" / "synria_checkers_lerobot_demos",
}

# GR00T-Mimic amplified dataset paths. Output of running GR00T-Mimic
# (Cosmos-based augmentation) on the seed demos above.
SYNRIA_MIMIC_AMPLIFIED_ROOTS: dict[str, Path] = {
    "ludo": _REPO_ROOT / "reports" / "training" / "synria_ludo_mimic_amplified",
    "chess": _REPO_ROOT / "reports" / "training" / "synria_chess_mimic_amplified",
    "checkers": _REPO_ROOT / "reports" / "training" / "synria_checkers_mimic_amplified",
}

# Seed-set sizing — the SO-101 recipe used ~50-200 human demos as the seed.
# Same magnitude here. GR00T-Mimic typically amplifies seed × 1000 in
# ~10 hours of A100 / H100 compute (per NVIDIA's public benchmark).
RECOMMENDED_SEED_DEMOS_PER_GAME = 100
EXPECTED_MIMIC_AMPLIFICATION = 1000  # 100 seed → 100K synthetic trajectories


def modality_json_spec() -> dict:
    """Return the ``meta/modality.json`` payload for a Synria demo dataset.

    GR00T N1.7 reads this file from the dataset root to slice the flat
    LeRobot ``observation.state`` / ``action`` vectors into named
    modalities. Index ranges MUST match the recorder's column layout and
    the keys MUST match ``modality_config.py``:

        state/action[0:6] — Joint1..Joint6 positions (rad)
        state/action[6:8] — left_finger (0..0.025 m), right_finger
                            (0..-0.025 m)
    """
    return {
        "state": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 8},
        },
        "action": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 8},
        },
        "video": {
            "wrist": {"original_key": "observation.images.wrist"},
            "overhead": {"original_key": "observation.images.overhead"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def write_modality_json(dataset_root: Path) -> Path:
    """Write ``meta/modality.json`` into a recorded LeRobot dataset root."""
    import json

    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "modality.json"
    out.write_text(json.dumps(modality_json_spec(), indent=4) + "\n")
    return out

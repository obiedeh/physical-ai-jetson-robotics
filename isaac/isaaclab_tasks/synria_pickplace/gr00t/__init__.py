"""NVIDIA Isaac GR00T integration for the Synria pick-and-place task.

Targets GR00T N1.7 (latest as of 2026-05). The pattern mirrors NVIDIA's
public post-training recipe for the LeRobot SO-101 single-arm
manipulator — the Synria 6DOF arm is structurally similar (6 revolute
joints + parallel-jaw gripper + wrist camera) so the same recipe applies
with embodiment registration changes.

Pipeline (Windows = scaffold-only; RTX = runtime):

    1. embodiment.py — register Synria as a GR00T embodiment
       (joint names, action space, camera config).
    2. dataset.py   — collect/convert demonstrations into LeRobot format
       that GR00T can consume.
    3. finetune.py  — post-train GR00T N1.7 on the Synria dataset
       (LoRA-style on RTX 5090; full fine-tune likely needs cloud GPU).
    4. inference.py — load the fine-tuned checkpoint and run inside the
       Isaac Lab task to validate against the V1 pick-and-place spec.

Two integration paths are supported (combine in the same fine-tune):

    Path A — fine-tune GR00T directly on Synria demos as the deployed policy.
    Path C — use GR00T-Mimic to amplify a small seed dataset
             (~100 demos -> 100K+ synthetic trajectories) before fine-tuning.

This package imports without GR00T / huggingface_hub / transformers
installed — same Windows-import-safe pattern as the parent task scaffold.
"""

from __future__ import annotations

try:
    import gr00t  # type: ignore[import-not-found]  # noqa: F401

    _GR00T_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware-gated import
    _GR00T_AVAILABLE = False


from .embodiment import (
    SYNRIA_EMBODIMENT_TAG,
    SynriaEmbodimentConfig,
    synria_modality_config,
)

__all__ = [
    "SYNRIA_EMBODIMENT_TAG",
    "SynriaEmbodimentConfig",
    "synria_modality_config",
]

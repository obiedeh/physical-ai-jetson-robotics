"""Synria 6DOF arm — GR00T embodiment registration.

GR00T fine-tuning is parameterized by an ``embodiment_tag`` (a short string
ID) and a ``modality_config`` (which joints, action dimensions, cameras,
and image sizes the model should expect). This module owns the Synria
values; the fine-tuning entry point in ``finetune.py`` passes them through
to GR00T's training API.

Reference: NVIDIA's public post-training recipe for the LeRobot SO-101 single
arm uses ``--embodiment_tag new_embodiment`` and a custom modality config.
The Synria adaptation here follows the same shape with Synria joint names
+ the C10 wrist camera.

This module is Windows-import-safe — it only declares constants and pure
dataclasses, no GR00T runtime calls. The actual embodiment registration
happens when ``finetune.py`` or ``inference.py`` runs on the RTX 5090.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Embodiment tag the fine-tuning CLI passes via --embodiment-tag.
# GR00T N1.7 requires a tag from the EmbodimentTag enum; custom robots use
# "new_embodiment" (arbitrary strings are rejected by EmbodimentTag.resolve).
# The Synria layout is bound to this tag by modality_config.py at import.
SYNRIA_EMBODIMENT_TAG = "new_embodiment"

# Synria joint names as exposed by the SolidWorks-export URDF
# (verified against isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf
# and Isaac Lab env_cfg.py — Synria uses capital-J `Joint1..Joint6`).
SYNRIA_ARM_JOINT_NAMES: list[str] = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]

# The Synria 50mm gripper has TWO independent prismatic finger joints
# (left + right), per the SolidWorks URDF and the Isaac Lab actuator config.
# Treat as 2-DOF action space, not a single open-width command.
SYNRIA_GRIPPER_JOINT_NAMES: list[str] = ["left_finger", "right_finger"]

# Action space: 6 arm-joint position targets + 2 finger position targets.
# Matches Isaac Lab's 8-float action space documented in
# isaac/isaaclab_tasks/synria_pickplace/README.md.
SYNRIA_ACTION_DIM: int = 8

# Camera keys as they appear in the LeRobot dataset and modality_config.py.
# GR00T-Mimic and GR00T-Dreams primarily use wrist + overhead views.
SYNRIA_CAMERA_KEYS: list[str] = [
    "wrist",     # Synria C10 wrist camera (per vendor URDF mount frame)
    "overhead",  # overhead board view (recorder env variant)
]
SYNRIA_CAMERA_RES: tuple[int, int] = (224, 224)  # GR00T standard input size


@dataclass(frozen=True)
class SynriaEmbodimentConfig:
    """All Synria-specific values GR00T needs at fine-tune + inference time."""

    embodiment_tag: str = SYNRIA_EMBODIMENT_TAG
    arm_joints: list[str] = field(default_factory=lambda: list(SYNRIA_ARM_JOINT_NAMES))
    gripper_joints: list[str] = field(default_factory=lambda: list(SYNRIA_GRIPPER_JOINT_NAMES))
    action_dim: int = SYNRIA_ACTION_DIM
    camera_keys: list[str] = field(default_factory=lambda: list(SYNRIA_CAMERA_KEYS))
    image_size: tuple[int, int] = SYNRIA_CAMERA_RES


def synria_modality_config() -> dict[str, Any]:
    """Return the GR00T ``modality_config`` dict for Synria.

    GR00T's training CLI accepts ``--modality_config <yaml-or-dict>`` and
    expects keys naming the observation modalities and the action head.
    Schema follows the SO-101 recipe; concrete key names may shift between
    GR00T versions, so this is a starting point — adjust on RTX against
    the installed GR00T version.

    Returns:
        Mapping that can be serialized to YAML and passed to GR00T's
        fine-tuning entry point.
    """
    cfg = SynriaEmbodimentConfig()

    return {
        "embodiment_tag": cfg.embodiment_tag,
        "state": {
            # 6 joint positions + 6 joint velocities + 1 gripper width
            "arm_joint_pos": {"shape": [len(cfg.arm_joints)], "dtype": "float32"},
            "arm_joint_vel": {"shape": [len(cfg.arm_joints)], "dtype": "float32"},
            "gripper_width": {"shape": [1], "dtype": "float32"},
        },
        "vision": {
            cam: {"shape": [*cfg.image_size, 3], "dtype": "uint8"}
            for cam in cfg.camera_keys
        },
        "language": {
            # Free-form task instruction; e.g. "pick the white pawn on e2 and
            # place it on the right staging zone". GR00T N1.5+ supports
            # language-conditioned policies natively.
            "instruction": {"max_length": 128},
        },
        "action": {
            "joint_target": {"shape": [len(cfg.arm_joints)], "dtype": "float32"},
            # 2-DOF finger action: left_finger (0..+0.025 m) + right_finger
            # (-0.025..0 m). Matches the Isaac Lab ActionsCfg shape.
            "finger_target": {"shape": [len(cfg.gripper_joints)], "dtype": "float32"},
        },
    }

"""Synria modality config registration for GR00T N1.7 fine-tuning.

This file is passed to ``launch_finetune.py --modality-config-path`` and is
imported at training start; importing it registers the Synria modality
layout under ``EmbodimentTag.NEW_EMBODIMENT`` (custom robots must use the
``new_embodiment`` tag — arbitrary tag strings are rejected by
``EmbodimentTag.resolve``).

Layout mirrors ``examples/SO100/so100_config.py`` from the Isaac-GR00T
repo, adapted to the Synria 6DOF arm + 2-finger gripper:

- state:  ``arm`` = 6 joint positions (rad), ``gripper`` = 2 finger
  positions (m) — raw 8-dim, matching the Isaac Lab env exactly so
  recorded actions can be compared 1:1 against predictions.
- action: same 8-dim split; arm uses RELATIVE representation (delta from
  current state, per the SO-101 recipe), gripper ABSOLUTE (open/close
  targets work better absolute).
- video:  ``wrist`` + ``overhead`` RGB from the recorder env variant.

The matching per-dataset ``meta/modality.json`` is produced by
``dataset.write_modality_json``.
"""

from __future__ import annotations

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

synria_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist", "overhead"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # ABSOLUTE, not RELATIVE (G23): our recorded actions are offsets
            # from a CONSTANT default pose — already absolute up to a shift.
            # RELATIVE (delta from current state) mixed PD-lagged state into
            # the target; fine-tune v1's open-loop MAE (1.79 rad) was worse
            # than a predict-zeros baseline (1.0) while loss fell normally.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(synria_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

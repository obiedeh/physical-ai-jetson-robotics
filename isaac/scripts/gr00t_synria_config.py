# Synria/Alicia-D 6-DOF modality config for GR00T N1.7 fine-tuning (E41).
# Mirrors the July gr00t_synria_v3 experiment_cfg exactly: video wrist +
# overhead, state/action arm(6) + gripper(2), 16-step ABSOLUTE joint-space
# action horizon, task-description language conditioning.
# Registered under the NEW_EMBODIMENT tag; pass via --modality-config-path.

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

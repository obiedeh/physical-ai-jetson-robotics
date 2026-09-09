"""LeRobot data layer for the Synria 6DOF arm.

This package is the repo-local implementation of the LeRobot / ALOHA-compatible
recording and imitation-learning workflow described in ``docs/LEROBOT_ALOHA.md``
and ``lerobot/configs/synria_aloha_act_notes.yaml``.

It is intentionally framework-agnostic: no dependency on the external ``lerobot``
package is required. When LeRobot is available (installed via ``pyproject.toml``
extras or ``scripts/linux_rtx/``), its tooling can be layered on top of the
data types defined here.

Public surface:
    schema    — observation, action, and episode data contracts
    dataset   — synthetic episode generator and dataset wrapper
    recorder  — episode recording interface (hardware-gated, mock by default)
    policy_eval — deterministic policy evaluation helpers
    cube_sort — colored-cube sorting simulation (MockColorDetector, CubeSortSimulation)
"""

from lerobot.cube_sort import (
    COLOR_TO_ZONE,
    CUBE_COLORS,
    CubeDetection,
    CubeSortSimulation,
    MockColorDetector,
)
from lerobot.dataset import (
    SynriaEpisodeDataset,
    generate_synthetic_episode,
    write_dataset_summary,
)
from lerobot.policy_eval import (
    DeterministicArmPolicy,
    PolicyEvalResult,
    evaluate_policy_on_dataset,
    evaluate_policy_on_episode,
)
from lerobot.recorder import EpisodeRecorder, RecordingSession
from lerobot.schema import (
    ActionFrame,
    EEPose,
    Episode,
    EpisodeMetadata,
    EpisodeStep,
    JointState,
    ObservationFrame,
)

__all__ = [
    # schema
    "ActionFrame",
    "EEPose",
    "Episode",
    "EpisodeMetadata",
    "EpisodeStep",
    "JointState",
    "ObservationFrame",
    # dataset
    "SynriaEpisodeDataset",
    "generate_synthetic_episode",
    "write_dataset_summary",
    # recorder
    "EpisodeRecorder",
    "RecordingSession",
    # policy_eval
    "DeterministicArmPolicy",
    "PolicyEvalResult",
    "evaluate_policy_on_episode",
    "evaluate_policy_on_dataset",
    # cube_sort
    "CUBE_COLORS",
    "COLOR_TO_ZONE",
    "CubeDetection",
    "CubeSortSimulation",
    "MockColorDetector",
]

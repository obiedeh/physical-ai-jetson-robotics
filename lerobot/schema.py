"""Observation, action, and episode data contracts for the Synria LeRobot track.

These types define the canonical representation of a Synria arm demonstration
episode as described in ``docs/LEROBOT_ALOHA.md`` and
``lerobot/configs/synria_aloha_act_notes.yaml``.

Observation contract
---------------------
* **joint_state**: 6 joint positions (rad) + 6 joint velocities (rad/s) +
  gripper open width (m) → 13 scalars
* **ee_pose**: 3 position (m) + 4 quaternion (w, x, y, z) → 7 scalars
  (derived from planar FK stub until a full DH table is confirmed)

Action contract
---------------
* **joint_deltas_rad**: 6-DOF Δjoint per step (rad)
* **gripper_command**: scalar in [−1.0, +1.0] — negative = close, positive = open

Both contracts match the ACT / ALOHA format used by
``Synria-Robotics/lerobot`` and the upstream LeRobot library.

Usage::

    from lerobot.schema import JointState, EEPose, ActionFrame, Episode

    js = JointState(
        positions_rad=(0.0,) * 6,
        velocities_rad_s=(0.0,) * 6,
        gripper_open_m=0.0,
        timestamp_s=0.0,
    )
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES: tuple[str, ...] = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
N_JOINTS: int = len(JOINT_NAMES)

#: Maximum gripper open width in metres (Synria Alicia-D spec).
GRIPPER_OPEN_M: float = 0.085
#: Gripper fully closed.
GRIPPER_CLOSED_M: float = 0.0

VALID_TASK_VARIANTS: tuple[str, ...] = ("V1", "V2", "V3")
VALID_GAMES: tuple[str, ...] = ("ludo", "chess", "checkers")
VALID_STAGING_ZONES: tuple[str, ...] = ("left", "right", "top", "bottom")

DEFAULT_FPS: float = 30.0
DEFAULT_ROBOT_ID: str = "synria-arm-01"


# ---------------------------------------------------------------------------
# Observation components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointState:
    """Joint-space observation: positions, velocities, and gripper width.

    Attributes:
        positions_rad: 6-tuple of joint angles in radians.
        velocities_rad_s: 6-tuple of joint velocities in rad/s.
        gripper_open_m: Gripper open width in metres
            (0.0 = closed, GRIPPER_OPEN_M = fully open).
        timestamp_s: Wall-clock timestamp of the sample (seconds).
    """

    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    gripper_open_m: float
    timestamp_s: float

    def __post_init__(self) -> None:
        if len(self.positions_rad) != N_JOINTS:
            raise ValueError(
                f"positions_rad must have {N_JOINTS} elements, "
                f"got {len(self.positions_rad)}."
            )
        if len(self.velocities_rad_s) != N_JOINTS:
            raise ValueError(
                f"velocities_rad_s must have {N_JOINTS} elements, "
                f"got {len(self.velocities_rad_s)}."
            )
        if not (GRIPPER_CLOSED_M <= self.gripper_open_m <= GRIPPER_OPEN_M):
            raise ValueError(
                f"gripper_open_m={self.gripper_open_m} out of range "
                f"[{GRIPPER_CLOSED_M}, {GRIPPER_OPEN_M}]."
            )

    def as_vector(self) -> tuple[float, ...]:
        """Return flat 13-float observation vector: positions + velocities + gripper."""
        return (*self.positions_rad, *self.velocities_rad_s, self.gripper_open_m)

    def as_dict(self) -> dict[str, object]:
        return {
            "positions_rad": list(self.positions_rad),
            "velocities_rad_s": list(self.velocities_rad_s),
            "gripper_open_m": self.gripper_open_m,
            "timestamp_s": self.timestamp_s,
        }


@dataclass(frozen=True)
class EEPose:
    """End-effector pose: XYZ position (m) + unit quaternion orientation.

    The quaternion convention is (w, x, y, z). An identity quaternion
    represents the arm's zero-rotation reference frame.

    Attributes:
        x_m: End-effector X position in metres.
        y_m: End-effector Y position in metres.
        z_m: End-effector Z position in metres (above table surface).
        qw: Quaternion scalar part.
        qx: Quaternion x component.
        qy: Quaternion y component.
        qz: Quaternion z component.
        timestamp_s: Wall-clock timestamp.
    """

    x_m: float
    y_m: float
    z_m: float
    qw: float
    qx: float
    qy: float
    qz: float
    timestamp_s: float

    def as_vector(self) -> tuple[float, ...]:
        """Return 7-float pose vector: (x, y, z, qw, qx, qy, qz)."""
        return (self.x_m, self.y_m, self.z_m, self.qw, self.qx, self.qy, self.qz)

    def as_dict(self) -> dict[str, object]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "qw": self.qw,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "timestamp_s": self.timestamp_s,
        }


@dataclass(frozen=True)
class ObservationFrame:
    """One time-step observation bundling joint state and end-effector pose.

    Attributes:
        joint_state: 6-DOF joint state + gripper.
        ee_pose: End-effector Cartesian pose.
        frame_index: Zero-based index within the episode.
        timestamp_s: Wall-clock timestamp.
    """

    joint_state: JointState
    ee_pose: EEPose
    frame_index: int
    timestamp_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "joint_state": self.joint_state.as_dict(),
            "ee_pose": self.ee_pose.as_dict(),
        }


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionFrame:
    """One time-step action: joint deltas and gripper command.

    Attributes:
        joint_deltas_rad: 6-tuple of target Δjoint values (radians).
            Applied by adding to the current joint positions.
        gripper_command: Gripper open/close command in [−1.0, +1.0].
            +1.0 = fully open, −1.0 = fully closed.
        timestamp_s: Wall-clock timestamp of the command.
    """

    joint_deltas_rad: tuple[float, ...]
    gripper_command: float
    timestamp_s: float

    def __post_init__(self) -> None:
        if len(self.joint_deltas_rad) != N_JOINTS:
            raise ValueError(
                f"joint_deltas_rad must have {N_JOINTS} elements, "
                f"got {len(self.joint_deltas_rad)}."
            )
        if not (-1.0 <= self.gripper_command <= 1.0):
            raise ValueError(
                f"gripper_command={self.gripper_command} out of range [−1.0, +1.0]."
            )

    def as_vector(self) -> tuple[float, ...]:
        """Return 7-float action vector: (Δj1, …, Δj6, gripper)."""
        return (*self.joint_deltas_rad, self.gripper_command)

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_deltas_rad": list(self.joint_deltas_rad),
            "gripper_command": self.gripper_command,
            "timestamp_s": self.timestamp_s,
        }


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStep:
    """One (observation, action) pair within an episode.

    Attributes:
        observation: Arm + EE state at this step.
        action: Command issued at this step.
        step_index: Zero-based step number within the episode.
    """

    observation: ObservationFrame
    action: ActionFrame
    step_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "observation": self.observation.as_dict(),
            "action": self.action.as_dict(),
        }


@dataclass(frozen=True)
class EpisodeMetadata:
    """Identity and provenance of one demonstration episode.

    Attributes:
        episode_id: Unique episode identifier.
        task_variant: Pick-place task variant — ``"V1"``, ``"V2"``, or ``"V3"``.
        game: Board game context — ``"ludo"``, ``"chess"``, or ``"checkers"``.
        staging_zone: Target staging zone — ``"left"``, ``"right"``, ``"top"``, or
            ``"bottom"``.
        robot_id: Robot that performed (or will perform) the demonstration.
        fps: Recording / generation frame rate in Hz.
        n_steps: Total number of (obs, action) steps in the episode.
        captured_at: ISO-8601 UTC timestamp of recording.
        synthetic: ``True`` when generated programmatically; ``False`` for real
            hardware demonstrations.
        notes: Free-text context (surface, speed, conditions, etc.).
    """

    episode_id: str
    task_variant: str
    game: str
    staging_zone: str
    robot_id: str
    fps: float
    n_steps: int
    captured_at: str
    synthetic: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task_variant not in VALID_TASK_VARIANTS:
            raise ValueError(
                f"task_variant={self.task_variant!r} must be one of "
                f"{VALID_TASK_VARIANTS}."
            )
        if self.game not in VALID_GAMES:
            raise ValueError(
                f"game={self.game!r} must be one of {VALID_GAMES}."
            )
        if self.staging_zone not in VALID_STAGING_ZONES:
            raise ValueError(
                f"staging_zone={self.staging_zone!r} must be one of "
                f"{VALID_STAGING_ZONES}."
            )
        if self.fps <= 0.0:
            raise ValueError(f"fps={self.fps} must be positive.")
        if self.n_steps < 0:
            raise ValueError(f"n_steps={self.n_steps} must be non-negative.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Episode:
    """A complete demonstration episode: metadata + ordered (obs, action) steps.

    Attributes:
        metadata: Episode provenance and task context.
        steps: Ordered list of :class:`EpisodeStep` objects.
    """

    metadata: EpisodeMetadata
    steps: list[EpisodeStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[EpisodeStep]:
        return iter(self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.as_dict(),
            "steps": [s.as_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()

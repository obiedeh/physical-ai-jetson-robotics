"""Pure-Python data contract for physical Synria demonstrations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TypedDict

from synria_lerobot.task_registry import (
    EPISODE_WINDOW_KEYS as EPISODE_WINDOW_KEYS,
)
from synria_lerobot.task_registry import (
    TaskDefinition,
)
from synria_lerobot.task_registry import (
    episode_window_metadata as episode_window_metadata,
)

CONTRACT_VERSION = "synria_physical_v1"
DRIVER_JOINT_NAMES = (
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
)
SIMULATION_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
JOINT_NAME_MAPPING = dict(zip(DRIVER_JOINT_NAMES, SIMULATION_JOINT_NAMES, strict=True))
GRIPPER_STROKE_M = {"50mm": 0.025, "100mm": 0.05}
IMAGE_KEYS = ("observation.images.wrist", "observation.images.front")
ACTION_TIMING_KEYS = (
    "action_lookahead_steps", "effective_action_lookahead_steps",
    "nominal_action_lookahead_s", "requested_rate_hz",
)
DIRECT_JOINT_COMMAND_TOPIC = "/joint_commands"
DEFAULT_COMMAND_TOPICS = (DIRECT_JOINT_COMMAND_TOPIC, "/policy_joint_targets")
STATE_SOURCE_KINDS = ("standalone_driver", "ros2_control")


def require_absolute_topic(topic: str) -> None:
    """Reject relative or malformed graph names before checking command publishers."""
    if not isinstance(topic, str) or re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", topic) is None:
        raise ValueError("topic must be an explicit absolute ROS topic name")


def guarded_command_topics(additional: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Retain mandatory command topics while adding distinct operator declarations."""
    topics = tuple(dict.fromkeys((*DEFAULT_COMMAND_TOPICS, *additional)))
    for topic in topics:
        require_absolute_topic(topic)
    return topics


@dataclass(frozen=True)
class StateSourceProvenance:
    """Operator declaration, not automatic identification of the publishing node."""

    state_source: str
    follower_topic: str
    guarded_command_topics: tuple[str, ...] = DEFAULT_COMMAND_TOPICS
    declaration: str = "operator-declared"

    def __post_init__(self) -> None:
        """Reject malformed identity, dimensions or timing at the data boundary."""
        if self.state_source not in STATE_SOURCE_KINDS or self.declaration != "operator-declared":
            raise ValueError("state source requires an explicit operator declaration")
        require_absolute_topic(self.follower_topic)
        if not isinstance(self.guarded_command_topics, (tuple, list)):
            raise ValueError("guarded command topics must be a sequence")
        topics = tuple(self.guarded_command_topics)
        if topics != guarded_command_topics(topics):
            raise ValueError("guarded command topics must retain defaults without duplicates")
        object.__setattr__(self, "guarded_command_topics", topics)

    def as_dict(self) -> dict[str, Any]:
        """Serialize canonical evidence fields for exact contract and resume comparison."""
        return {
            "state_source": self.state_source, "follower_topic": self.follower_topic,
            "guarded_command_topics": list(self.guarded_command_topics),
            "declaration": self.declaration,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> StateSourceProvenance:
        """Construct a validated object from explicitly serialized evidence."""
        required = {"state_source", "follower_topic", "guarded_command_topics", "declaration"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("incomplete or invalid operator-declared state source evidence")
        return cls(**payload)


class ActionSource(str, Enum):
    """Origin of the seven-float physical action vector."""

    LEADER = "leader"
    NEXT_STATE = "next_state"


class ActionTimingMetadata(TypedDict):
    """Describe configured and effective lookahead without claiming measured latency."""
    action_lookahead_steps: int
    effective_action_lookahead_steps: int
    nominal_action_lookahead_s: float
    requested_rate_hz: float


@dataclass(frozen=True)
class StateRateMeasurement:
    """One callback-count measurement, not an assumed driver publication rate."""

    rate_hz: float
    message_count: int
    duration_s: float
    started_monotonic_s: float
    ended_monotonic_s: float
    max_callback_gap_s: float

    def __post_init__(self) -> None:
        """Require callback count, elapsed interval, rate and maximum gap to agree."""
        values = (self.rate_hz, self.duration_s, self.started_monotonic_s,
                  self.ended_monotonic_s, self.max_callback_gap_s)
        if not all(not isinstance(value, bool) and math.isfinite(value) for value in values):
            raise ValueError("state rate measurement must be finite")
        if type(self.message_count) is not int or self.message_count <= 0 or self.duration_s < 2:
            raise ValueError("state rate measurement requires callbacks over at least two seconds")
        if not math.isclose(self.ended_monotonic_s - self.started_monotonic_s, self.duration_s):
            raise ValueError("state rate measurement interval is inconsistent")
        if not math.isclose(self.rate_hz, self.message_count / self.duration_s):
            raise ValueError("state rate measurement count and rate are inconsistent")
        if not 0 <= self.max_callback_gap_s <= self.duration_s:
            raise ValueError("state rate callback gap is outside the measured interval")


def action_timing_metadata(
    source: ActionSource, steps: int, fps: float
) -> ActionTimingMetadata:
    """Describe configured and effective horizons using the requested sample rate."""
    if type(steps) is not int or steps < 0:
        raise ValueError("action lookahead steps must be a non-negative integer")
    if not isinstance(source, ActionSource):
        raise ValueError("action source must be leader or next_state")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0 or int(fps) != fps:
        raise ValueError("requested sample rate must be a positive finite integer")
    effective_steps = steps if source is ActionSource.NEXT_STATE else 0
    return {
        "action_lookahead_steps": steps,
        "effective_action_lookahead_steps": effective_steps,
        "nominal_action_lookahead_s": effective_steps / fps,
        "requested_rate_hz": fps,
    }


def _require_finite(values: tuple[float, ...], label: str) -> None:
    """Refuse nonfinite vector components before recording or mapping values."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")


@dataclass(frozen=True)
class PhysicalDatasetContract:
    """Dataset-wide fields that cannot change between physical episodes."""

    gripper_type: str
    action_source: ActionSource
    state_has_velocity: bool
    version: str = CONTRACT_VERSION
    action_lookahead_steps: int = 1
    task_id: str = field(kw_only=True)
    task_definition: TaskDefinition = field(kw_only=True)
    recording_purpose: str = field(default="qualifying", kw_only=True)

    def __post_init__(self) -> None:
        """Reject malformed identity, dimensions or timing at the data boundary."""
        if self.gripper_type not in GRIPPER_STROKE_M:
            raise ValueError("gripper_type must be '50mm' or '100mm'")
        if self.version != CONTRACT_VERSION:
            raise ValueError(f"unsupported physical contract version: {self.version}")
        action_timing_metadata(self.action_source, self.action_lookahead_steps, 1)
        if not isinstance(self.task_definition, TaskDefinition) or self.task_id != (
            self.task_definition.task_id
        ):
            raise ValueError("contract requires the matching fixed-scene task snapshot")
        self.task_definition.recording_window(self.recording_purpose)

    @property
    def min_episode_s(self) -> float:
        """Read the effective task minimum for this dataset purpose."""
        return self.task_definition.recording_window(self.recording_purpose)["min_episode_s"]

    @property
    def max_episode_s(self) -> float:
        """Read the effective task maximum for this dataset purpose."""
        return self.task_definition.recording_window(self.recording_purpose)["max_episode_s"]

    @property
    def task_text(self) -> str:
        """Return the immutable instruction associated with every recorded frame."""
        return self.task_definition.task_text

    def require_qualifying(self) -> None:
        """Refuse smoke and synthetic contracts at training and physical evaluation boundaries."""
        if self.recording_purpose != "qualifying":
            raise ValueError(
                "disposable smoke or synthetic demo is not eligible for training "
                "or physical evaluation"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhysicalDatasetContract:
        """Construct a validated object from explicitly serialized evidence."""
        task = TaskDefinition.from_metadata(payload)
        contract = cls(
            payload["gripper_type"], ActionSource(payload["action_source"]),
            payload["state_has_velocity"], version=payload["contract_version"],
            action_lookahead_steps=payload["action_lookahead_steps"],
            task_id=task.task_id, task_definition=task,
            recording_purpose=payload["recording_purpose"],
        )
        if contract.as_dict(fps=payload.get("requested_rate_hz")) != payload:
            raise ValueError("inconsistent physical contract metadata")
        return contract

    @property
    def gripper_stroke_m(self) -> float:
        """Return the declared hardware gripper stroke in driver units."""
        return GRIPPER_STROKE_M[self.gripper_type]

    def prepare_state(self, state: PhysicalState) -> PhysicalState:
        """Select the dataset's state fields, never infer them from one message."""
        if self.state_has_velocity:
            if state.joint_velocities_rad_s is None:
                raise ValueError("physical contract requires six reported joint velocities")
            return state
        if state.joint_velocities_rad_s is not None:
            return replace(state, joint_velocities_rad_s=None)
        return state

    def as_dict(self, *, fps: float | None = None) -> dict[str, object]:
        """Serialize canonical evidence fields for exact contract and resume comparison."""
        payload: dict[str, object] = {
            "contract_version": self.version,
            "gripper_type": self.gripper_type,
            "action_source": self.action_source.value,
            "state_has_velocity": self.state_has_velocity,
            "state_names": [*DRIVER_JOINT_NAMES, "Gripper"],
            "image_keys": list(IMAGE_KEYS),
            "action_lookahead_steps": self.action_lookahead_steps,
            **self.task_definition.metadata(self.recording_purpose),
        }
        if fps is not None:
            payload.update(
                action_timing_metadata(self.action_source, self.action_lookahead_steps, fps)
            )
        return payload


@dataclass(frozen=True)
class PhysicalState:
    """Follower state with source timestamps and optional reported velocities."""

    joint_positions_rad: tuple[float, ...]
    gripper_m: float
    monotonic_timestamp_s: float
    ros_header_stamp_s: float
    joint_velocities_rad_s: tuple[float, ...] | None = None
    ros_arrival_stamp_s: float | None = None

    def __post_init__(self) -> None:
        """Require six finite positions, one gripper value and consistent optional velocities."""
        if len(self.joint_positions_rad) != 6:
            raise ValueError("joint_positions_rad must contain six values")
        _require_finite(self.joint_positions_rad, "joint_positions_rad")
        _require_finite(
            (self.gripper_m, self.monotonic_timestamp_s, self.ros_header_stamp_s),
            "state scalars",
        )
        if self.gripper_m < 0.0:
            raise ValueError("gripper_m must be non-negative")
        if self.ros_arrival_stamp_s is not None and not math.isfinite(self.ros_arrival_stamp_s):
            raise ValueError("ROS arrival stamp must be finite")
        if self.joint_velocities_rad_s is not None:
            if len(self.joint_velocities_rad_s) != 6:
                raise ValueError("joint_velocities_rad_s must contain six values")
            _require_finite(self.joint_velocities_rad_s, "joint_velocities_rad_s")

    def observation_vector(self) -> tuple[float, ...]:
        """Serialize joint positions and gripper, followed only by declared velocities."""
        base = (*self.joint_positions_rad, self.gripper_m)
        if self.joint_velocities_rad_s is None:
            return base
        return (*base, *self.joint_velocities_rad_s)


@dataclass(frozen=True)
class ImageFrame:
    """One camera frame and its host monotonic capture timestamp."""

    data: Any
    monotonic_timestamp_s: float
    native_resolution: tuple[int, int] | None = None
    source_id: str | None = None
    native_data: Any | None = None

    def __post_init__(self) -> None:
        """Reject malformed identity, dimensions or timing at the data boundary."""
        if not math.isfinite(self.monotonic_timestamp_s):
            raise ValueError("image monotonic timestamp must be finite")
        if self.native_resolution is not None and (
            len(self.native_resolution) != 2
            or any(type(value) is not int or value <= 0 for value in self.native_resolution)
        ):
            raise ValueError("native image resolution must contain positive width and height")


@dataclass(frozen=True)
class PhysicalFrame:
    """One synchronized physical observation/action record."""

    state: PhysicalState
    action: tuple[float, ...]
    action_monotonic_timestamp_s: float
    wrist: ImageFrame
    front: ImageFrame
    sample_monotonic_timestamp_s: float | None = None
    action_ros_header_stamp_s: float | None = None
    action_ros_arrival_stamp_s: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed identity, dimensions or timing at the data boundary."""
        if len(self.action) != 7:
            raise ValueError("action must contain six joint positions and one gripper value")
        _require_finite(self.action, "action")
        if not math.isfinite(self.action_monotonic_timestamp_s):
            raise ValueError("action monotonic timestamp must be finite")

    def timestamps(self) -> dict[str, float]:
        """Preserve independent source and sampling clock values for freshness gates."""
        result = {
            "state_monotonic_s": self.state.monotonic_timestamp_s,
            "state_ros_header_s": self.state.ros_header_stamp_s,
            "action_monotonic_s": self.action_monotonic_timestamp_s,
            "wrist_monotonic_s": self.wrist.monotonic_timestamp_s,
            "front_monotonic_s": self.front.monotonic_timestamp_s,
        }
        for name, value in (
            ("sample_monotonic_s", self.sample_monotonic_timestamp_s),
            ("state_ros_arrival_s", self.state.ros_arrival_stamp_s),
            ("action_ros_header_s", self.action_ros_header_stamp_s),
            ("action_ros_arrival_s", self.action_ros_arrival_stamp_s),
        ):
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True)
class SimulationMapping:
    """Explicit physical-to-simulation joint and gripper mapping."""

    joint_positions_rad: dict[str, float]
    gripper_open_m: float


def map_physical_state_to_simulation(
    state: PhysicalState,
    contract: PhysicalDatasetContract,
    *,
    simulation_gripper_open_m: float,
) -> SimulationMapping:
    """Map driver names and closed-stroke gripper sense to the simulation schema.

    The physical driver reports ``0`` fully open and positive stroke fully
    closed. The simulation schema reports ``0`` fully closed and a positive
    width fully open, so the mapping reverses and scales the physical value.
    """
    if not math.isfinite(simulation_gripper_open_m) or simulation_gripper_open_m <= 0:
        raise ValueError("simulation_gripper_open_m must be positive and finite")
    if state.gripper_m > contract.gripper_stroke_m:
        raise ValueError("physical gripper value exceeds configured stroke")
    joint_positions = {
        JOINT_NAME_MAPPING[driver_name]: value
        for driver_name, value in zip(
            DRIVER_JOINT_NAMES, state.joint_positions_rad, strict=True
        )
    }
    closed_fraction = state.gripper_m / contract.gripper_stroke_m
    return SimulationMapping(
        joint_positions_rad=joint_positions,
        gripper_open_m=simulation_gripper_open_m * (1.0 - closed_fraction),
    )

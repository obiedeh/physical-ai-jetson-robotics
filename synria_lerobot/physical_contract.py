"""Pure-Python data contract for physical Synria demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

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


class ActionSource(str, Enum):
    """Origin of the seven-float physical action vector."""

    LEADER = "leader"
    NEXT_STATE = "next_state"


def _require_finite(values: tuple[float, ...], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")


@dataclass(frozen=True)
class PhysicalDatasetContract:
    """Dataset-wide fields that cannot change between physical episodes."""

    gripper_type: str
    action_source: ActionSource
    state_has_velocity: bool
    version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.gripper_type not in GRIPPER_STROKE_M:
            raise ValueError("gripper_type must be '50mm' or '100mm'")
        if self.version != CONTRACT_VERSION:
            raise ValueError(f"unsupported physical contract version: {self.version}")

    @property
    def gripper_stroke_m(self) -> float:
        return GRIPPER_STROKE_M[self.gripper_type]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.version,
            "gripper_type": self.gripper_type,
            "action_source": self.action_source.value,
            "state_has_velocity": self.state_has_velocity,
            "state_names": [*DRIVER_JOINT_NAMES, "Gripper"],
            "image_keys": list(IMAGE_KEYS),
        }


@dataclass(frozen=True)
class PhysicalState:
    """Follower state with source timestamps and optional reported velocities."""

    joint_positions_rad: tuple[float, ...]
    gripper_m: float
    monotonic_timestamp_s: float
    ros_header_stamp_s: float
    joint_velocities_rad_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.joint_positions_rad) != 6:
            raise ValueError("joint_positions_rad must contain six values")
        _require_finite(self.joint_positions_rad, "joint_positions_rad")
        _require_finite(
            (self.gripper_m, self.monotonic_timestamp_s, self.ros_header_stamp_s),
            "state scalars",
        )
        if self.gripper_m < 0.0:
            raise ValueError("gripper_m must be non-negative")
        if self.joint_velocities_rad_s is not None:
            if len(self.joint_velocities_rad_s) != 6:
                raise ValueError("joint_velocities_rad_s must contain six values")
            _require_finite(self.joint_velocities_rad_s, "joint_velocities_rad_s")

    def observation_vector(self) -> tuple[float, ...]:
        base = (*self.joint_positions_rad, self.gripper_m)
        if self.joint_velocities_rad_s is None:
            return base
        return (*base, *self.joint_velocities_rad_s)


@dataclass(frozen=True)
class ImageFrame:
    """One camera frame and its host monotonic capture timestamp."""

    data: Any
    monotonic_timestamp_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.monotonic_timestamp_s):
            raise ValueError("image monotonic timestamp must be finite")


@dataclass(frozen=True)
class PhysicalFrame:
    """One synchronized physical observation/action record."""

    state: PhysicalState
    action: tuple[float, ...]
    action_monotonic_timestamp_s: float
    wrist: ImageFrame
    front: ImageFrame

    def __post_init__(self) -> None:
        if len(self.action) != 7:
            raise ValueError("action must contain six joint positions and one gripper value")
        _require_finite(self.action, "action")
        if not math.isfinite(self.action_monotonic_timestamp_s):
            raise ValueError("action monotonic timestamp must be finite")

    def timestamps(self) -> dict[str, float]:
        return {
            "state_monotonic_s": self.state.monotonic_timestamp_s,
            "state_ros_header_s": self.state.ros_header_stamp_s,
            "action_monotonic_s": self.action_monotonic_timestamp_s,
            "wrist_monotonic_s": self.wrist.monotonic_timestamp_s,
            "front_monotonic_s": self.front.monotonic_timestamp_s,
        }


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

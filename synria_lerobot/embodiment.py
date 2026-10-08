"""Physical Synria observations; no device or simulation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .physical_contract import ImageFrame, PhysicalDatasetContract, PhysicalState


@dataclass(frozen=True)
class SynriaObservation:
    state: PhysicalState
    wrist: ImageFrame
    front: ImageFrame
    task: str


@dataclass(frozen=True)
class SynriaEmbodiment:
    contract: PhysicalDatasetContract
    name: str = "synria_alicia_d"

    def observation(self, value: SynriaObservation) -> dict[str, Any]:
        if value.state.gripper_m > self.contract.gripper_stroke_m:
            raise ValueError("gripper exceeds contract stroke")
        if self.contract.state_has_velocity != (value.state.joint_velocities_rad_s is not None):
            raise ValueError("velocity availability differs from contract")
        if value.wrist.data is None or value.front.data is None:
            raise ValueError("both camera images are required")
        return {
            **self.contract.as_dict(),
            "embodiment": self.name,
            "observation.state": value.state.observation_vector(),
            "observation.images.wrist": value.wrist.data,
            "observation.images.front": value.front.data,
            "timestamps": {
                "state": value.state.monotonic_timestamp_s,
                "state_ros": value.state.ros_header_stamp_s,
                "wrist": value.wrist.monotonic_timestamp_s,
                "front": value.front.monotonic_timestamp_s,
            },
            "task": value.task,
        }

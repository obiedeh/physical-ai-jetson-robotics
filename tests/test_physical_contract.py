from __future__ import annotations

import pytest

from synria_lerobot.physical_contract import (
    CONTRACT_VERSION,
    DRIVER_JOINT_NAMES,
    IMAGE_KEYS,
    SIMULATION_JOINT_NAMES,
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalFrame,
    PhysicalState,
    map_physical_state_to_simulation,
)


def _state(
    *,
    gripper_m: float = 0.0,
    velocities: tuple[float, ...] | None = None,
) -> PhysicalState:
    return PhysicalState(
        joint_positions_rad=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        gripper_m=gripper_m,
        monotonic_timestamp_s=10.0,
        ros_header_stamp_s=20.0,
        joint_velocities_rad_s=velocities,
    )


def test_contract_requires_gripper_type_and_records_action_source() -> None:
    with pytest.raises(TypeError):
        PhysicalDatasetContract(  # type: ignore[call-arg]
            action_source=ActionSource.LEADER,
            state_has_velocity=False,
        )
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=ActionSource.LEADER,
        state_has_velocity=False,
    )
    payload = contract.as_dict()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["action_source"] == "leader"
    assert payload["image_keys"] == list(IMAGE_KEYS)


def test_observation_has_seven_values_without_reported_velocity() -> None:
    assert _state().observation_vector() == pytest.approx(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0)
    )


def test_observation_appends_velocity_only_when_reported() -> None:
    velocities = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert _state(velocities=velocities).observation_vector()[-6:] == velocities


def test_frame_has_seven_float_action_and_all_timestamps() -> None:
    frame = PhysicalFrame(
        state=_state(),
        action=(0.0,) * 7,
        action_monotonic_timestamp_s=10.01,
        wrist=ImageFrame(data=b"wrist", monotonic_timestamp_s=10.02),
        front=ImageFrame(data=b"front", monotonic_timestamp_s=10.03),
    )
    assert set(frame.timestamps()) == {
        "state_monotonic_s",
        "state_ros_header_s",
        "action_monotonic_s",
        "wrist_monotonic_s",
        "front_monotonic_s",
    }
    with pytest.raises(ValueError, match="action must contain"):
        PhysicalFrame(
            state=_state(),
            action=(0.0,) * 6,
            action_monotonic_timestamp_s=10.01,
            wrist=frame.wrist,
            front=frame.front,
        )


@pytest.mark.parametrize(
    ("gripper_type", "closed_value"), (("50mm", 0.025), ("100mm", 0.05))
)
def test_mapping_names_and_reverses_gripper_sense(
    gripper_type: str, closed_value: float
) -> None:
    contract = PhysicalDatasetContract(
        gripper_type=gripper_type,
        action_source=ActionSource.NEXT_STATE,
        state_has_velocity=False,
    )
    open_mapping = map_physical_state_to_simulation(
        _state(gripper_m=0.0), contract, simulation_gripper_open_m=0.085
    )
    closed_mapping = map_physical_state_to_simulation(
        _state(gripper_m=closed_value), contract, simulation_gripper_open_m=0.085
    )
    assert tuple(open_mapping.joint_positions_rad) == SIMULATION_JOINT_NAMES
    assert tuple(DRIVER_JOINT_NAMES) == tuple(
        f"Joint{index}" for index in range(1, 7)
    )
    assert open_mapping.gripper_open_m == pytest.approx(0.085)
    assert closed_mapping.gripper_open_m == pytest.approx(0.0)

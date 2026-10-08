from __future__ import annotations

import pytest
from test_task_registry import synthetic_task

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
    action_timing_metadata,
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
            task_id="die_into_cup", task_definition=synthetic_task(20, 30),
        )
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=ActionSource.LEADER,
        state_has_velocity=False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
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


@pytest.mark.parametrize("required", [False, True])
def test_contract_selects_velocity_fields(required: bool) -> None:
    contract = PhysicalDatasetContract("50mm", ActionSource.NEXT_STATE, required,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    )
    state = _state(velocities=(0.2,) * 6)
    prepared = contract.prepare_state(state)
    assert len(prepared.observation_vector()) == (13 if required else 7)
    assert state.joint_velocities_rad_s == (0.2,) * 6
    if required:
        with pytest.raises(ValueError, match="requires six reported joint velocities"):
            contract.prepare_state(_state())


@pytest.mark.parametrize("steps", [-1, 1.5, True, "2", None])
def test_contract_rejects_malformed_action_lookahead(steps: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        PhysicalDatasetContract(
            "50mm", ActionSource.NEXT_STATE, False, action_lookahead_steps=steps,
            task_id="die_into_cup", task_definition=synthetic_task(20, 30),
        )


@pytest.mark.parametrize("fps", [0, -1, 30.5, True, float("nan"), float("inf")])
def test_nominal_action_time_requires_supported_requested_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="positive finite integer"):
        action_timing_metadata(ActionSource.NEXT_STATE, 2, fps)


@pytest.mark.parametrize("source", list(ActionSource))
def test_contract_records_configured_and_effective_action_horizon(source: ActionSource) -> None:
    contract = PhysicalDatasetContract("50mm", source, False, action_lookahead_steps=2,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    )
    payload = contract.as_dict(fps=30.0)
    assert payload["action_lookahead_steps"] == 2
    expected_steps = 2 if source is ActionSource.NEXT_STATE else 0
    assert payload["effective_action_lookahead_steps"] == expected_steps
    assert payload["nominal_action_lookahead_s"] == expected_steps / 30
    assert payload["requested_rate_hz"] == 30


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
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
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

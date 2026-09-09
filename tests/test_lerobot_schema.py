"""Tests for lerobot.schema — observation, action, and episode data types."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.schema import (
    GRIPPER_OPEN_M,
    N_JOINTS,
    ActionFrame,
    EEPose,
    Episode,
    EpisodeMetadata,
    EpisodeStep,
    JointState,
    ObservationFrame,
    now_iso,
)

# ---------------------------------------------------------------------------
# JointState
# ---------------------------------------------------------------------------


def _zero_js(t: float = 0.0) -> JointState:
    return JointState(
        positions_rad=(0.0,) * N_JOINTS,
        velocities_rad_s=(0.0,) * N_JOINTS,
        gripper_open_m=GRIPPER_OPEN_M,
        timestamp_s=t,
    )


def test_joint_state_valid() -> None:
    js = _zero_js()
    assert len(js.positions_rad) == N_JOINTS
    assert js.gripper_open_m == GRIPPER_OPEN_M


def test_joint_state_as_vector_length() -> None:
    js = _zero_js()
    vec = js.as_vector()
    assert len(vec) == N_JOINTS * 2 + 1  # positions + velocities + gripper


def test_joint_state_as_dict_keys() -> None:
    js = _zero_js()
    d = js.as_dict()
    assert "positions_rad" in d
    assert "velocities_rad_s" in d
    assert "gripper_open_m" in d
    assert "timestamp_s" in d


def test_joint_state_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="positions_rad must have"):
        JointState(
            positions_rad=(0.0, 0.0),  # too short
            velocities_rad_s=(0.0,) * N_JOINTS,
            gripper_open_m=0.0,
            timestamp_s=0.0,
        )


def test_joint_state_bad_gripper_raises() -> None:
    with pytest.raises(ValueError, match="gripper_open_m"):
        JointState(
            positions_rad=(0.0,) * N_JOINTS,
            velocities_rad_s=(0.0,) * N_JOINTS,
            gripper_open_m=-0.01,  # below 0
            timestamp_s=0.0,
        )


def test_joint_state_gripper_over_max_raises() -> None:
    with pytest.raises(ValueError, match="gripper_open_m"):
        JointState(
            positions_rad=(0.0,) * N_JOINTS,
            velocities_rad_s=(0.0,) * N_JOINTS,
            gripper_open_m=GRIPPER_OPEN_M + 0.001,
            timestamp_s=0.0,
        )


# ---------------------------------------------------------------------------
# EEPose
# ---------------------------------------------------------------------------


def _identity_ee(t: float = 0.0) -> EEPose:
    return EEPose(x_m=0.1, y_m=0.0, z_m=0.4, qw=1.0, qx=0.0, qy=0.0, qz=0.0, timestamp_s=t)


def test_ee_pose_as_vector() -> None:
    ee = _identity_ee()
    vec = ee.as_vector()
    assert len(vec) == 7
    assert vec[3] == pytest.approx(1.0)  # qw


def test_ee_pose_as_dict_keys() -> None:
    ee = _identity_ee()
    d = ee.as_dict()
    for key in ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "timestamp_s"):
        assert key in d


# ---------------------------------------------------------------------------
# ActionFrame
# ---------------------------------------------------------------------------


def _zero_action(t: float = 0.0) -> ActionFrame:
    return ActionFrame(
        joint_deltas_rad=(0.0,) * N_JOINTS,
        gripper_command=0.0,
        timestamp_s=t,
    )


def test_action_frame_valid() -> None:
    af = _zero_action()
    vec = af.as_vector()
    assert len(vec) == N_JOINTS + 1


def test_action_frame_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="joint_deltas_rad must have"):
        ActionFrame(
            joint_deltas_rad=(0.0, 0.0),
            gripper_command=0.0,
            timestamp_s=0.0,
        )


def test_action_frame_gripper_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="gripper_command"):
        ActionFrame(
            joint_deltas_rad=(0.0,) * N_JOINTS,
            gripper_command=1.5,
            timestamp_s=0.0,
        )


def test_action_frame_gripper_extremes_valid() -> None:
    for cmd in (-1.0, 0.0, 1.0):
        af = ActionFrame(
            joint_deltas_rad=(0.0,) * N_JOINTS,
            gripper_command=cmd,
            timestamp_s=0.0,
        )
        assert af.gripper_command == cmd


# ---------------------------------------------------------------------------
# EpisodeMetadata
# ---------------------------------------------------------------------------


def _meta(**kwargs: object) -> EpisodeMetadata:
    defaults: dict[str, object] = dict(
        episode_id="ep-0001",
        task_variant="V1",
        game="chess",
        staging_zone="left",
        robot_id="synria-arm-01",
        fps=30.0,
        n_steps=60,
        captured_at="2026-05-27T10:00:00+00:00",
        synthetic=True,
    )
    defaults.update(kwargs)
    return EpisodeMetadata(**defaults)  # type: ignore[arg-type]


def test_episode_metadata_valid() -> None:
    m = _meta()
    assert m.task_variant == "V1"
    assert m.game == "chess"
    assert m.staging_zone == "left"


def test_episode_metadata_invalid_variant_raises() -> None:
    with pytest.raises(ValueError, match="task_variant"):
        _meta(task_variant="V9")


def test_episode_metadata_invalid_game_raises() -> None:
    with pytest.raises(ValueError, match="game"):
        _meta(game="monopoly")


def test_episode_metadata_invalid_zone_raises() -> None:
    with pytest.raises(ValueError, match="staging_zone"):
        _meta(staging_zone="diagonal")


def test_episode_metadata_invalid_fps_raises() -> None:
    with pytest.raises(ValueError, match="fps"):
        _meta(fps=-1.0)


def test_episode_metadata_as_dict_round_trip() -> None:
    m = _meta()
    d = m.as_dict()
    assert d["task_variant"] == "V1"
    assert d["synthetic"] is True


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def _make_episode(n_steps: int = 3) -> Episode:
    meta = _meta(n_steps=n_steps)
    steps = []
    for i in range(n_steps):
        t = i / 30.0
        js = JointState(
            positions_rad=(0.0,) * N_JOINTS,
            velocities_rad_s=(0.0,) * N_JOINTS,
            gripper_open_m=GRIPPER_OPEN_M,
            timestamp_s=t,
        )
        ee = _identity_ee(t=t)
        obs = ObservationFrame(joint_state=js, ee_pose=ee, frame_index=i, timestamp_s=t)
        act = _zero_action(t=t)
        steps.append(EpisodeStep(observation=obs, action=act, step_index=i))
    return Episode(metadata=meta, steps=steps)


def test_episode_len() -> None:
    ep = _make_episode(5)
    assert len(ep) == 5


def test_episode_to_dict_keys() -> None:
    ep = _make_episode(2)
    d = ep.to_dict()
    assert "metadata" in d
    assert "steps" in d
    assert len(d["steps"]) == 2


def test_episode_to_json_is_valid() -> None:
    ep = _make_episode(2)
    data = json.loads(ep.to_json())
    assert data["metadata"]["task_variant"] == "V1"
    assert len(data["steps"]) == 2


def test_episode_iteration() -> None:
    ep = _make_episode(4)
    for i, step in enumerate(ep):
        assert step.step_index == i


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_now_iso_is_string() -> None:
    ts = now_iso()
    assert isinstance(ts, str)
    assert "T" in ts

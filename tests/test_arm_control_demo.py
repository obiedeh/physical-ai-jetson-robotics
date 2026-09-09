"""Tests for arm_control.demo trajectory sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_control.demo import (
    POSE_HOME,
    STAGING_POSES,
    board_approach_trajectory,
    full_pick_place_sequence,
    home_trajectory,
    staging_zone_trajectory,
    validate_all_demo_trajectories,
)
from arm_control.safety import ArmSafetyGate


def test_home_trajectory_is_valid() -> None:
    traj = home_trajectory()
    assert traj.validate() == []
    assert traj.name == "synria_home"
    assert len(traj.waypoints) == 1


def test_home_trajectory_duration() -> None:
    traj = home_trajectory(duration_s=5.0)
    assert traj.duration_s() == pytest.approx(5.0)


def test_board_approach_trajectory_is_valid() -> None:
    traj = board_approach_trajectory()
    assert traj.validate() == []
    assert len(traj.waypoints) == 3


def test_staging_zone_trajectory_all_zones() -> None:
    for zone in ("left", "right", "top", "bottom"):
        traj = staging_zone_trajectory(zone)
        assert traj.validate() == [], f"Validation failed for zone={zone}"
        assert traj.name == f"synria_staging_{zone}"


def test_staging_zone_trajectory_invalid_zone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown staging zone"):
        staging_zone_trajectory("diagonal")


def test_full_pick_place_sequence_length_and_validity() -> None:
    sequence = full_pick_place_sequence("left")
    assert len(sequence) == 4
    for traj in sequence:
        assert traj.validate() == [], f"Validation failed: {traj.name}"


def test_full_pick_place_sequence_all_zones() -> None:
    for zone in STAGING_POSES:
        sequence = full_pick_place_sequence(zone)
        assert len(sequence) == 4


def test_all_demo_trajectories_pass_safety_gate() -> None:
    validation = validate_all_demo_trajectories()

    for name, issues in validation.items():
        assert issues == [], f"Safety/validation issues in '{name}': {issues}"


def test_all_demo_trajectories_within_urdf_limits() -> None:
    """Explicitly verify every waypoint in every demo trajectory against URDF limits."""
    gate = ArmSafetyGate()
    sequences = [
        home_trajectory(),
        board_approach_trajectory(),
        *[staging_zone_trajectory(z) for z in STAGING_POSES],
        *full_pick_place_sequence("right"),
    ]
    for traj in sequences:
        for wp in traj.waypoints:
            results = gate.check_joint_positions(wp.positions_rad)
            violations = [r for r in results if not r.passed]
            assert violations == [], (
                f"Traj '{traj.name}': safety violation at t={wp.time_from_start_s}s: "
                f"{[r.message for r in violations]}"
            )


def test_pose_home_all_joints_zero() -> None:
    assert all(v == 0.0 for v in POSE_HOME.values())

"""Tests for the arm_control safety, kinematics, and trajectory modules."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make arm_control importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_control.kinematics import (
    LINK_FOREARM_M,
    LINK_UPPER_ARM_M,
    SYNRIA_GEOMETRIC_MAX_REACH_M,
    CartesianPose,
    check_workspace_reachability,
    estimate_max_reach_m,
    forward_kinematics_planar,
)
from arm_control.safety import (
    ArmSafetyConfig,
    ArmSafetyGate,
    SafetyViolation,
)
from arm_control.trajectory import (
    JointTrajectory,
    JointWaypoint,
    TrajectoryExecutionRecord,
    TrajectoryStatus,
)

# ---------------------------------------------------------------------------
# Safety gate tests
# ---------------------------------------------------------------------------


def test_safety_gate_passes_valid_positions() -> None:
    gate = ArmSafetyGate()
    results = gate.check_joint_positions(
        {"joint_1": 0.0, "joint_2": 0.5, "joint_3": -1.0, "joint_4": 1.0}
    )
    assert all(r.passed for r in results)


def test_safety_gate_flags_lower_limit_violation() -> None:
    gate = ArmSafetyGate()
    results = gate.check_joint_positions({"joint_2": -2.0})  # limit is ±1.5708
    violations = [r for r in results if not r.passed]
    assert any(r.violation == SafetyViolation.joint_limit for r in violations)
    assert any(r.joint_name == "joint_2" for r in violations)


def test_safety_gate_flags_upper_limit_violation() -> None:
    gate = ArmSafetyGate()
    results = gate.check_joint_positions({"joint_2": 2.0})  # limit is ±1.5708
    violations = [r for r in results if not r.passed]
    assert any(r.violation == SafetyViolation.joint_limit for r in violations)


def test_safety_gate_allows_boundary_values() -> None:
    gate = ArmSafetyGate()
    results = gate.check_joint_positions({"joint_2": 1.5708, "joint_2_": -1.5708})
    # joint_2_: not a known joint, ignored. joint_2 at exact limit should pass.
    violations = [r for r in results if not r.passed]
    assert not violations


def test_safety_gate_emergency_stop_blocks_all() -> None:
    gate = ArmSafetyGate(ArmSafetyConfig(emergency_stop_active=True))
    results = gate.check_joint_positions({"joint_1": 0.0})
    assert all(r.violation == SafetyViolation.emergency_stop for r in results)


def test_safety_gate_velocity_limit_with_fraction() -> None:
    # joint_1 max vel = 1.5 rad/s; fraction = 0.5 → limit = 0.75 rad/s
    gate = ArmSafetyGate(ArmSafetyConfig(speed_fraction=0.5))
    results = gate.check_joint_velocities({"joint_1": 1.0})
    violations = [r for r in results if not r.passed]
    assert any(r.violation == SafetyViolation.velocity_limit for r in violations)


def test_safety_gate_velocity_passes_within_fraction() -> None:
    gate = ArmSafetyGate(ArmSafetyConfig(speed_fraction=0.5))
    results = gate.check_joint_velocities({"joint_1": 0.5})
    assert all(r.passed for r in results)


def test_safety_gate_validate_collects_position_and_velocity_violations() -> None:
    gate = ArmSafetyGate(ArmSafetyConfig(speed_fraction=0.5))
    results = gate.validate(
        positions={"joint_2": 2.5},  # out of range
        velocities={"joint_1": 2.0},  # too fast
    )
    violations = [r for r in results if not r.passed]
    signals = {r.violation for r in violations}
    assert SafetyViolation.joint_limit in signals
    assert SafetyViolation.velocity_limit in signals


def test_safety_gate_unknown_joints_ignored() -> None:
    gate = ArmSafetyGate()
    results = gate.check_joint_positions({"gripper_joint": 99.0})
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Kinematics tests
# ---------------------------------------------------------------------------


def test_estimate_max_reach_m_matches_sum_of_links() -> None:
    reach = estimate_max_reach_m()
    assert reach == pytest.approx(SYNRIA_GEOMETRIC_MAX_REACH_M, abs=1e-6)
    assert reach > 0.5  # sanity: > 50 cm


def test_workspace_check_reachable_target() -> None:
    target = CartesianPose(x=0.3, y=0.0, z=0.4)
    result = check_workspace_reachability(target)
    assert result.reachable is True
    assert result.radial_distance_m == pytest.approx(target.distance_from_origin(), abs=1e-4)


def test_workspace_check_too_far() -> None:
    target = CartesianPose(x=1.0, y=0.0, z=0.0)  # > SYNRIA_GEOMETRIC_MAX_REACH_M
    result = check_workspace_reachability(target)
    assert result.reachable is False
    assert "exceeds max reach" in result.message


def test_workspace_check_too_close() -> None:
    target = CartesianPose(x=0.01, y=0.0, z=0.01)
    result = check_workspace_reachability(target, min_reach_m=0.05)
    assert result.reachable is False
    assert "singularity" in result.message


def test_workspace_check_custom_max_reach() -> None:
    target = CartesianPose(x=0.4, y=0.0, z=0.0)
    result = check_workspace_reachability(target, max_reach_m=0.3)
    assert result.reachable is False


def test_forward_kinematics_planar_zero_angles() -> None:
    result = forward_kinematics_planar(0.0, 0.0)
    # At zero angles both links point straight up in Z.
    expected_z = 0.09 + LINK_UPPER_ARM_M + LINK_FOREARM_M  # base_height + upper + forearm
    assert result.x_m == pytest.approx(0.0, abs=1e-4)
    assert result.z_m == pytest.approx(expected_z, abs=1e-4)


def test_forward_kinematics_planar_full_horizontal() -> None:
    # At q2=pi/2 the upper arm points horizontally forward.
    result = forward_kinematics_planar(math.pi / 2, 0.0)
    assert result.x_m > 0.0
    assert result.reach_m > 0.0


def test_cartesian_pose_distance_from_origin() -> None:
    pose = CartesianPose(x=3.0, y=4.0, z=0.0)
    assert pose.distance_from_origin() == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------


def _make_valid_trajectory() -> JointTrajectory:
    return JointTrajectory(
        name="test_traj",
        waypoints=[
            JointWaypoint(
                positions_rad={"joint_1": 0.0, "joint_2": 0.0},
                time_from_start_s=0.0,
            ),
            JointWaypoint(
                positions_rad={"joint_1": 0.5, "joint_2": 0.3},
                time_from_start_s=1.0,
            ),
            JointWaypoint(
                positions_rad={"joint_1": 1.0, "joint_2": 0.6},
                time_from_start_s=2.0,
            ),
        ],
    )


def test_trajectory_validate_passes_valid() -> None:
    errors = _make_valid_trajectory().validate()
    assert errors == []


def test_trajectory_duration() -> None:
    traj = _make_valid_trajectory()
    assert traj.duration_s() == pytest.approx(2.0)


def test_trajectory_validate_flags_empty_waypoints() -> None:
    traj = JointTrajectory(name="empty", waypoints=[])
    errors = traj.validate()
    assert any("no waypoints" in e for e in errors)


def test_trajectory_validate_flags_non_increasing_time() -> None:
    traj = JointTrajectory(
        name="bad_time",
        waypoints=[
            JointWaypoint(positions_rad={"joint_1": 0.0}, time_from_start_s=1.0),
            JointWaypoint(positions_rad={"joint_1": 0.5}, time_from_start_s=0.5),
        ],
    )
    errors = traj.validate()
    assert any("not strictly increasing" in e for e in errors)


def test_trajectory_validate_flags_unknown_joints() -> None:
    traj = JointTrajectory(
        name="bad_joints",
        waypoints=[
            JointWaypoint(
                positions_rad={"joint_99": 0.0},
                time_from_start_s=0.0,
            )
        ],
    )
    errors = traj.validate()
    assert any("unknown joint" in e for e in errors)


def test_trajectory_execution_record_max_error() -> None:
    traj = _make_valid_trajectory()
    record = TrajectoryExecutionRecord(trajectory=traj)
    record.status = TrajectoryStatus.executing

    cmd1 = JointWaypoint({"joint_1": 0.0, "joint_2": 0.0}, 0.0)
    meas1 = JointWaypoint({"joint_1": 0.02, "joint_2": 0.01}, 0.0)
    cmd2 = JointWaypoint({"joint_1": 0.5, "joint_2": 0.3}, 1.0)
    meas2 = JointWaypoint({"joint_1": 0.53, "joint_2": 0.29}, 1.0)

    record.commanded_waypoints.extend([cmd1, cmd2])
    record.measured_waypoints.extend([meas1, meas2])

    err = record.max_position_error_rad()
    assert err is not None
    assert err == pytest.approx(0.03, abs=1e-5)


def test_trajectory_execution_record_no_pairs_returns_none() -> None:
    traj = _make_valid_trajectory()
    record = TrajectoryExecutionRecord(trajectory=traj)
    assert record.max_position_error_rad() is None


def test_trajectory_to_dict() -> None:
    traj = _make_valid_trajectory()
    d = traj.to_dict()
    assert d["name"] == "test_traj"
    assert d["n_waypoints"] == 3
    assert len(d["waypoints"]) == 3

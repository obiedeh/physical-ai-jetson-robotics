"""Forward and inverse kinematics of the committed chain against a scene sample."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from synria_lerobot.arm_kinematics import KinematicChain
from synria_lerobot.quality_gates import load_limits

REPOSITORY = Path(__file__).resolve().parents[1]
CHAIN = KinematicChain.load(REPOSITORY / "config" / "synria_kinematics.json")
BOUNDS = load_limits(REPOSITORY / "config" / "synria_limits.yaml").joint_limits_rad
# A still sample from the policy-mode scene on 2026-10-18: reported joints and the
# published gripper_center world position.
SAMPLE_JOINTS = (-0.008768702857196331, 1.586155443655098, -0.5822755601384512,
                 0.03433183953166008, -0.7887949347496033, -0.04994433745741844)
SAMPLE_POSITION = (-0.1454150378704071, 0.005528927780687809, 0.107237808406353)


def grasp_frame(cup_xy: tuple[float, float], pitch_deg: float = 54.0):
    """Approach pitched toward the cup along the heading; fingers across it."""
    heading = np.array(cup_xy) / np.linalg.norm(cup_xy)
    pitch = math.radians(pitch_deg)
    approach = np.array([heading[0] * math.sin(pitch), heading[1] * math.sin(pitch),
                         -math.cos(pitch)])
    return approach, np.array([-heading[1], heading[0], 0.0])


def test_forward_kinematics_matches_the_scene_with_the_lane_offsets() -> None:
    """With the simulation offsets the chain reproduces the published gripper centre."""
    position = CHAIN.position(SAMPLE_JOINTS, "simulation")
    assert np.linalg.norm(position - np.array(SAMPLE_POSITION)) < 0.001
    without = CHAIN.position(SAMPLE_JOINTS, "physical")
    assert np.linalg.norm(without - np.array(SAMPLE_POSITION)) > 0.01, "offsets matter"
    assert CHAIN.lanes == ("simulation", "physical")
    with pytest.raises(ValueError, match="offsets"):
        CHAIN.offsets("other")


def test_inverse_kinematics_reaches_a_grasp_pose_inside_the_margin() -> None:
    """A cup at the demonstrations' median distance is solved on the operator's branch."""
    approach, finger = grasp_frame((-0.32, -0.05))
    target = np.array([-0.32, -0.05, 0.03])
    posture = tuple(math.radians(v) for v in (-0.6, -29.2, -3.7, 2.0, 48.2, -7.3))
    solution = CHAIN.solve_ik(target, approach, SAMPLE_JOINTS, "simulation", BOUNDS,
                              finger_axis=finger, posture_rad=posture)
    assert solution is not None
    assert solution.position_error_m < 0.002 and solution.axis_error_rad < math.radians(2)
    reached = CHAIN.position(solution.joints_rad, "simulation")
    assert np.linalg.norm(reached - target) < 0.002
    assert abs(float(np.dot(CHAIN.approach_axis(solution.joints_rad, "simulation"),
                            approach)) - 1.0) < 0.01
    assert abs(CHAIN.finger_axis(solution.joints_rad, "simulation")[2]) < 0.05
    margin = math.radians(3.0)
    for value, (lower, upper) in zip(solution.joints_rad, BOUNDS, strict=True):
        assert lower + margin - 1e-9 <= value <= upper - margin + 1e-9
    assert solution.joints_rad[1] < 0, "the shoulder bends forward as in the takes"


def test_unreachable_target_is_reported_as_none_and_bad_inputs_refused() -> None:
    """Beyond the arm's reach the solver gives None rather than a wrong answer."""
    approach, finger = grasp_frame((-0.9, 0.0))
    assert CHAIN.solve_ik((-0.9, 0.0, 0.03), approach, SAMPLE_JOINTS, "simulation", BOUNDS,
                          finger_axis=finger) is None
    with pytest.raises(ValueError, match="approach axis"):
        CHAIN.solve_ik((0.3, 0.0, 0.1), (0.0, 0.0, 0.0), SAMPLE_JOINTS, "simulation", BOUNDS)
    with pytest.raises(ValueError, match="six operator bounds"):
        CHAIN.solve_ik((0.3, 0.0, 0.1), (0.0, 0.0, -1.0), SAMPLE_JOINTS, "simulation", BOUNDS[:5])
    with pytest.raises(ValueError, match="six finite"):
        CHAIN.forward((0.0,) * 5, "simulation")

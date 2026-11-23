"""Tests for moving teleoperated takes to new cup placements (MimicGen method)."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from synria_lerobot.arm_kinematics import KinematicChain
from synria_lerobot.mimic_generation import (
    MimicReplayExecutor,
    SourceTake,
    choose_source,
    find_grasp_frame,
    object_segment,
    transform_take,
)
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.sim_policy_runner import ObjectPose

REPOSITORY = Path(__file__).resolve().parents[1]
CHAIN = KinematicChain.load(REPOSITORY / "config" / "synria_kinematics.json")
LIMITS = load_limits(REPOSITORY / "config" / "synria_limits.yaml")
LANE = "simulation"
# The paired campaigns' start pose (six joints), a reachable configuration behind the base.
HOME = np.array([-0.00875, 1.59184, -0.58202, 0.03414, -0.78877, -0.04995])


def _gripper_trace(frames: int, close_at: int, stall_m: float | None) -> tuple[np.ndarray,
                                                                               np.ndarray]:
    """Commanded and measured gripper openings: open, then a close at ``close_at``."""
    commanded = np.zeros(frames)
    commanded[close_at:] = 0.02
    measured = commanded.copy() if stall_m is None else np.where(commanded > 0, stall_m, 0.0)
    return commanded, measured


def _arrays(commanded: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((len(commanded), 7))
    actions = np.zeros((len(commanded), 7))
    states[:, 6], actions[:, 6] = measured, commanded
    return states, actions


def test_grasp_is_a_low_close_where_the_fingers_stall():
    commanded, measured = _gripper_trace(40, 10, stall_m=0.004)
    states, actions = _arrays(commanded, measured)
    assert find_grasp_frame(states, actions, np.full(40, 0.03)) == 10


def test_a_close_in_free_air_or_high_up_is_not_a_grasp():
    commanded, measured = _gripper_trace(40, 10, stall_m=None)
    states, actions = _arrays(commanded, measured)
    assert find_grasp_frame(states, actions, np.full(40, 0.03)) is None
    commanded, measured = _gripper_trace(40, 10, stall_m=0.004)
    states, actions = _arrays(commanded, measured)
    assert find_grasp_frame(states, actions, np.full(40, 0.15)) is None


def test_the_last_of_two_grasps_is_used():
    commanded = np.zeros(80)
    commanded[10:30] = 0.02
    commanded[50:] = 0.02
    measured = np.where(commanded > 0, 0.004, 0.0)
    states, actions = _arrays(commanded, measured)
    assert find_grasp_frame(states, actions, np.full(80, 0.03)) == 50


def test_the_segment_runs_from_the_final_approach_to_just_after_the_lift():
    frames = 60
    positions = np.zeros((frames, 3))
    positions[:, 0] = np.where(np.arange(frames) < 20, -0.20, -0.30)  # far, then at the cup
    positions[:, 2] = 0.03
    positions[40:, 2] = 0.03 + 0.01 * np.arange(frames - 40)  # lifting from frame 40
    segment = object_segment(positions, grasp=30, cup_xy=(-0.30, 0.0))
    assert segment == (20, 46 + 5)
    flat = positions.copy()
    flat[:, 2] = 0.03
    assert object_segment(flat, grasp=30, cup_xy=(-0.30, 0.0)) is None


def _source(frames: int = 12) -> SourceTake:
    """A short smooth take near the home pose, its 'grasp' in the middle."""
    actions = np.zeros((frames, 7))
    for index in range(frames):
        actions[index, :6] = HOME + np.array([0.0, -0.01, 0.01, 0.0, 0.005, 0.0]) * index
    actions[frames // 2:, 6] = 0.02
    grasp = frames // 2
    cup = CHAIN.position(actions[grasp, :6], LANE)
    return SourceTake(7, actions, grasp, (float(cup[0]), float(cup[1])), (0, frames), frames)


def test_moving_to_the_same_cup_reproduces_the_take():
    take = _source()
    result = transform_take(take, take.cup_xy, CHAIN, LANE, LIMITS.joint_limits_rad,
                            margin_rad=math.radians(0.5))
    assert result.commands is not None
    for moved, original in zip(result.commands, take.actions, strict=True):
        error = np.linalg.norm(CHAIN.position(moved[:6], LANE)
                               - CHAIN.position(original[:6], LANE))
        assert error < 0.003
    assert np.array_equal(result.commands[:, 6], take.actions[:, 6])


def test_moving_to_a_new_bearing_and_radius_puts_the_grasp_on_the_new_cup():
    take = _source()
    bearing = take.bearing_rad + math.radians(8.0)
    radius = take.radius_m + 0.01
    cup = (radius * math.cos(bearing), radius * math.sin(bearing))
    result = transform_take(take, cup, CHAIN, LANE, LIMITS.joint_limits_rad,
                            margin_rad=math.radians(0.5))
    assert result.commands is not None
    assert math.isclose(math.degrees(result.delta_bearing_rad), 8.0, abs_tol=1e-6)
    assert math.isclose(result.radial_shift_m, 0.01, abs_tol=1e-9)
    grasp = CHAIN.position(result.commands[take.grasp_frame, :6], LANE)
    assert math.hypot(grasp[0] - cup[0], grasp[1] - cup[1]) < 0.003


def test_the_source_is_chosen_among_the_nearest_radii():
    near, far = _source(), _source()
    far = SourceTake(9, far.actions, far.grasp_frame, (far.cup_xy[0] * 2, far.cup_xy[1] * 2),
                     far.source_frames, far.lifted_frame)
    picked = {choose_source([near, far], near.cup_xy, random.Random(seed), nearest=1)
              .episode_index for seed in range(10)}
    assert picked == {7}


def test_the_executor_walks_to_the_first_frame_then_replays_and_holds():
    take = _source(6)
    pose = ObjectPose((*take.cup_xy, 0.0), 0.0)
    executor = MimicReplayExecutor([take], lambda: pose, CHAIN, LANE, LIMITS.joint_limits_rad,
                                   random.Random(0), margin_rad=math.radians(0.5),
                                   max_step_rad=0.05, max_gripper_step_m=0.004)
    assert executor.record.feasible and executor.phase == "walk"
    state = (*(HOME - 0.1), 0.0)
    phases = []
    for _ in range(100):
        command, phase = executor.step(state)
        assert np.all(np.abs(np.asarray(command[:6]) - np.asarray(state[:6])) <= 0.05 + 1e-9)
        state = command
        phases.append(phase)
        if phase == "done":
            break
    assert phases[-1] == "done" and "replay" in phases
    assert executor.record.replayed_frames == 6
    assert executor.step(state) == (state, "done")


def test_an_unreachable_cup_leaves_the_executor_infeasible_and_still():
    take = _source(6)
    pose = ObjectPose((-2.0, 0.0, 0.0), 0.0)
    executor = MimicReplayExecutor([take], lambda: pose, CHAIN, LANE, LIMITS.joint_limits_rad,
                                   random.Random(0), margin_rad=math.radians(0.5),
                                   max_step_rad=0.05, max_gripper_step_m=0.004)
    state = (*HOME, 0.0)
    assert executor.phase == "infeasible" and not executor.record.feasible
    assert executor.step(state) == (tuple(float(v) for v in state), "infeasible")

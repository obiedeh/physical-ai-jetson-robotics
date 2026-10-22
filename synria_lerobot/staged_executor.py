"""A staged what-where-how-plan-execute executor, served on the policy protocol.

The executor answers four questions before it moves and writes every answer into
a stage record: what (the task and the object from the registry), where (the
object's and the target's world positions from the scene's published poses today,
from a camera locator later), how (a grasp strategy per object class, derived from
the operator's own demonstrations), and plan (joint-space waypoints from inverse
kinematics inside the operator limits with a margin). Only then does it execute:
approach to a pre-grasp pose, descend along the approach axis, squeeze, lift, and
for a place task carry, lower, release and retreat. A plan that cannot be solved
is reported as infeasible and the arm holds; nothing is guessed.

It is served through the same HTTP protocol as a learned policy, so the trial
runner, the safety client, the gripper guard, the limits and the scene scorer
apply to it unchanged, and its campaigns are comparable with a learned policy's
on the same seeded placements. It is a scripted expert and a demonstration
engine, not a learned policy, and it says so in its metadata. Simulation lane
only; it never publishes on a real-arm topic and the physical lane refuses it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synria_lerobot.arm_kinematics import KinematicChain
from synria_lerobot.physical_contract import (
    IMAGE_KEYS,
    SIMULATION_LANE,
    PhysicalDatasetContract,
)
from synria_lerobot.policy_codec import decode_policy_request, encode_policy_response
from synria_lerobot.quality_gates import PhysicalLimits
from synria_lerobot.sim_policy_runner import LIFT_SCORE, PLACE_SCORE, SCORE_BY_TASK, ObjectPose

EXECUTOR_KIND = "staged_expert"
EXECUTOR_VERSION = "staged_expert_v2"
PHASES = ("plan", "approach", "look", "descend", "check", "correct", "squeeze", "lift", "reopen",
          "backoff", "secure", "carry", "lower", "release", "retreat", "rest", "done",
          "infeasible")
# Reached when every joint is this close to its waypoint; the sim arm sags about half
# a degree under load, so the tolerance must be wider than that.
REACHED_RAD = math.radians(1.5)
# A phase that has not reached its waypoint after this many steps moves on anyway,
# so a sagging joint cannot stall a trial; the stall is recorded.
PHASE_STEP_BUDGET = 60
CARTESIAN_STEP_M = 0.01
# The cup counts as seated between the jaws when the gripper centre (FK on the observed
# joints) is within SEAT_SIDEWAYS_M of the cup's axis across the jaws, has been pushed
# at least SEAT_MIN_DEPTH_M past the axis, and sits inside the height band above the
# cup's base. The strategy aims deeper; in the scene the arm stops 11 to 14 mm past the
# axis with the cup's widest section well inside the fingers, the second pilot's
# measure, so the seat accepts 8 mm and up. The scene's cup is 41.3 mm across and the open jaws
# 51 mm apart, about 5 mm of play a side, and the fingers run 70 mm back from their
# tips; the traced corpus policy closed 17 to 34 mm off the axis, against the cup's
# wall, and lifted only from 11 to 14 mm.
SEAT_SIDEWAYS_M = 0.006
SEAT_MIN_DEPTH_M = 0.008
# A cup that slides ahead of the push stops getting deeper: once a correction gains less
# than SEAT_PLATEAU_GAIN_M with the cup centred, in the height band and at least
# SEAT_PLATEAU_DEPTH_M past its axis, it is as far in as it will go and counts as seated.
# The fourth pilot's one failure held 6.5 to 7.9 mm through three corrections.
SEAT_PLATEAU_DEPTH_M = 0.005
SEAT_PLATEAU_GAIN_M = 0.001
SEAT_HEIGHT_M = (0.015, 0.05)
# The cup counts as shoved by a fingertip once it has moved this far sideways (across
# the heading from the base) from where the current descent was planned; the push-in
# moves it along the heading, and the 150 takes of staged_place_02 moved it sideways
# 1.5 mm at the median and 4.5 mm at p90 before the lift.
CONTACT_MOVE_M = 0.004
# The open jaws (51 mm) leave about 5 mm each side of the 41.3 mm cup; a gripper
# centre within twice that is around the cup, beyond it the leading fingertip is on
# the cup's wall. Only then does a shove mean backing out.
FINGERTIP_ACROSS_M = 0.010
# The scene cup's height is 69.9 mm; the wrist view aims at its middle.
CUP_MID_HEIGHT_M = 0.035
# Steps the arm dwells with the gripper open before judging the seat (0.4 s at 15 fps),
# so a demonstration shows the cup settling into the open jaws before the close.
CHECK_STEPS = 6
# The wrist camera in the tool frame (the gripper centre, axes as Link6): mounted on the
# back of the gripper body and tilted 38 degrees from the approach axis toward +X, with
# an 88 degree horizontal field (scripts/isaac_sim/teleop_cameras.py in the teleop
# repository). The cup counts as found when its mid-height point lies within
# LOOK_CONE_DEG of the optical axis, inside the 44 degree half field. The third pilot
# measured 23 to 27 degrees at a reached pre-grasp and 36 to 50 from a stalled approach
# out of a turned start, so 35 passes the first and sends the second round again.
WRIST_CAMERA_TOOL_M = (-0.0452, 0.0, -0.05598)
WRIST_CAMERA_TILT_DEG = 38.0
LOOK_CONE_DEG = 35.0
# Re-approaches allowed when the cup is not in the wrist view at the pre-grasp.
MAX_LOOKS = 3
# The squeeze holds the arm still until the measured gripper width has moved less than
# GRIP_SETTLE_M a step for GRIP_SETTLE_STEPS steps running, the fingers pressed on the
# cup, then lifts; GRIP_BUDGET_STEPS caps the wait and the lift check catches a slip.
# A fixed count started the lift while the fingers were still closing in the fourth
# pilot, a grip that looked lazy and let go.
GRIP_SETTLE_M = 0.0003
GRIP_SETTLE_STEPS = 5
GRIP_BUDGET_STEPS = 45
# Once the gripper centre has risen this far above the grasp, the cup must have risen at
# least LIFT_CONFIRM_M off its rest, or the grasp missed and the executor re-grasps.
LIFT_CHECK_M = 0.03
LIFT_CONFIRM_M = 0.01
# After the lift the gripper holds still this many steps and confirms the cup rose with
# it before the carry, so a demonstration marks the moment the cup is secured.
SECURE_STEPS = 6
# At the rest pose after the retreat the arm holds still this many steps, so a
# demonstration ends sitting back empty instead of mid-motion.
REST_HOLD_STEPS = 30
# When a group of waypoints does not solve at the strategy's pitch, steeper approaches
# are tried in this order; a mark close to the base needs the gripper more upright.
FALLBACK_PITCHES_RAD = (math.radians(40.0), math.radians(25.0), math.radians(10.0))


@dataclass(frozen=True)
class GraspStrategy:
    """How one object class is grasped: heights, approach, squeeze and the posture branch.

    ``centre_height_m`` is the gripper centre (the fingertip plane) above the object's
    base at the squeeze; ``pitch_from_vertical_rad`` tilts the approach axis from
    straight down toward the object along the heading from the base; the fingers open
    horizontally across that heading. ``posture_rad`` is the joint configuration the
    inverse kinematics is pulled toward, which selects the arm's branch.
    """

    name: str
    object_name: str
    centre_height_m: float
    pitch_from_vertical_rad: float
    pregrasp_back_m: float
    closed_m: float
    open_m: float
    lift_m: float
    squeeze_steps: int
    release_steps: int
    posture_rad: tuple[float, ...]
    source: str
    # How far past the object's axis the gripper centre is pushed, horizontally along the
    # heading at the squeeze height, so the object sits deep between the fingers rather
    # than at the fingertips. 0 grasps with the axis on the fingertip plane.
    push_in_m: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a strategy with nonsense geometry."""
        values = (self.centre_height_m, self.pitch_from_vertical_rad, self.pregrasp_back_m,
                  self.closed_m, self.open_m, self.lift_m)
        if (any(not math.isfinite(v) or v < 0 for v in (*values, self.push_in_m))
                or self.lift_m == 0):
            raise ValueError("grasp strategy distances must be finite and non-negative")
        if not 0 <= self.pitch_from_vertical_rad < math.pi / 2:
            raise ValueError("the approach pitch must lie between straight down and horizontal")
        if self.squeeze_steps < 1 or self.release_steps < 1 or len(self.posture_rad) != 6:
            raise ValueError("squeeze and release steps must be positive; posture needs six joints")


# Medians of the 49 pick_cup takes of 2026-10-18 at the leader's final squeeze, from
# forward kinematics on the recorded follower state: gripper centre 29 mm above the
# floor, approach pitched 54 degrees from vertical, fingers within 5 degrees of
# horizontal, lift 78 mm by the end of the take, joints (-0.6, -29.2, -3.7, 2.0, 48.2,
# -7.3) degrees. The squeeze command is where the leader closes on the cup.
CUP_SIDE_PITCHED = GraspStrategy(
    name="cup_side_pitched", object_name="cup", centre_height_m=0.03,
    pitch_from_vertical_rad=math.radians(54.0), pregrasp_back_m=0.08, closed_m=0.022,
    open_m=0.0, lift_m=0.08, squeeze_steps=10, release_steps=4,
    posture_rad=tuple(math.radians(v) for v in (-0.6, -29.2, -3.7, 2.0, 48.2, -7.3)),
    source="medians of the 49 pick_cup takes of 2026-10-18 at the leader's final squeeze",
    push_in_m=0.02,
)
STRATEGIES = {CUP_SIDE_PITCHED.object_name: CUP_SIDE_PITCHED}
# Objects the registry tasks act on; the place target is the scene's mark.
OBJECT_BY_TASK = {"pick_cup": "cup", "cup_pick_and_place": "cup"}
# The cup is released this far above its rest height so it drops, not drags.
PLACE_CLEARANCE_M = 0.005


@dataclass
class Waypoint:
    """One planned joint configuration with the inverse-kinematics residuals."""

    name: str
    joints_rad: tuple[float, ...]
    position_error_m: float
    axis_error_rad: float


@dataclass
class StageRecord:
    """Every answer the executor gave for one trial, in the order it gave them."""

    what: dict[str, Any] = field(default_factory=dict)
    where: dict[str, Any] = field(default_factory=dict)
    how: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    feasible: bool | None = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    stalls: list[dict[str, Any]] = field(default_factory=list)
    final_phase: str = "plan"
    steps: int = 0


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


class StagedExecutor:
    """One trial's state machine: answers, plan, then phase by phase to the score."""

    def __init__(
        self,
        chain: KinematicChain,
        lane: str,
        limits: PhysicalLimits,
        gripper_type: str,
        strategy: GraspStrategy,
        score: str,
        object_pose: Callable[[], ObjectPose],
        target_pose: Callable[[], ObjectPose] | None,
        *,
        max_step_rad: float,
        max_gripper_step_m: float,
        margin_rad: float,
        clock: Callable[[], float] = time.monotonic,
        arrival_offset_m: tuple[float, float] = (0.0, 0.0),
        max_corrections: int = 3,
        max_regrasps: int = 2,
        max_recentres: int = 2,
        rest_joints_rad: tuple[float, ...] | None = None,
    ) -> None:
        """Bind the scene's pose readers and the motion limits; nothing moves until step().

        ``arrival_offset_m`` (back along the approach axis, sideways along the finger
        axis) ends the first descent short of the grasp on purpose, so a demonstration
        shows the open gripper arriving off the cup, the seat check failing, and the
        correction that seats the cup before the close. Every descent ends in the seat
        check; ``max_corrections`` bounds the corrections per descent and
        ``max_regrasps`` the reopen-and-retry cycles after a lift that left the cup.
        A descent or a correction that shoves the cup sideways while the gripper is
        off the cup's centre (a fingertip met the wall first), or a seat check that
        finds it so, backs out to the pre-grasp and comes
        in again on the cup's fresh pose instead of sliding sideways with the cup
        between the jaws;
        ``max_recentres`` bounds those. ``rest_joints_rad`` (six joints), when given,
        ends a place take back at that pose with the gripper open, held still.
        """
        if score not in (LIFT_SCORE, PLACE_SCORE):
            raise ValueError("the executor runs the lift or the place score")
        if score == PLACE_SCORE and target_pose is None:
            raise ValueError("the place score needs the target's pose")
        if not all(math.isfinite(v) and v > 0 for v in (max_step_rad, max_gripper_step_m)):
            raise ValueError("step limits must be positive and finite")
        if gripper_type not in limits.gripper_limits_m:
            raise ValueError(f"unknown gripper type {gripper_type!r}")
        back, side = (float(v) for v in arrival_offset_m)
        if not (math.isfinite(back) and math.isfinite(side) and 0 <= back <= 0.05
                and abs(side) <= 0.02):
            raise ValueError("the arrival offset must lie within 5 cm back and 2 cm sideways")
        if max_corrections < 1 or max_regrasps < 0 or max_recentres < 0:
            raise ValueError("corrections must be positive; regrasps and recentres "
                             "non-negative")
        if rest_joints_rad is not None and (
                len(rest_joints_rad) != 6
                or any(not math.isfinite(v) for v in rest_joints_rad)):
            raise ValueError("the rest pose needs six finite joints")
        self.chain, self.lane, self.limits, self.strategy = chain, lane, limits, strategy
        self.score, self.object_pose, self.target_pose = score, object_pose, target_pose
        self.max_step_rad, self.max_gripper_step_m = max_step_rad, max_gripper_step_m
        self.margin_rad, self.clock = margin_rad, clock
        self.bounds = limits.joint_limits_rad
        self.gripper_bounds = limits.gripper_limits_m[gripper_type]
        self.record = StageRecord()
        self.phase = "plan"
        self._phase_steps = 0
        self._waypoints: dict[str, Waypoint] = {}
        self._path: list[Waypoint] = []
        self._path_index = 0
        self._gripper_command = strategy.open_m
        self._hold: tuple[float, ...] | None = None
        self.arrival_offset_m = (back, side)
        self.max_corrections, self.max_regrasps = max_corrections, max_regrasps
        self.max_recentres = max_recentres
        self.rest_joints_rad = (None if rest_joints_rad is None
                                else tuple(float(v) for v in rest_joints_rad))
        self._corrections = 0
        self._regrasps = 0
        self._recentres = 0
        self._looks = 0
        self._last_depth: float | None = None
        self._last_width: float | None = None
        # The arrival offset applies to the first descent only; a re-plan before it
        # (a failed look) keeps it, one after it (a correction, a regrasp) drops it.
        self._descended = False
        self._still_steps = 0
        self._pick_pitch: float | None = None
        self._cup_rest_z: float | None = None
        self._grasp_z: float | None = None
        self._planned_cup_xy: np.ndarray | None = None

    # Answers ---------------------------------------------------------------------

    def _frame(self, position: np.ndarray,
               pitch: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The approach and finger axes for an object at ``position``: pitched toward it."""
        heading = position[:2].copy()
        if float(np.linalg.norm(heading)) < 0.05:
            raise ValueError("the object sits over the base; no heading")
        heading = _unit(heading)
        pitch = self.strategy.pitch_from_vertical_rad if pitch is None else pitch
        approach = np.array([heading[0] * math.sin(pitch), heading[1] * math.sin(pitch),
                             -math.cos(pitch)])
        finger = np.array([-heading[1], heading[0], 0.0])
        return approach, finger

    def _solve(self, name: str, position: np.ndarray, approach: np.ndarray,
               finger: np.ndarray, seed: tuple[float, ...]) -> Waypoint | None:
        solution = self.chain.solve_ik(
            position, approach, seed, self.lane, self.bounds, finger_axis=finger,
            posture_rad=self.strategy.posture_rad, margin_rad=self.margin_rad,
        )
        if solution is None:
            self.record.plan[name] = {"feasible": False, "position_m": position.round(4).tolist()}
            return None
        waypoint = Waypoint(name, solution.joints_rad, solution.position_error_m,
                            solution.axis_error_rad)
        self.record.plan[name] = {
            "feasible": True, "position_m": position.round(4).tolist(),
            "joints_deg": [round(math.degrees(v), 2) for v in solution.joints_rad],
            "position_error_mm": round(solution.position_error_m * 1000, 2),
            "axis_error_deg": round(math.degrees(solution.axis_error_rad), 2),
        }
        return waypoint

    def _plan(self, state: tuple[float, ...]) -> bool:
        """Answer what, where and how, then solve every waypoint; False when any fails."""
        strategy = self.strategy
        self.record.what = {"score": self.score, "object": strategy.object_name,
                            "goal": "lift" if self.score == LIFT_SCORE else "the scene's mark"}
        obj = self.object_pose()
        objective = np.asarray(obj.position_m, dtype=float)
        self.record.where = {"object_m": objective.round(4).tolist(),
                             "object_age_s": round(self.clock() - obj.monotonic_timestamp_s, 3),
                             "source": "scene published pose"}
        target = None
        if self.target_pose is not None:
            mark = self.target_pose()
            target = np.asarray(mark.position_m, dtype=float)
            self.record.where["target_m"] = target.round(4).tolist()
        try:
            approach, finger = self._frame(objective)
        except ValueError as error:
            self.record.how = {"strategy": strategy.name, "error": str(error)}
            return False
        self.record.how = {
            "strategy": strategy.name, "source": strategy.source,
            "approach_axis": approach.round(3).tolist(), "finger_axis": finger.round(3).tolist(),
            "centre_height_m": strategy.centre_height_m,
            "pitch_from_vertical_deg": round(math.degrees(strategy.pitch_from_vertical_rad), 1),
            "closed_m": strategy.closed_m, "lift_m": strategy.lift_m,
        }
        self.record.how["arrival_offset_m"] = list(self.arrival_offset_m)
        self._cup_rest_z = float(objective[2])
        seed = self._solve_pick(objective, tuple(state[:6]), with_arrival=True)
        if seed is None:
            return False
        if target is not None:
            place = target + np.array([0.0, 0.0, strategy.centre_height_m + PLACE_CLEARANCE_M])
            above = place + np.array([0.0, 0.0, strategy.lift_m])
            if self._solve_group(
                "place", target, seed,
                lambda axis: [("carry", above), ("place", place), ("retreat", above)],
            ) is None:
                return False
        return True

    def _solve_pick(self, objective: np.ndarray, seed: tuple[float, ...], *,
                    with_arrival: bool) -> tuple[float, ...] | None:
        """Solve pregrasp, grasp and lift for a cup at ``objective``; the arrival when offset.

        The first plan tries every pitch; a re-plan after a correction or a missed
        grasp keeps the pitch the first plan chose, so the approach does not swing.
        """
        strategy = self.strategy
        heading = np.array([objective[0], objective[1], 0.0])
        heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
        finger = np.array([-heading[1], heading[0], 0.0])
        grasp = (objective + np.array([0.0, 0.0, strategy.centre_height_m])
                 + strategy.push_in_m * heading)
        lifted = grasp + np.array([0.0, 0.0, strategy.lift_m])
        back, side = self.arrival_offset_m
        offset = with_arrival and (back > 0 or side != 0)

        def waypoints(axis: np.ndarray) -> list[tuple[str, np.ndarray]]:
            named = [("pregrasp", grasp - strategy.pregrasp_back_m * axis)]
            if offset:
                named.append(("arrival", grasp - back * axis + side * finger))
            return [*named, ("grasp", grasp), ("lift", lifted)]

        pitches = None if self._pick_pitch is None else (self._pick_pitch,)
        result = self._solve_group("pick", objective, seed, waypoints, pitches=pitches)
        if result is not None:
            self._grasp_z = float(grasp[2])
            self._planned_cup_xy = np.asarray(objective[:2], dtype=float).copy()
            if not offset:
                self._waypoints.pop("arrival", None)
        return result

    def _seated(self, state: tuple[float, ...]) -> tuple[bool, dict[str, float]]:
        """Whether the cup sits centred and pushed all the way in, on the cup's fresh pose.

        The gripper centre's offset from the cup's axis is split along the heading from
        the base (depth: how far past the axis the jaws reach) and across it (between
        the fingers); the squeeze waits until both and the height are inside the seat.
        """
        centre = self.chain.position(state[:6], self.lane)
        cup = np.asarray(self.object_pose().position_m, dtype=float)
        heading = np.array([cup[0], cup[1]]) / max(float(np.hypot(cup[0], cup[1])), 1e-9)
        offset = np.array([centre[0] - cup[0], centre[1] - cup[1]])
        depth = float(offset @ heading)
        across = float(offset @ np.array([-heading[1], heading[0]]))
        height = float(centre[2] - cup[2])
        low, high = SEAT_HEIGHT_M
        seated = (abs(across) <= SEAT_SIDEWAYS_M
                  and depth >= SEAT_MIN_DEPTH_M
                  and low <= height <= high)
        return seated, {"depth_mm": round(depth * 1000, 1), "across_mm": round(across * 1000, 1),
                        "height_mm": round(height * 1000, 1)}

    def _here(self, state: tuple[float, ...]) -> None:
        """The observed joints as the ``here`` waypoint, a start for a re-planned line."""
        self._waypoints["here"] = Waypoint("here", tuple(state[:6]), 0.0, 0.0)

    def _solve_group(
        self, group: str, anchor: np.ndarray, seed: tuple[float, ...],
        waypoints: Callable[[np.ndarray], list[tuple[str, np.ndarray]]],
        *, pitches: tuple[float, ...] | None = None,
    ) -> tuple[float, ...] | None:
        """Solve one group of waypoints at the strategy's pitch, then steeper fallbacks.

        The group's pitch is recorded under ``how``; the seed of the last waypoint is
        returned so the next group starts from it, or None when no pitch solves.
        """
        if pitches is None:
            pitches = (self.strategy.pitch_from_vertical_rad, *FALLBACK_PITCHES_RAD)
        for pitch in pitches:
            try:
                approach, finger = self._frame(anchor, pitch)
            except ValueError as error:
                self.record.how[f"{group}_error"] = str(error)
                return None
            solved: dict[str, Waypoint] = {}
            last = seed
            for name, position in waypoints(approach):
                waypoint = self._solve(name, position, approach, finger, last)
                if waypoint is None:
                    break
                solved[name] = waypoint
                last = waypoint.joints_rad
            else:
                self._waypoints.update(solved)
                if group == "pick":
                    self._pick_pitch = pitch
                self.record.how[f"{group}_pitch_deg"] = round(math.degrees(pitch), 1)
                self.record.how[f"{group}_approach_axis"] = approach.round(3).tolist()
                return last
        return None

    def _line(self, start: str, end: str, axis_name: str) -> list[Waypoint] | None:
        """Waypoints along the straight line between two planned poses, 1 cm apart."""
        a, b = self._waypoints[start], self._waypoints[end]
        p0 = self.chain.position(a.joints_rad, self.lane)
        p1 = self.chain.position(b.joints_rad, self.lane)
        approach = self.chain.approach_axis(a.joints_rad, self.lane)
        finger = self.chain.finger_axis(a.joints_rad, self.lane)
        count = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / CARTESIAN_STEP_M)))
        path: list[Waypoint] = []
        seed = a.joints_rad
        for index in range(1, count + 1):
            position = p0 + (p1 - p0) * (index / count)
            waypoint = self._solve(f"{axis_name}_{index}", position, approach, finger, seed)
            if waypoint is None:
                return None
            path.append(waypoint)
            seed = waypoint.joints_rad
        path[-1] = b
        return path

    # Execution -------------------------------------------------------------------

    def _enter(self, phase: str, state: tuple[float, ...]) -> None:
        self.record.phases.append({"phase": phase, "step": self.record.steps,
                                   "t_s": round(self.clock(), 3),
                                   "gripper_centre_m": self.chain.position(
                                       state[:6], self.lane).round(4).tolist()})
        self.phase, self._phase_steps, self._path_index = phase, 0, 0
        self.record.final_phase = phase

    def _toward(self, current: tuple[float, ...], target: tuple[float, ...]) -> tuple[float, ...]:
        """One step from the current joints toward the target, all joints arriving together."""
        delta = np.asarray(target[:6]) - np.asarray(current[:6])
        scale = max(1.0, float(np.max(np.abs(delta))) / self.max_step_rad)
        return tuple(float(v) for v in np.asarray(current[:6]) + delta / scale)

    def _reached(self, state: tuple[float, ...], target: tuple[float, ...]) -> bool:
        return bool(np.max(np.abs(np.asarray(state[:6]) - np.asarray(target[:6]))) <= REACHED_RAD)

    def _advance_or_stall(self, state: tuple[float, ...], target: tuple[float, ...]) -> bool:
        """True when the waypoint is reached or the phase budget is spent (recorded)."""
        if self._reached(state, target):
            return True
        if self._phase_steps >= PHASE_STEP_BUDGET:
            self.record.stalls.append({
                "phase": self.phase, "step": self.record.steps,
                "worst_joint_error_deg": round(math.degrees(float(np.max(np.abs(
                    np.asarray(state[:6]) - np.asarray(target[:6]))))), 2),
            })
            return True
        return False

    def _walk_path(self, state: tuple[float, ...], next_phase: str) -> tuple[float, ...]:
        """Move through the current Cartesian path; enter ``next_phase`` at its end."""
        target = self._path[self._path_index].joints_rad
        if self._advance_or_stall(state, target):
            self._phase_steps = 0
            if self._path_index + 1 < len(self._path):
                self._path_index += 1
                target = self._path[self._path_index].joints_rad
            else:
                self._enter(next_phase, state)
                return target
        return self._toward(state, target)

    def step(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """One command for the observed seven-value state; returns (action, phase)."""
        if len(state) < 7 or any(not math.isfinite(v) for v in state[:7]):
            raise ValueError("the executor needs seven finite state values")
        state = tuple(float(v) for v in state[:7])
        self.record.steps += 1
        self._phase_steps += 1
        strategy = self.strategy
        joints: tuple[float, ...]
        if self.phase == "plan":
            feasible = self._plan(state)
            self.record.feasible = feasible
            if feasible:
                self._enter("approach", state)
            else:
                self._enter("infeasible", state)
                self._hold = state
        if self.phase == "infeasible" or self.phase == "done":
            held = self._hold if self._hold is not None else state
            return (*held[:6], self._gripper_command), self.phase
        if self.phase == "approach":
            target = self._waypoints["pregrasp"].joints_rad
            self._gripper_command = strategy.open_m
            if self._advance_or_stall(state, target):
                self._enter("look", state)
                return (*state[:6], self._gripper_command), self.phase
            return (*self._toward(state, target), self._gripper_command), self.phase
        if self.phase == "look":
            return self._look(state)
        if self.phase in ("descend", "correct"):
            self._gripper_command = strategy.open_m
            if self._fingertip_on_wall(state):
                return self._recentre(state)
            joints = self._walk_path(state, "check")
            return (*joints, self._gripper_command), self.phase
        if self.phase == "check":
            return self._check(state)
        if self.phase == "squeeze":
            self._gripper_command = strategy.closed_m
            joints = self._waypoints["here"].joints_rad
            width = state[6]
            moved = (self._last_width is None
                     or abs(width - self._last_width) >= GRIP_SETTLE_M)
            self._still_steps = 0 if moved else self._still_steps + 1
            self._last_width = width
            firm = self._still_steps >= GRIP_SETTLE_STEPS
            if self._phase_steps >= strategy.squeeze_steps and (
                    firm or self._phase_steps >= GRIP_BUDGET_STEPS):
                self.record.phases[-1]["grip"] = {
                    "firm": firm, "steps": self._phase_steps,
                    "width_mm": round(width * 1000, 2)}
                path = self._line("here", "lift", "lift")
                if path is None:
                    self._enter("infeasible", state)
                    self._hold = state
                    return (*state[:6], self._gripper_command), self.phase
                self._path = path
                self._enter("lift", state)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "lift":
            if self._missed(state):
                return self._begin_regrasp(state)
            next_phase = "done" if self.score == LIFT_SCORE else "secure"
            joints = self._walk_path(state, next_phase)
            if self.phase == "done":
                self._hold = (*self._waypoints["lift"].joints_rad, strategy.closed_m)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "reopen":
            self._gripper_command = strategy.open_m
            joints = self._waypoints["here"].joints_rad
            if self._phase_steps >= strategy.release_steps:
                return self._backoff(state)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "backoff":
            self._gripper_command = strategy.open_m
            joints = self._walk_path(state, "approach")
            return (*joints, self._gripper_command), self.phase
        if self.phase == "secure":
            joints = self._waypoints["lift"].joints_rad
            if self._phase_steps <= SECURE_STEPS:
                return (*joints, self._gripper_command), self.phase
            if self._missed(state):
                return self._begin_regrasp(state)
            self.record.phases[-1]["secured"] = {
                "cup_rise_mm": round(1000 * (float(self.object_pose().position_m[2])
                                             - float(self._cup_rest_z or 0.0)), 1)}
            self._enter("carry", state)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "carry":
            target = self._waypoints["carry"].joints_rad
            if self._advance_or_stall(state, target):
                path = self._line("carry", "place", "lower")
                if path is None:
                    self._enter("infeasible", state)
                    self._hold = state
                    return (*state[:6], self._gripper_command), self.phase
                self._path = path
                self._enter("lower", state)
                joints = self._toward(state, path[0].joints_rad)
            else:
                joints = self._toward(state, target)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "lower":
            joints = self._walk_path(state, "release")
            return (*joints, self._gripper_command), self.phase
        if self.phase == "release":
            self._gripper_command = strategy.open_m
            joints = self._waypoints["place"].joints_rad
            if self._phase_steps >= strategy.release_steps:
                path = self._line("place", "retreat", "retreat")
                if path is None:
                    self._enter("done", state)
                    self._hold = (*joints, strategy.open_m)
                    return (*joints, self._gripper_command), self.phase
                self._path = path
                self._enter("retreat", state)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "retreat":
            next_phase = "done" if self.rest_joints_rad is None else "rest"
            joints = self._walk_path(state, next_phase)
            if self.phase == "rest":
                self._still_steps = 0
            if self.phase == "done":
                self._hold = (*self._waypoints["retreat"].joints_rad, strategy.open_m)
            return (*joints, self._gripper_command), self.phase
        if self.phase == "rest":
            assert self.rest_joints_rad is not None
            target = self.rest_joints_rad
            if self._reached(state, target) or self._phase_steps >= PHASE_STEP_BUDGET:
                self._still_steps += 1
            if self._still_steps >= REST_HOLD_STEPS:
                self._enter("done", state)
                self._hold = (*target, strategy.open_m)
            return (*self._toward(state, target), self._gripper_command), self.phase
        raise RuntimeError(f"unknown phase {self.phase}")

    def _give_up(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        self._enter("infeasible", state)
        self._hold = state
        return (*state[:6], self._gripper_command), self.phase

    def cup_view_angle_deg(self, state: tuple[float, ...]) -> float:
        """Angle between the wrist camera's optical axis and the cup's mid-height point."""
        frame = self.chain.forward(state[:6], self.lane)
        rotation, origin = frame[:3, :3], frame[:3, 3]
        camera = origin + rotation @ np.asarray(WRIST_CAMERA_TOOL_M)
        tilt = math.radians(WRIST_CAMERA_TILT_DEG)
        optical = rotation @ np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        cup = np.asarray(self.object_pose().position_m, dtype=float)
        sight = cup + np.array([0.0, 0.0, CUP_MID_HEIGHT_M]) - camera
        cosine = float(optical @ sight) / max(float(np.linalg.norm(sight)), 1e-9)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _look(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """Dwell open at the pre-grasp; descend once the cup is in the wrist view.

        When it is not, the cup is re-located from its fresh pose and the arm
        re-approaches, so a demonstration never moves on without the cup in view.
        """
        self._gripper_command = self.strategy.open_m
        if self._phase_steps <= CHECK_STEPS:
            return (*state[:6], self._gripper_command), self.phase
        angle = self.cup_view_angle_deg(state)
        found = angle <= LOOK_CONE_DEG
        self.record.phases[-1]["view"] = {"found": found, "angle_deg": round(angle, 1)}
        if not found:
            if self._looks >= MAX_LOOKS:
                return self._give_up(state)
            self._looks += 1
            objective = np.asarray(self.object_pose().position_m, dtype=float)
            if self._solve_pick(objective, tuple(state[:6]),
                                with_arrival=not self._descended) is None:
                return self._give_up(state)
            self._enter("approach", state)
            return (*state[:6], self._gripper_command), self.phase
        self._here(state)
        end = "arrival" if "arrival" in self._waypoints else "grasp"
        path = self._line("here", end, "descend")
        if path is None:
            return self._give_up(state)
        self._path = path
        self._descended = True
        self._enter("descend", state)
        return (*self._toward(state, path[0].joints_rad), self._gripper_command), self.phase

    def _check(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """Dwell open, then squeeze when seated or correct toward the cup's fresh pose."""
        strategy = self.strategy
        self._gripper_command = strategy.open_m
        if self._phase_steps <= CHECK_STEPS:
            return (*state[:6], self._gripper_command), self.phase
        seated, measure = self._seated(state)
        depth = measure["depth_mm"] / 1000
        plateau = (not seated and self._last_depth is not None
                   and depth >= SEAT_PLATEAU_DEPTH_M
                   and depth - self._last_depth < SEAT_PLATEAU_GAIN_M
                   and abs(measure["across_mm"]) <= SEAT_SIDEWAYS_M * 1000
                   and SEAT_HEIGHT_M[0] <= measure["height_mm"] / 1000 <= SEAT_HEIGHT_M[1])
        seated = seated or plateau
        self._last_depth = depth
        self.record.phases[-1]["seat"] = {"seated": seated, "plateau": plateau, **measure}
        self._here(state)
        if seated:
            self._last_width, self._still_steps = None, 0
            self._enter("squeeze", state)
            self._gripper_command = strategy.closed_m
            return (*state[:6], self._gripper_command), self.phase
        if self._fingertip_on_wall(state):
            return self._recentre(state)
        if self._corrections >= self.max_corrections:
            return self._give_up(state)
        self._corrections += 1
        objective = np.asarray(self.object_pose().position_m, dtype=float)
        if self._solve_pick(objective, tuple(state[:6]), with_arrival=False) is None:
            return self._give_up(state)
        path = self._line("here", "grasp", "correct")
        if path is None:
            return self._give_up(state)
        self._path = path
        self._enter("correct", state)
        return (*self._toward(state, path[0].joints_rad), self._gripper_command), self.phase

    def _missed(self, state: tuple[float, ...]) -> bool:
        """True once the gripper has risen LIFT_CHECK_M while the cup stayed down."""
        if self._grasp_z is None or self._cup_rest_z is None:
            return False
        risen = float(self.chain.position(state[:6], self.lane)[2]) - self._grasp_z
        cup_rise = float(self.object_pose().position_m[2]) - self._cup_rest_z
        return risen >= LIFT_CHECK_M and cup_rise < LIFT_CONFIRM_M

    def _begin_regrasp(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """Open where the arm is; the backoff and new approach follow."""
        if self._regrasps >= self.max_regrasps:
            return self._give_up(state)
        self._regrasps += 1
        self._corrections = 0
        self._last_depth = None
        self.record.stalls.append({"phase": "lift", "step": self.record.steps,
                                   "missed_grasp": self._regrasps})
        self._here(state)
        self._enter("reopen", state)
        self._gripper_command = self.strategy.open_m
        return (*state[:6], self._gripper_command), self.phase

    def _pushed(self) -> bool:
        """True when the cup has moved CONTACT_MOVE_M sideways since the descent's plan."""
        if self._planned_cup_xy is None:
            return False
        planned = self._planned_cup_xy
        heading = planned / max(float(np.linalg.norm(planned)), 1e-9)
        moved = np.asarray(self.object_pose().position_m[:2], dtype=float) - planned
        return abs(float(moved @ np.array([-heading[1], heading[0]]))) >= CONTACT_MOVE_M

    def _fingertip_on_wall(self, state: tuple[float, ...]) -> bool:
        """The cup was shoved sideways by a gripper that came in off its centre: back out.

        The offset is measured from where the cup stood when the descent was planned,
        since a shoved cup moves with the fingertip and stays against it. A cup nudged
        by a gripper that came in centred is the push-in seating it, and the descent
        carries on.
        """
        if not self._pushed() or self._planned_cup_xy is None:
            return False
        planned = self._planned_cup_xy
        heading = planned / max(float(np.linalg.norm(planned)), 1e-9)
        centre = self.chain.position(state[:6], self.lane)
        offset = np.asarray(centre[:2], dtype=float) - planned
        return abs(float(offset @ np.array([-heading[1], heading[0]]))) > FINGERTIP_ACROSS_M

    def _recentre(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """Back out open to a fresh pre-grasp, then look and come in centred again."""
        if self._recentres >= self.max_recentres:
            return self._give_up(state)
        self._recentres += 1
        self._corrections = 0
        self._last_depth = None
        self.record.stalls.append({"phase": self.phase, "step": self.record.steps,
                                   "recentre": self._recentres})
        return self._backoff(state)

    def _backoff(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """Re-plan from the cup's fresh pose and back out to the new pre-grasp."""
        objective = np.asarray(self.object_pose().position_m, dtype=float)
        self._cup_rest_z = float(objective[2])
        if self._solve_pick(objective, tuple(state[:6]), with_arrival=False) is None:
            return self._give_up(state)
        self._here(state)
        path = self._line("here", "pregrasp", "backoff")
        if path is None:
            return self._give_up(state)
        self._path = path
        self._enter("backoff", state)
        return (*self._toward(state, path[0].joints_rad), self._gripper_command), self.phase


class StagedPolicyService:
    """Serve staged executors on the policy protocol, one executor per trial reset."""

    def __init__(
        self,
        contract: PhysicalDatasetContract,
        chain: KinematicChain,
        limits: PhysicalLimits,
        object_pose: Callable[[], ObjectPose],
        target_pose: Callable[[], ObjectPose] | None,
        *,
        lane: str = SIMULATION_LANE,
        command_period_s: float,
        max_joint_speed_rad_s: float,
        max_gripper_speed_m_s: float,
        speed_fraction: float = 0.8,
        margin_rad: float = math.radians(3.0),
        stage_log: Path | None = None,
        requested_rate_hz: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Fix the task, the strategy and the step limits; refuse the physical lane."""
        if lane != SIMULATION_LANE:
            raise ValueError("the staged executor is served in the simulation lane only")
        score = SCORE_BY_TASK.get(contract.task_id)
        if score is None:
            raise ValueError(f"task {contract.task_id} has no scene score")
        object_name = OBJECT_BY_TASK.get(contract.task_id)
        if object_name is None or object_name not in STRATEGIES:
            raise ValueError(f"no grasp strategy for the object of task {contract.task_id}")
        if not 0 < speed_fraction <= 1 or command_period_s <= 0:
            raise ValueError("speed fraction must lie in (0, 1] and the period be positive")
        self.contract, self.chain, self.limits, self.lane = contract, chain, limits, lane
        self.score, self.strategy = score, STRATEGIES[object_name]
        self.object_pose, self.target_pose = object_pose, target_pose
        self.command_period_s, self.requested_rate_hz = command_period_s, requested_rate_hz
        self.max_step_rad = max_joint_speed_rad_s * command_period_s * speed_fraction
        self.max_gripper_step_m = max_gripper_speed_m_s * command_period_s * speed_fraction
        self.margin_rad, self.stage_log, self.clock = margin_rad, stage_log, clock
        self.trial = 0
        self.executor: StagedExecutor | None = None
        self._lock = threading.Lock()
        if stage_log is not None:
            stage_log.parent.mkdir(parents=True, exist_ok=True)
            stage_log.touch()

    def metadata(self) -> dict[str, Any]:
        """What a trial report records about this executor; no checkpoint, no dataset."""
        stride = max(1, int(round(self.command_period_s * self.requested_rate_hz)))
        return {
            "policy_id": EXECUTOR_VERSION,
            "step": None,
            "checkpoint_content_sha256": None,
            "dataset_content_sha256": None,
            "probe_sha256": None,
            "task_id": self.contract.task_id,
            "physical_contract": self.contract.as_dict(fps=self.requested_rate_hz),
            "cadence": {"command_period_s": self.command_period_s,
                        "command_stride_steps": stride},
            "response_timeout_s": None,
            "temporal_ensemble_coeff": None,
            "image_transforms": None,
            "model_feature_shapes": {key: None for key in IMAGE_KEYS},
            "lane": self.lane,
            "physical_eligible": False,
            "fixed_scene_motion_eligible": False,
            "simulation_motion_eligible": True,
            "eligible_task_ids": [self.contract.task_id],
            "varying_target_motion_eligible": False,
            "task_text_conditioning": False,
            "status": "implemented, unmeasured",
            "stage_status": "planned",
            "executor": {
                "kind": EXECUTOR_KIND, "version": EXECUTOR_VERSION,
                "strategy": asdict(self.strategy), "phases": list(PHASES),
                "where_source": "scene published poses",
                "max_step_rad": self.max_step_rad, "margin_rad": self.margin_rad,
                "note": ("a scripted expert with inverse kinematics on the committed chain; "
                         "not a learned policy; simulation lane only"),
            },
        }

    def _log(self, entry: dict[str, Any]) -> None:
        if self.stage_log is None:
            return
        with self.stage_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")

    def _finish_trial(self) -> None:
        if self.executor is None:
            return
        record = self.executor.record
        self._log({"kind": "trial", "trial": self.trial, **asdict(record)})

    def request(self, body: bytes) -> bytes:
        """Decode one policy request, step the trial's executor, encode the action."""
        started = self.clock()
        payload = decode_policy_request(body)
        state = tuple(float(v) for v in payload["observation.state"][:7])
        with self._lock:
            if payload.get("reset") or self.executor is None:
                self._finish_trial()
                self.trial += 1
                self.executor = StagedExecutor(
                    self.chain, self.lane, self.limits, self.contract.gripper_type,
                    self.strategy, self.score, self.object_pose, self.target_pose,
                    max_step_rad=self.max_step_rad, max_gripper_step_m=self.max_gripper_step_m,
                    margin_rad=self.margin_rad, clock=self.clock,
                )
            before = self.executor.phase
            action, phase = self.executor.step(state)
            if phase != before:
                self._log({"kind": "phase", "trial": self.trial, "phase": phase,
                           "step": self.executor.record.steps,
                           "feasible": self.executor.record.feasible})
        return encode_policy_response({
            "request_id": payload["request_id"],
            "observation_timestamp_s": payload["observation_timestamp_s"],
            "action": [float(v) for v in action],
            "inference_s": max(0.0, self.clock() - started),
            "stage": {"trial": self.trial, "phase": phase},
        })

    def close(self) -> None:
        """Write the last trial's record."""
        with self._lock:
            self._finish_trial()
            self.executor = None

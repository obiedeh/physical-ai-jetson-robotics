"""Closed-loop policy trials in the Isaac scene, scored from the scene's object poses.

The runner is the simulation lane's counterpart of the physical probe runner. It reads
the sim arm state and the two recorded camera views from the scene's sim topics, asks a
policy server started with ``--lane simulation`` for actions through the ordinary
:class:`~synria_lerobot.policy_client.PolicyClient`, and publishes seven-value joint
targets on a sim-only topic that the scene's policy launch mode mirrors. Every trial
starts with an approach from wherever the arm is to a fixed start pose, then a
randomized object reset laid out for that pose, and a policy reset; it ends when the
task's score has been met for
``hold_s`` without a break, or at the trial time limit. ``pick_cup`` scores a lift: the
cup held ``lift_m`` above its rest height. ``cup_pick_and_place`` scores a placement:
the cup lifted at least once, then at rest inside the scene's mark with the gripper
backed clear. Both come from the scene's published poses, not from an operator label,
so a data-scaling study can compare runs on the same rule. With a placement seed every
trial's reset is seeded (seed plus trial index) and the arm stands at the same start
pose when the layout is drawn, so two campaigns with the same seed face the same
placements and their success rates can be compared trial by trial.

A hold is not a dead end here. When the client holds because the sim state sits a
fraction past an operator bound (the physics drive overshoots a clamped target, or the
sim gripper reads a hair over its stroke), the runner publishes a recovery target
clamped just inside the bounds so the drive pulls the joint back and the policy resumes
on the next step. The physical client keeps holding in that case, as it must; this
recovery exists only in the simulation lane and is counted per trial.

Nothing here can reach the real arm: the target topic is refused when it is any of the
real arm's command or feedback topics, the contract must carry the simulation purpose,
and the server must report the simulation lane. Reports say so in every record.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .embodiment import SynriaObservation
from .physical_contract import (
    DEFAULT_COMMAND_TOPICS,
    DRIVER_JOINT_NAMES,
    SIMULATION_LANE,
    ImageFrame,
    PhysicalState,
    require_absolute_topic,
    training_lane,
)
from .policy_client import PolicyClient
from .recorder import _close_all, _discard_result, _LatestSource

SIM_STATE_TOPIC = "/isaac_joint_states"
SIM_WRIST_TOPIC = "/isaac/wrist_camera/image_raw"
SIM_OVERHEAD_TOPIC = "/isaac/overhead_camera/image_raw"
SIM_TARGET_TOPIC = "/isaac/policy_joint_targets"
SIM_RANDOMIZE_TOPIC = "/pick_objects_randomize"
SIM_RANDOMIZE_SCALE_TOPIC = "/pick_objects_randomize_scale"
SIM_RANDOMIZE_SEEDED_TOPIC = "/pick_objects_randomize_seeded"
RANDOMIZE_SCALE_BOUNDS = (0.5, 3.0)
SIM_CUP_POSE_TOPIC = "/isaac/pick_objects/cup/pose"
SIM_GRIPPER_POSE_TOPIC = "/isaac/pick_objects/gripper/pose"
SIM_PLACE_TARGET_POSE_TOPIC = "/isaac/pick_objects/place_target/pose"
LIFT_SCORE = "lift"
PLACE_SCORE = "place"
# How each recordable task is scored in the scene; a task not named here runs no trials.
SCORE_BY_TASK = {"pick_cup": LIFT_SCORE, "cup_pick_and_place": PLACE_SCORE}
# The real arm's command topics, its feedback topic and the deployment bridge's own name.
REAL_ARM_TOPICS = frozenset({*DEFAULT_COMMAND_TOPICS, "/joint_states", "/policy_arm"})
REPORT_KIND = "simulation_policy_trials"
# Hold reasons the client gives when the observed state, not the policy, is at fault.
RECOVERABLE_HOLDS = frozenset({"current state outside limits", "gripper exceeds contract stroke"})


def require_sim_only_topic(topic: str) -> str:
    """An absolute topic that is not one of the real arm's; the runner publishes on no other."""
    require_absolute_topic(topic)
    if topic in REAL_ARM_TOPICS:
        raise ValueError(
            f"{topic} belongs to the real arm; the simulation runner never publishes there"
        )
    return topic


@dataclass(frozen=True)
class ObjectPose:
    """One object's world position with the host time it arrived."""

    position_m: tuple[float, float, float]
    monotonic_timestamp_s: float


class PoseSource(Protocol):
    """Latest world pose of one scene object."""

    def read(self) -> ObjectPose: ...
    def close(self) -> None: ...


class StateSource(Protocol):
    """Latest sim arm state."""

    def read(self) -> PhysicalState: ...
    def close(self) -> None: ...


class FrameSource(Protocol):
    """Latest frame of one sim camera."""

    def read(self) -> ImageFrame: ...
    def close(self) -> None: ...


class TargetSink(Protocol):
    """Where accepted actions go; ``hold`` repeats the last target or sends nothing."""

    def offer(self, action: tuple[float, ...]) -> None: ...
    def hold(self) -> None: ...


@dataclass(frozen=True)
class TrialConfig:
    """The fixed rules of one trial campaign, recorded in full in the report."""

    trials: int
    command_period_s: float
    start_pose: tuple[float, ...]
    max_trial_s: float = 30.0
    lift_m: float = 0.03
    hold_s: float = 1.0
    settle_s: float = 2.0
    approach_s: float = 2.0
    recovery_margin_rad: float = math.radians(0.5)
    recovery_gripper_margin_m: float = 0.001
    # The score the task is judged by, and the placement rule's radii (place score only).
    score: str = LIFT_SCORE
    place_radius_m: float = 0.03
    place_height_m: float = 0.01
    gripper_clear_m: float = 0.06
    # The scene's Randomize scale (the panel's 1x, 2x, 3x) and, when set, the seed of
    # trial 0's layout draw; trial i draws from seed + i. None leaves the scene's own draws.
    randomize_scale: float = 1.0
    placement_seed: int | None = None
    # Diagnostic close gate, simulation lane only: while the scene's gripper centre is
    # farther than this from the cup's centre, a command that would close the gripper
    # further is replaced by its current width. None leaves every command as offered.
    # It reads the scene's true poses, so a gated campaign measures how much of a
    # policy's failure is the timing of its close; it is never a deployable rule.
    close_gate_m: float | None = None
    # Seated gate, the same diagnostic judged against the cup's shape: a close passes
    # only with the gripper centre within ``seat_sideways_m`` of the cup's vertical axis
    # and between the two heights of ``seat_height_m`` above the cup's base, so the cup
    # sits between the jaws rather than against one finger. Both gates apply when set.
    seat_sideways_m: float | None = None
    seat_height_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Reject a campaign that could not be scored or replayed."""
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        for name in ("command_period_s", "max_trial_s", "lift_m", "hold_s", "settle_s",
                     "approach_s", "recovery_margin_rad", "recovery_gripper_margin_m",
                     "place_radius_m", "place_height_m", "gripper_clear_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.score not in (LIFT_SCORE, PLACE_SCORE):
            raise ValueError(f"score must be {LIFT_SCORE} or {PLACE_SCORE}")
        low, high = RANDOMIZE_SCALE_BOUNDS
        if (type(self.randomize_scale) not in (int, float)
                or not low <= float(self.randomize_scale) <= high):
            raise ValueError(f"randomize_scale must be between {low:g} and {high:g}")
        if self.placement_seed is not None and (type(self.placement_seed) is not int
                                                or self.placement_seed < 0):
            raise ValueError("placement_seed must be a nonnegative integer or None")
        if self.close_gate_m is not None and (
                type(self.close_gate_m) not in (int, float)
                or not math.isfinite(self.close_gate_m) or self.close_gate_m <= 0):
            raise ValueError("close_gate_m must be a positive finite length or None")
        if (self.seat_sideways_m is None) != (self.seat_height_m is None):
            raise ValueError("the seated gate needs both seat_sideways_m and seat_height_m")
        if self.seat_sideways_m is not None:
            assert self.seat_height_m is not None
            band = tuple(float(value) for value in self.seat_height_m)
            if (type(self.seat_sideways_m) not in (int, float)
                    or not math.isfinite(self.seat_sideways_m) or self.seat_sideways_m <= 0):
                raise ValueError("seat_sideways_m must be a positive finite length")
            if (len(band) != 2 or not all(math.isfinite(value) for value in band)
                    or not 0 <= band[0] < band[1]):
                raise ValueError("seat_height_m must be two finite heights, low below high")
            object.__setattr__(self, "seat_height_m", band)
        pose = tuple(float(value) for value in self.start_pose)
        if len(pose) != 7 or any(not math.isfinite(value) for value in pose) or pose[6] < 0:
            raise ValueError("start_pose needs six joint angles in radians and a gripper width")
        object.__setattr__(self, "start_pose", pose)
        if self.hold_s >= self.max_trial_s:
            raise ValueError("hold_s must be shorter than max_trial_s")


class LiftScorer:
    """Success when the cup stays ``lift_m`` above its rest height for ``hold_s`` unbroken."""

    def __init__(self, rest_z_m: float, lift_m: float, hold_s: float) -> None:
        """Remember the rest height measured before the trial started."""
        if not all(math.isfinite(value) for value in (rest_z_m, lift_m, hold_s)):
            raise ValueError("scorer heights and hold time must be finite")
        self.rest_z_m = float(rest_z_m)
        self.lift_m = float(lift_m)
        self.hold_s = float(hold_s)
        self.max_lift_m = 0.0
        self.success_at_s: float | None = None
        self._lifted_since: float | None = None

    def feed(self, z_m: float, now_s: float) -> bool:
        """Record one cup height sample; True once the hold has been met."""
        lift = float(z_m) - self.rest_z_m
        self.max_lift_m = max(self.max_lift_m, lift)
        if lift >= self.lift_m:
            if self._lifted_since is None:
                self._lifted_since = now_s
            elif self.success_at_s is None and now_s - self._lifted_since >= self.hold_s:
                self.success_at_s = now_s
        else:
            self._lifted_since = None
        return self.success_at_s is not None


class PlaceScorer:
    """Success when the cup, lifted at least once, rests inside the mark with the gripper clear.

    Inside the mark means the cup's ground-plane distance to the mark's centre is at most
    ``radius_m``. At rest means the cup's height is within ``height_m`` of its rest height,
    so a cup held above the mark, or dropped onto it and still bouncing, does not count.
    Clear means the gripper centre is at least ``clear_m`` from the cup. The cup must have
    risen ``lift_m`` at some earlier sample, so a cup nudged along the floor into the mark
    is a failure. The hold restarts whenever any condition breaks, as the lift hold does.
    """

    def __init__(self, rest_z_m: float, target_xy_m: tuple[float, float], lift_m: float,
                 radius_m: float, height_m: float, clear_m: float, hold_s: float) -> None:
        """Remember the rest height and the mark measured before the trial started."""
        values = (rest_z_m, *target_xy_m, lift_m, radius_m, height_m, clear_m, hold_s)
        if len(target_xy_m) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("scorer heights, mark, radii and hold time must be finite")
        self.rest_z_m = float(rest_z_m)
        self.target_xy_m = (float(target_xy_m[0]), float(target_xy_m[1]))
        self.lift_m, self.radius_m = float(lift_m), float(radius_m)
        self.height_m, self.clear_m, self.hold_s = float(height_m), float(clear_m), float(hold_s)
        self.max_lift_m = 0.0
        self.lifted = False
        self.min_target_distance_m: float | None = None
        self.final_target_distance_m: float | None = None
        self.success_at_s: float | None = None
        self._placed_since: float | None = None

    def feed(self, cup_m: tuple[float, float, float],
             gripper_m: tuple[float, float, float] | None, now_s: float) -> bool:
        """Record one cup and gripper sample; True once the placement hold has been met."""
        lift = float(cup_m[2]) - self.rest_z_m
        self.max_lift_m = max(self.max_lift_m, lift)
        self.lifted = self.lifted or lift >= self.lift_m
        distance = math.dist((float(cup_m[0]), float(cup_m[1])), self.target_xy_m)
        self.final_target_distance_m = distance
        self.min_target_distance_m = (distance if self.min_target_distance_m is None
                                      else min(self.min_target_distance_m, distance))
        clear = gripper_m is not None and math.dist(gripper_m, cup_m) >= self.clear_m
        placed = (self.lifted and distance <= self.radius_m and abs(lift) <= self.height_m
                  and clear)
        if placed:
            if self._placed_since is None:
                self._placed_since = now_s
            elif self.success_at_s is None and now_s - self._placed_since >= self.hold_s:
                self.success_at_s = now_s
        else:
            self._placed_since = None
        return self.success_at_s is not None


def approach_targets(
    current: tuple[float, ...], goal: tuple[float, ...], max_delta: tuple[float, ...],
    steps: int, *, max_steps: int | None = None,
) -> list[tuple[float, ...]]:
    """Targets that walk every value from ``current`` to ``goal`` in about ``steps`` steps.

    Each step moves each value by its share of the distance, never more than that
    value's ``max_delta``, so the published targets respect the same per-command bound
    the policy's actions do. When the bound makes ``steps`` too few, the walk continues
    at the bound until the goal is reached, up to ``max_steps`` (five times ``steps``
    by default); beyond that the start pose is too far for the approach budget.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("approach steps must be a positive integer")
    if len(current) != len(goal) or len(goal) != len(max_delta):
        raise ValueError("current pose, goal pose and step bounds must have the same length")
    limit = steps * 5 if max_steps is None else max_steps
    position = tuple(float(value) for value in current)
    targets: list[tuple[float, ...]] = []
    while any(abs(g - p) > 1e-9 for g, p in zip(goal, position, strict=True)):
        if len(targets) >= limit:
            raise ValueError("start pose is too far for the approach budget")
        remaining = max(1, steps - len(targets))
        position = tuple(
            p + max(-d, min(d, (g - p) / remaining))
            for p, g, d in zip(position, goal, max_delta, strict=True)
        )
        targets.append(position)
    return targets


def recovery_target(
    current: tuple[float, ...], bounds: tuple[tuple[float, float], ...],
    joint_margin_rad: float, gripper_margin_m: float,
) -> tuple[float, ...]:
    """The current state pulled just inside every bound.

    Joints come inside by the joint margin, the gripper to its stroke less its margin. A
    value already inside stays where it is, so the recovery moves only what clipped.
    """
    if len(current) != 7 or len(bounds) != 7:
        raise ValueError("recovery needs seven values and seven bounds")
    target = []
    for index, (value, (lo, hi)) in enumerate(zip(current, bounds, strict=True)):
        margin = gripper_margin_m if index == 6 else joint_margin_rad
        low = lo if index == 6 else lo + margin
        high = hi - margin
        if low > high:
            raise ValueError("recovery margin is wider than the bound")
        target.append(min(high, max(low, float(value))))
    return tuple(target)


@dataclass(frozen=True)
class TrialRecord:
    """What one trial did, enough to recount the campaign from the report."""

    index: int
    success: bool
    time_to_success_s: float | None
    duration_s: float
    steps: int
    holds: int
    hold_reasons: dict[str, int]
    max_lift_m: float
    cup_rest_z_m: float
    approach_steps: int
    recoveries: int
    min_gripper_cup_distance_m: float | None
    final_gripper_cup_distance_m: float | None
    final_state: tuple[float, ...]
    # Where the cup stood after the reset, and for the place score, the mark and how
    # close the cup came to it; the lift score records the cup's start only.
    cup_start_xy_m: tuple[float, float]
    target_xy_m: tuple[float, float] | None
    min_cup_target_distance_m: float | None
    final_cup_target_distance_m: float | None
    placement_seed: int | None
    # Steps on which the client's gripper guard kept the gripper where it was.
    guarded_steps: int
    # Steps on which the diagnostic close gate kept the gripper from closing further.
    close_gated_steps: int = 0


class SimulationTrialRunner:
    """Run scored closed-loop trials against a simulation-lane policy server."""

    def __init__(
        self,
        client: PolicyClient,
        sink: TargetSink,
        state_source: StateSource,
        wrist_source: FrameSource,
        front_source: FrameSource,
        cup_source: PoseSource,
        randomize: Callable[[int | None, float], None],
        config: TrialConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        gripper_source: PoseSource | None = None,
        target_source: PoseSource | None = None,
        trace_path: Path | None = None,
    ) -> None:
        """Refuse anything but a simulation contract whose command period matches the config.

        The config's score must be the one :data:`SCORE_BY_TASK` names for the contract's
        task. ``gripper_source`` is optional for the lift score: with it every trial
        records how close the gripper came to the cup and where it ended, which tells a
        near miss from a far one. The place score needs it and ``target_source``, the
        scene's mark pose. ``randomize`` takes the trial's seed (None when the config has
        no placement seed) and the Randomize scale. ``trace_path`` names a JSON Lines
        file that receives one row per policy step of every trial: the observed state,
        the action offered or the hold and its reason, and the cup and gripper poses
        the scene published after it. The corpus policies of 2026-10-19 failed in the
        last centimetres with no record of what they asked for; the trace is that record.
        """
        contract = client.embodiment.contract
        contract.require_lane(SIMULATION_LANE)
        client.config.require_command_period(config.command_period_s)
        expected = SCORE_BY_TASK.get(contract.task_id)
        if expected is None:
            raise ValueError(f"task {contract.task_id} has no scene score; trials run for "
                             f"{sorted(SCORE_BY_TASK)}")
        if config.score != expected:
            raise ValueError(f"task {contract.task_id} is scored by {expected}, not {config.score}")
        if config.score == PLACE_SCORE and (gripper_source is None or target_source is None):
            raise ValueError("the place score needs the gripper pose and the mark pose")
        if ((config.close_gate_m is not None or config.seat_sideways_m is not None)
                and gripper_source is None):
            raise ValueError("the close gate needs the gripper pose")
        self.client, self.sink, self.config = client, sink, config
        self.state_source, self.wrist_source, self.front_source = (
            state_source, wrist_source, front_source,
        )
        self.cup_source, self.randomize = cup_source, randomize
        self.gripper_source, self.target_source = gripper_source, target_source
        self.clock, self.sleep, self.log = clock, sleep, log
        self.task = contract.task_text
        self.trace_path = trace_path
        self._trace: Any = None
        self.trace_rows = 0

    def observe(self) -> SynriaObservation:
        """The newest sample of every source, with the trained instruction and the goal.

        The goal channels the served contract declares are filled from the scene's
        published poses: "object" from the cup, "target" from the mark; a declared
        channel without a source refuses the trial rather than sending a guess.
        """
        goal: tuple[float, ...] | None = None
        channels = self.client.embodiment.contract.goal_channels
        if channels:
            values: list[float] = []
            for name in channels:
                if name == "object":
                    values.extend(self.cup_source.read().position_m)
                elif name == "target" and self.target_source is not None:
                    values.extend(self.target_source.read().position_m)
                else:
                    raise ValueError(f"no scene source for goal channel {name!r}")
            goal = tuple(values)
        return SynriaObservation(
            self.state_source.read(), self.wrist_source.read(), self.front_source.read(),
            self.task, goal,
        )

    def approach(self) -> int:
        """Walk the arm from where it is to the start pose; returns the targets sent."""
        state = self.state_source.read()
        current = (*state.joint_positions_rad, state.gripper_m)
        steps = max(1, round(self.config.approach_s / self.config.command_period_s))
        targets = approach_targets(current, self.config.start_pose, self.client.config.max_delta,
                                   steps)
        for target in targets:
            self.sink.offer(target)
            self.sleep(self.config.command_period_s)
        return len(targets)

    def run_trial(self, index: int) -> TrialRecord:
        """One trial: approach, seeded random reset, policy reset, steps to the score or limit."""
        config = self.config
        approach_steps = self.approach()
        self.sleep(config.settle_s / 2)
        seed = None if config.placement_seed is None else config.placement_seed + index
        self.randomize(seed, config.randomize_scale)
        self.sleep(config.settle_s)
        cup_start = self.cup_source.read().position_m
        rest_z = cup_start[2]
        scorer: LiftScorer | PlaceScorer
        target_xy: tuple[float, float] | None = None
        if config.score == PLACE_SCORE:
            assert self.target_source is not None
            mark = self.target_source.read().position_m
            target_xy = (mark[0], mark[1])
            scorer = PlaceScorer(rest_z, target_xy, config.lift_m, config.place_radius_m,
                                 config.place_height_m, config.gripper_clear_m, config.hold_s)
        else:
            scorer = LiftScorer(rest_z, config.lift_m, config.hold_s)
        holds: Counter[str] = Counter()
        steps = 0
        recoveries = 0
        guarded = 0
        gated = 0
        min_distance: float | None = None
        final_distance: float | None = None
        last_state: tuple[float, ...] = ()
        started = self.clock()
        first = True
        offered: tuple[float, ...] | None = None
        close_gated = False
        try:
            while self.clock() - started < config.max_trial_s:
                tick = self.clock()
                observation = self.observe()
                last_state = (*observation.state.joint_positions_rad, observation.state.gripper_m)
                decision = self.client.step(observation, reset=first)
                first = False
                steps += 1
                offered, close_gated = None, False
                guarded += decision.gripper_guard is not None
                if decision.hold or decision.action is None:
                    holds[decision.reason] += 1
                    if decision.reason in RECOVERABLE_HOLDS:
                        state = observation.state
                        self.sink.offer(recovery_target(
                            (*state.joint_positions_rad, state.gripper_m), self.client.bounds,
                            config.recovery_margin_rad, config.recovery_gripper_margin_m,
                        ))
                        recoveries += 1
                    else:
                        self.sink.hold()
                else:
                    offered = self._gate_close(decision.action, observation.state.gripper_m)
                    close_gated = offered != decision.action
                    gated += close_gated
                    self.sink.offer(offered)
                cup = self.cup_source.read()
                gripper: tuple[float, float, float] | None = None
                if self.gripper_source is not None:
                    gripper = self.gripper_source.read().position_m
                    final_distance = math.dist(gripper, cup.position_m)
                    min_distance = (final_distance if min_distance is None
                                    else min(min_distance, final_distance))
                if isinstance(scorer, PlaceScorer):
                    done = scorer.feed(cup.position_m, gripper, self.clock())
                else:
                    done = scorer.feed(cup.position_m[2], self.clock())
                if self._trace is not None:
                    self._write_trace({
                        "trial": index, "step": steps, "t_s": round(tick - started, 4),
                        "state": list(last_state),
                        "action": None if offered is None else list(offered),
                        "hold": bool(decision.hold or decision.action is None),
                        "reason": decision.reason, "gripper_guard": decision.gripper_guard,
                        "recovery": bool((decision.hold or decision.action is None)
                                         and decision.reason in RECOVERABLE_HOLDS),
                        "close_gated": close_gated,
                        "inference_s": decision.inference_s,
                        "cup_m": list(cup.position_m), "cup_lift_m": cup.position_m[2] - rest_z,
                        "gripper_m": None if gripper is None else list(gripper),
                        "gripper_cup_distance_m": final_distance,
                    })
                if done:
                    break
                remaining = config.command_period_s - (self.clock() - tick)
                if remaining > 0:
                    self.sleep(remaining)
        finally:
            self.sink.hold()
        duration = self.clock() - started
        success_at = scorer.success_at_s
        success = success_at is not None
        record = TrialRecord(
            index=index, success=success,
            time_to_success_s=(success_at - started) if success_at is not None else None,
            duration_s=duration, steps=steps, holds=sum(holds.values()),
            hold_reasons=dict(sorted(holds.items())), max_lift_m=scorer.max_lift_m,
            cup_rest_z_m=rest_z, approach_steps=approach_steps, recoveries=recoveries,
            min_gripper_cup_distance_m=min_distance, final_gripper_cup_distance_m=final_distance,
            final_state=last_state,
            cup_start_xy_m=(cup_start[0], cup_start[1]), target_xy_m=target_xy,
            min_cup_target_distance_m=(scorer.min_target_distance_m
                                       if isinstance(scorer, PlaceScorer) else None),
            final_cup_target_distance_m=(scorer.final_target_distance_m
                                         if isinstance(scorer, PlaceScorer) else None),
            placement_seed=seed, guarded_steps=guarded, close_gated_steps=gated,
        )
        verdict = "success" if success else "failure"
        closest = "" if min_distance is None else f", closest {min_distance * 100:.1f} cm"
        placed = ""
        if isinstance(scorer, PlaceScorer) and scorer.final_target_distance_m is not None:
            placed = f", cup ended {scorer.final_target_distance_m * 100:.1f} cm from the mark"
        self.log(f"trial {index}: {verdict} after {duration:.1f} s, {steps} steps, "
                 f"{record.holds} holds, {recoveries} recoveries, {guarded} guarded, "
                 f"{gated} close-gated, "
                 f"max lift {scorer.max_lift_m * 100:.1f} cm{closest}{placed}")
        return record

    def _gate_close(self, action: tuple[float, ...], width_m: float) -> tuple[float, ...]:
        """Keep the gripper from closing further while the cup is outside the close gates.

        The gripper width grows as it closes (0 open, about 0.022 m shut). Joints and an
        opening or steady gripper pass unchanged; a closing command outside a gate
        becomes the current width, a step the client's speed bound already allows.
        """
        config = self.config
        if (config.close_gate_m is None and config.seat_sideways_m is None) \
                or action[6] <= width_m:
            return action
        assert self.gripper_source is not None
        gripper = self.gripper_source.read().position_m
        cup = self.cup_source.read().position_m
        allowed = config.close_gate_m is None or math.dist(gripper, cup) <= config.close_gate_m
        if config.seat_sideways_m is not None:
            assert config.seat_height_m is not None
            low, high = config.seat_height_m
            sideways = math.hypot(gripper[0] - cup[0], gripper[1] - cup[1])
            allowed = (allowed and sideways <= config.seat_sideways_m
                       and low <= gripper[2] - cup[2] <= high)
        return action if allowed else (*action[:6], width_m)

    def _write_trace(self, row: dict[str, Any]) -> None:
        """One JSON Lines row; the file is opened by :meth:`run` and never appended to."""
        self._trace.write(json.dumps(row, allow_nan=False) + "\n")
        self.trace_rows += 1

    def close_gate_rule(self) -> str | None:
        """The diagnostic close gates in words, as written into the report; None when off."""
        config = self.config
        parts = []
        if config.close_gate_m is not None:
            parts.append(f"the scene's gripper centre is more than "
                         f"{config.close_gate_m * 100:.1f} cm from the cup's centre")
        if config.seat_sideways_m is not None:
            assert config.seat_height_m is not None
            low, high = config.seat_height_m
            parts.append(f"the gripper centre is more than {config.seat_sideways_m * 100:.1f} cm "
                         f"sideways from the cup's axis or outside {low * 100:.1f} to "
                         f"{high * 100:.1f} cm above the cup's base")
        if not parts:
            return None
        return ("diagnostic, simulation only: a command closing the gripper further is "
                "replaced by its current width while " + " or while ".join(parts))

    def score_rule(self) -> str:
        """The scoring rule in words, as written into the report."""
        config = self.config
        if config.score == PLACE_SCORE:
            return (f"cup lifted {config.lift_m * 100:.1f} cm at least once, then at rest within "
                    f"{config.place_radius_m * 100:.1f} cm of the mark's centre and within "
                    f"{config.place_height_m * 100:.1f} cm of its rest height with the gripper "
                    f"centre at least {config.gripper_clear_m * 100:.1f} cm from the cup, for "
                    f"{config.hold_s:.1f} s without a break, from the scene's published cup, "
                    f"mark and gripper poses")
        return (f"cup {config.lift_m * 100:.1f} cm above its rest height for "
                f"{config.hold_s:.1f} s without a break, from the scene's published cup pose")

    def run(self, policy: dict[str, Any]) -> dict[str, Any]:
        """Run every trial and return the campaign report; the server must be in the sim lane."""
        if training_lane(policy) != SIMULATION_LANE:
            raise ValueError("the policy server is not serving the simulation lane")
        contract = self.client.embodiment.contract
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.trace_path.open("x", encoding="utf-8")
        try:
            trials = [self.run_trial(index) for index in range(self.config.trials)]
        finally:
            if self._trace is not None:
                self._trace.close()
                self._trace = None
        successes = sum(1 for trial in trials if trial.success)
        return {
            "kind": REPORT_KIND,
            "lane": SIMULATION_LANE,
            "physical_evidence": False,
            "utc": datetime.now(timezone.utc).isoformat(),
            "task_id": contract.task_id,
            "task_text": contract.task_text,
            "policy": {key: policy.get(key) for key in (
                "policy_id", "step", "checkpoint_content_sha256", "dataset_content_sha256",
                "probe_sha256", "lane",
            )},
            "config": asdict(self.config),
            "safety": asdict(self.client.config),
            "score": self.config.score,
            "score_rule": self.score_rule(),
            "recovery_rule": ("a hold for a state past an operator bound publishes the state "
                              "clamped just inside the bounds, simulation lane only"),
            "placement_rule": ("the arm is walked to the start pose first, then the scene "
                               "draws a Randomize layout at the configured scale"
                               + (", seeded with placement_seed plus the trial index so "
                                  "campaigns sharing the seed face the same placements"
                                  if self.config.placement_seed is not None else
                                  " from the scene's own unseeded draws")),
            "close_gate_rule": self.close_gate_rule(),
            "trials": [asdict(trial) for trial in trials],
            "trace": (None if self.trace_path is None else
                      {"path": str(self.trace_path), "rows": self.trace_rows,
                       "row": "one per policy step: trial, step, t_s, state, action offered or "
                              "hold with reason, gripper_guard, recovery, close_gated, "
                              "inference_s, "
                              "cup_m, cup_lift_m, gripper_m, gripper_cup_distance_m"}),
            "successes": successes,
            "success_rate": successes / len(trials),
            "note": "Simulation lane; scored in the Isaac scene; never physical evidence.",
        }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the campaign report once; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_placement_tolerance(path: Path, task_id: str) -> float:
    """The committed placement radius for one task; refused when the task has none."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "synria_placement_tolerances_v1":
        raise ValueError("unsupported placement tolerances file")
    radius = payload.get("radius_m", {}).get(task_id)
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"no placement tolerance for task {task_id}")
    return float(radius)


def start_pose_from_dataset(root: Path) -> tuple[float, ...]:
    """The mean first-frame state of every episode in a LeRobot dataset, as a start pose."""
    from pyarrow import parquet

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("dataset has no data files")
    firsts: list[list[float]] = []
    for path in files:
        table = parquet.read_table(path, columns=["observation.state", "frame_index"])
        rows = table.to_pylist()
        firsts.extend([float(v) for v in row["observation.state"][:7]]
                      for row in rows if row["frame_index"] == 0)
    if not firsts:
        raise ValueError("dataset has no first frames")
    return tuple(sum(values) / len(firsts) for values in zip(*firsts, strict=True))


class RosPoseSource(_LatestSource[ObjectPose]):
    """Owned-context, depth-one subscriber to a geometry_msgs/PoseStamped topic."""

    def __init__(self, topic: str, *, node_name: str, timeout_s: float = 1.0) -> None:
        """Create the subscription inside an owned ROS context; close everything on failure."""
        super().__init__(timeout_s=timeout_s)
        require_absolute_topic(topic)
        self._topic = topic
        try:
            import rclpy  # type: ignore[import-not-found]
            from geometry_msgs.msg import PoseStamped  # type: ignore[import-not-found]
            from rclpy.context import Context  # type: ignore[import-not-found]
            from rclpy.executors import (  # type: ignore[import-not-found]
                ExternalShutdownException,
                SingleThreadedExecutor,
            )
        except ImportError as exc:  # pragma: no cover - optional ROS dependency
            raise RuntimeError("ROS 2 pose dependencies are unavailable") from exc
        self._rclpy = rclpy
        self._context = Context()
        with ExitStack() as startup:
            rclpy.init(context=self._context)
            startup.callback(_close_all, lambda: rclpy.shutdown(context=self._context))
            self._node = rclpy.create_node(node_name, context=self._context)
            startup.callback(_close_all, self._node.destroy_node)
            self._executor = SingleThreadedExecutor(context=self._context)
            startup.callback(_close_all, _discard_result(
                lambda: self._executor.shutdown(timeout_sec=self._timeout_s)
            ))

            def on_pose(message: Any) -> None:
                """Cache the newest position with its host arrival time."""
                position = message.pose.position
                self._store(ObjectPose(
                    (float(position.x), float(position.y), float(position.z)), time.monotonic(),
                ))

            self._subscription = self._node.create_subscription(PoseStamped, topic, on_pose, 1)
            startup.callback(_close_all, _discard_result(
                lambda: self._node.destroy_subscription(self._subscription)
            ))
            self._executor.add_node(self._node)

            def spin() -> None:
                """Run callbacks until owned shutdown and surface unexpected stops."""
                try:
                    self._executor.spin()
                    if not self._stop.is_set():
                        raise RuntimeError("ROS executor stopped unexpectedly")
                except ExternalShutdownException:
                    if not self._stop.is_set():
                        raise
            self._start_worker(spin, node_name)
            startup.pop_all()

    def read(self) -> ObjectPose:
        """Read the latest pose or report that none has arrived yet."""
        try:
            return self._read_latest()
        except RuntimeError as error:
            if "no source sample" in str(error):
                raise RuntimeError(
                    f"No pose received on {self._topic}; check that Isaac policy mode is "
                    "publishing object poses and that this terminal uses the same ROS domain."
                ) from error
            raise

    def close(self) -> None:
        """Stop the executor and release the owned context."""
        if not self._closed:
            self._stop.set()
            self._executor.shutdown(timeout_sec=self._timeout_s)
            self._join()
            _close_all(
                _discard_result(lambda: self._node.destroy_subscription(self._subscription)),
                self._node.destroy_node,
                lambda: self._rclpy.shutdown(context=self._context),
            )
            self._closed = True


class RosSimPublisher:
    """Joint targets on the sim-only target topic and reset requests on the randomize topics.

    The target topic is checked against the real arm's topics before any publisher
    exists. ``hold`` republishes the last target so the scene's drive keeps its place;
    before any target it sends nothing. A reset request goes on the plain randomize
    topic (Bool, 1x), the scale topic (Float64) for another scale, or the seeded topic
    (Float64MultiArray of seed and scale) when the trial carries a placement seed.
    """

    def __init__(
        self, *, target_topic: str = SIM_TARGET_TOPIC, randomize_topic: str = SIM_RANDOMIZE_TOPIC,
        scale_topic: str = SIM_RANDOMIZE_SCALE_TOPIC,
        seeded_topic: str = SIM_RANDOMIZE_SEEDED_TOPIC,
        node_name: str = "synria_sim_policy_runner",
    ) -> None:
        """Create the publishers inside an owned ROS context."""
        for topic in (target_topic, randomize_topic, scale_topic, seeded_topic):
            require_sim_only_topic(topic)
        try:
            import rclpy  # type: ignore[import-not-found]
            from rclpy.context import Context  # type: ignore[import-not-found]
            from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
            from std_msgs.msg import (  # type: ignore[import-not-found]
                Bool,
                Float64,
                Float64MultiArray,
            )
        except ImportError as exc:  # pragma: no cover - optional ROS dependency
            raise RuntimeError("ROS 2 publisher dependencies are unavailable") from exc
        self._rclpy = rclpy
        self._JointState, self._Bool = JointState, Bool
        self._Float64, self._Float64MultiArray = Float64, Float64MultiArray
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(node_name, context=self._context)
        self._targets = self._node.create_publisher(JointState, target_topic, 10)
        self._randomize = self._node.create_publisher(Bool, randomize_topic, 10)
        self._scale = self._node.create_publisher(Float64, scale_topic, 10)
        self._seeded = self._node.create_publisher(Float64MultiArray, seeded_topic, 10)
        self._lock = threading.Lock()
        self._last: tuple[float, ...] | None = None
        self.sent = 0
        self.target_topic, self.randomize_topic = target_topic, randomize_topic
        self.scale_topic, self.seeded_topic = scale_topic, seeded_topic
        self._closed = False

    def offer(self, action: tuple[float, ...]) -> None:
        """Publish one seven-value target in the driver's joint order."""
        values = tuple(float(value) for value in action)
        if len(values) != 7 or any(not math.isfinite(value) for value in values):
            raise ValueError("a target needs seven finite values")
        message = self._JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = [*DRIVER_JOINT_NAMES, "Gripper"]
        message.position = list(values)
        with self._lock:
            self._targets.publish(message)
            self._last = values
            self.sent += 1

    def hold(self) -> None:
        """Repeat the last target, or send nothing when none has been sent."""
        with self._lock:
            last = self._last
        if last is not None:
            self.offer(last)

    def request_randomize(self, seed: int | None = None, scale: float = 1.0) -> None:
        """Ask the scene for a Randomize and reset at ``scale``, seeded when ``seed`` is given."""
        low, high = RANDOMIZE_SCALE_BOUNDS
        if not low <= float(scale) <= high:
            raise ValueError(f"randomize scale must be between {low:g} and {high:g}")
        if seed is not None:
            seeded = self._Float64MultiArray()
            seeded.data = [float(int(seed)), float(scale)]
            self._seeded.publish(seeded)
        elif float(scale) == 1.0:
            message = self._Bool()
            message.data = True
            self._randomize.publish(message)
        else:
            scaled = self._Float64()
            scaled.data = float(scale)
            self._scale.publish(scaled)

    def close(self) -> None:
        """Destroy the publishers and release the owned context."""
        if not self._closed:
            _close_all(
                _discard_result(lambda: self._node.destroy_publisher(self._targets)),
                _discard_result(lambda: self._node.destroy_publisher(self._randomize)),
                _discard_result(lambda: self._node.destroy_publisher(self._scale)),
                _discard_result(lambda: self._node.destroy_publisher(self._seeded)),
                self._node.destroy_node,
                lambda: self._rclpy.shutdown(context=self._context),
            )
            self._closed = True

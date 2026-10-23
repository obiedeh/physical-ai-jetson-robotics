"""The demonstration engine: the staged executor drives the scene while the recorder records.

A dataset of hundreds of varied, goal-labelled demonstrations is what a vision policy
needs and what teleoperation cannot supply at fifty takes a session. This engine
records such a dataset in the Isaac scene under the same contract, writer and
quality gates as a teleoperated take: the recorder samples the sim state, both
cameras and the goal channels at the dataset rate; on every sample the executor
computes its next command from the sampled state, publishes it on the sim-only
target topic, and that command is the frame's leader-type action; the scene's
scorer decides the take's label. Successes are labelled success, failures are
labelled failure and appended to the dataset's exclusion log, so the record is
honest and training sees only the takes that reached the score.

Each take: walk to the start pose, request a seeded Randomize so the placements are
reproducible and distinct from the evaluation seeds, settle, start a fresh executor,
record until the score or the limit. The executor's stage record is kept per take.
Simulation lane only; the recorder refuses an injected action source anywhere else.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synria_lerobot.exclusions import append_exclusion
from synria_lerobot.physical_contract import ActionSource as ActionSourceKind
from synria_lerobot.physical_contract import PhysicalState
from synria_lerobot.recorder import ActionSample, PhysicalEpisodeRecorder
from synria_lerobot.sim_policy_runner import (
    LIFT_SCORE,
    PLACE_SCORE,
    LiftScorer,
    ObjectPose,
    PlaceScorer,
    TargetSink,
    approach_targets,
)
from synria_lerobot.staged_executor import StagedExecutor

OPERATOR = "staged_expert"
# After the score is met the take continues this long so the hold is on record.
SCORED_TAIL_S = 1.0
# A take whose executor is done or infeasible without the score ends after this grace.
UNSCORED_GRACE_S = 2.0


class ExecutorActionSource:
    """Leader-type actions that are the staged executor's commands, published as they are read.

    ``read`` is called by the recorder once per sample with the follower state it just
    sampled; the executor's command for that state is clamped to the operator bounds,
    as the policy client clamps every command, sent to the sink and returned as the
    frame's action, stamped with the follower sample's clocks so the skew and freshness
    gates see a contemporaneous source. The clamp matters: a step interpolated from a
    joint that sags past its stop would otherwise command past the stop, and the first
    corpus showed the sim arm sagging two degrees beyond on Joint3 when it did.
    ``reset`` starts a new executor for the next take.
    """

    kind = ActionSourceKind.LEADER

    def __init__(self, factory: Callable[[], StagedExecutor], sink: TargetSink,
                 bounds: tuple[tuple[float, float], ...]) -> None:
        """Keep the executor factory, the target sink and the seven bounds; no executor yet."""
        if len(bounds) != 7 or any(lo >= hi for lo, hi in bounds):
            raise ValueError("seven ordered operator bounds are required")
        self.factory, self.sink, self.bounds = factory, sink, tuple(bounds)
        self.executor: StagedExecutor | None = None
        self.steps = 0
        self.clamped = 0

    def reset(self) -> StagedExecutor:
        """A fresh executor for a new take."""
        self.executor = self.factory()
        self.steps = 0
        return self.executor

    @property
    def phase(self) -> str:
        """The current executor's phase, or ``plan`` before the first step."""
        return "plan" if self.executor is None else self.executor.phase

    def read(self, follower_state: PhysicalState) -> ActionSample:
        """One command for the sampled state, published and returned as the action."""
        if self.executor is None:
            raise RuntimeError("reset the action source before a take")
        state = (*follower_state.joint_positions_rad, follower_state.gripper_m)
        raw, _ = self.executor.step(state)
        action = tuple(min(hi, max(lo, float(v)))
                       for v, (lo, hi) in zip(raw, self.bounds, strict=True))
        self.clamped += action != tuple(float(v) for v in raw)
        self.sink.offer(action)
        self.steps += 1
        return ActionSample(
            values=action,
            monotonic_timestamp_s=follower_state.monotonic_timestamp_s,
            ros_header_stamp_s=follower_state.ros_header_stamp_s,
            ros_arrival_stamp_s=follower_state.ros_arrival_stamp_s,
        )

    def close(self) -> None:
        """Nothing owned beyond the executor."""
        self.executor = None


@dataclass(frozen=True)
class DemonstrationConfig:
    """How many takes, how they are placed, and the take's timing."""

    takes: int
    fps: float
    start_pose: tuple[float, ...]
    score: str
    placement_seed: int
    randomize_scale: float = 1.0
    max_take_s: float = 30.0
    approach_s: float = 2.0
    settle_s: float = 2.0
    lift_m: float = 0.03
    hold_s: float = 1.0
    place_radius_m: float = 0.03
    place_height_m: float = 0.01
    gripper_clear_m: float = 0.06
    max_step_rad: float = 0.05
    # Per-joint half-ranges, in radians, of a uniform offset added to the start pose of
    # each take, drawn from the take's placement seed; empty keeps every take's start
    # the same. A turned base starts some takes with the cup outside the wrist view.
    start_jitter_rad: tuple[float, ...] = ()
    # Keep a scored take recording until the executor is done (back at its rest pose,
    # held still), so the demonstration shows how the task ends, not just the score.
    finish_at_rest: bool = False

    def __post_init__(self) -> None:
        """Refuse nonsense counts and timing."""
        if type(self.takes) is not int or self.takes < 1 or type(self.placement_seed) is not int:
            raise ValueError("takes must be a positive integer and the seed an integer")
        if self.score not in (LIFT_SCORE, PLACE_SCORE) or len(self.start_pose) != 7:
            raise ValueError("score must be lift or place; the start pose has seven values")
        for name in ("fps", "max_take_s", "approach_s", "settle_s", "lift_m", "hold_s",
                     "place_radius_m", "place_height_m", "gripper_clear_m", "max_step_rad",
                     "randomize_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        jitter = tuple(float(value) for value in self.start_jitter_rad)
        if jitter and (len(jitter) != 6
                       or any(not math.isfinite(v) or not 0 <= v <= 1.0 for v in jitter)):
            raise ValueError("start jitter needs six half-ranges between 0 and 1 rad")
        object.__setattr__(self, "start_jitter_rad", jitter)

    def take_start_pose(self, index: int) -> tuple[float, ...]:
        """The start pose of take ``index``: the configured pose plus its seeded jitter."""
        if not self.start_jitter_rad:
            return self.start_pose
        rng = random.Random(self.placement_seed + index)
        joints = tuple(value + rng.uniform(-half, half)
                       for value, half in zip(self.start_pose[:6], self.start_jitter_rad,
                                              strict=True))
        return (*joints, self.start_pose[6])


@dataclass
class TakeRecord:
    """What one take did and how it was labelled."""

    index: int
    episode_index: int | None
    seed: int
    success: bool
    final_phase: str
    feasible: bool | None
    steps: int
    duration_s: float
    max_lift_m: float
    stage_record: dict[str, Any]
    clamped_steps: int = 0


class DemonstrationSession:
    """Record takes with the recorder while the executor drives the scene."""

    def __init__(
        self,
        recorder: PhysicalEpisodeRecorder,
        actions: ExecutorActionSource,
        sink: TargetSink,
        state_source: Any,
        cup_source: Callable[[], ObjectPose],
        gripper_source: Callable[[], ObjectPose] | None,
        target_source: Callable[[], ObjectPose] | None,
        randomize: Callable[[int | None, float], None],
        config: DemonstrationConfig,
        *,
        dataset_root: Path,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        """Bind the recorder and the scene; takes run on ``run``."""
        if config.score == PLACE_SCORE and (gripper_source is None or target_source is None):
            raise ValueError("the place score needs the gripper and the mark poses")
        self.recorder, self.actions, self.sink, self.state_source = (
            recorder, actions, sink, state_source)
        self.cup_source, self.gripper_source, self.target_source = (
            cup_source, gripper_source, target_source)
        self.randomize, self.config, self.dataset_root = randomize, config, dataset_root
        self.clock, self.sleep, self.log = clock, sleep, log
        self.takes: list[TakeRecord] = []

    def approach(self, index: int = 0) -> None:
        """Walk from wherever the arm is to the take's start pose at the executor's own step."""
        state = self.state_source.read()
        current = (*state.joint_positions_rad, state.gripper_m)
        period = 1.0 / self.config.fps
        steps = max(1, round(self.config.approach_s * self.config.fps))
        limits = (*(self.config.max_step_rad,) * 6, 0.004)
        for target in approach_targets(current, self.config.take_start_pose(index), limits,
                                       steps):
            self.sink.offer(target)
            self.sleep(period)

    def _scorer(self) -> LiftScorer | PlaceScorer:
        rest_z = self.cup_source().position_m[2]
        config = self.config
        if config.score == PLACE_SCORE:
            assert self.target_source is not None
            target = self.target_source().position_m
            return PlaceScorer(rest_z, (target[0], target[1]), config.lift_m,
                               config.place_radius_m, config.place_height_m,
                               config.gripper_clear_m, config.hold_s)
        return LiftScorer(rest_z, config.lift_m, config.hold_s)

    def run_take(self, index: int) -> TakeRecord:
        """One take: approach, seeded reset, settle, record to the score or the limit."""
        config = self.config
        self.approach(index)
        self.sleep(config.settle_s / 2)
        seed = config.placement_seed + index
        self.randomize(seed, config.randomize_scale)
        self.sleep(config.settle_s)
        scorer = self._scorer()
        executor = self.actions.reset()
        period = 1.0 / config.fps
        self.recorder.start()
        started = self.clock()
        scored_at: float | None = None
        ended_at: float | None = None
        next_due = started
        while True:
            now = self.clock()
            if now < next_due:
                self.sleep(next_due - now)
                continue
            next_due += period
            if not self.recorder.capture_once():
                break  # the hard cap stopped the take
            cup = self.cup_source().position_m
            now = self.clock()
            if isinstance(scorer, PlaceScorer):
                assert self.gripper_source is not None
                done = scorer.feed(cup, self.gripper_source().position_m, now)
            else:
                done = scorer.feed(cup[2], now)
            if done and scored_at is None:
                scored_at = now
            if executor.phase in ("done", "infeasible") and ended_at is None:
                ended_at = now
            finished = (not config.finish_at_rest
                        or (ended_at is not None and now - ended_at >= SCORED_TAIL_S))
            if scored_at is not None and now - scored_at >= SCORED_TAIL_S and finished:
                break
            if ended_at is not None and now - ended_at >= UNSCORED_GRACE_S:
                break
            if now - started >= config.max_take_s:
                break
        if self.recorder.state.name == "RECORDING":
            self.recorder.stop()
        success = scored_at is not None
        episode = self.recorder.mark_success() if success else self.recorder.mark_failure()
        if not success:
            append_exclusion(
                self.dataset_root, episode.episode_index, action="exclude", reason_code="other",
                note=f"staged executor did not reach the score; final phase {executor.phase}",
                operator=OPERATOR,
            )
        record = TakeRecord(
            index=index, episode_index=episode.episode_index, seed=seed, success=success,
            final_phase=executor.phase, feasible=executor.record.feasible,
            steps=self.actions.steps, duration_s=episode.duration_s,
            max_lift_m=scorer.max_lift_m, stage_record=asdict(executor.record),
            clamped_steps=self.actions.clamped,
        )
        self.takes.append(record)
        self.log(f"take {index}: {'success' if success else 'failure'} after "
                 f"{record.duration_s:.1f} s, {record.steps} steps, phase {record.final_phase}, "
                 f"max lift {100 * record.max_lift_m:.1f} cm")
        return record

    def run(self) -> dict[str, Any]:
        """Every take in order, then the summary."""
        for index in range(self.config.takes):
            self.run_take(index)
        successes = sum(1 for take in self.takes if take.success)
        return {
            "kind": "simulation_staged_demonstrations",
            "lane": "simulation",
            "physical_evidence": False,
            "utc": datetime.now(timezone.utc).isoformat(),
            "operator": OPERATOR,
            "dataset_root": str(self.dataset_root),
            "config": asdict(self.config),
            "takes": [asdict(take) for take in self.takes],
            "successes": successes,
            "failures_excluded": len(self.takes) - successes,
            "note": ("Demonstrations driven by the staged executor in the Isaac scene; failures "
                     "are labelled and excluded, successes train; never physical evidence."),
        }


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write the session summary once; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")

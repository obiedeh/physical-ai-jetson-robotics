"""Closed-loop policy trials in the Isaac scene, scored by the cup's own height.

The runner is the simulation lane's counterpart of the physical probe runner. It reads
the sim arm state and the two recorded camera views from the scene's sim topics, asks a
policy server started with ``--lane simulation`` for actions through the ordinary
:class:`~synria_lerobot.policy_client.PolicyClient`, and publishes seven-value joint
targets on a sim-only topic that the scene's policy launch mode mirrors. Every trial
starts with a randomized object reset, an approach from wherever the arm is to a fixed
start pose, and a policy reset; it ends when the cup has been held ``lift_m`` above its
rest height for ``hold_s`` without a break, or at the trial time limit. The score comes
from the scene's published cup pose, not from an operator label, so a data-scaling
study can compare runs on the same rule.

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
SIM_CUP_POSE_TOPIC = "/isaac/pick_objects/cup/pose"
# The real arm's command topics, its feedback topic and the deployment bridge's own name.
REAL_ARM_TOPICS = frozenset({*DEFAULT_COMMAND_TOPICS, "/joint_states", "/policy_arm"})
REPORT_KIND = "simulation_policy_trials"


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

    def __post_init__(self) -> None:
        """Reject a campaign that could not be scored or replayed."""
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        for name in ("command_period_s", "max_trial_s", "lift_m", "hold_s", "settle_s",
                     "approach_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
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
        randomize: Callable[[], None],
        config: TrialConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        """Refuse anything but a simulation contract whose command period matches the config."""
        client.embodiment.contract.require_lane(SIMULATION_LANE)
        client.config.require_command_period(config.command_period_s)
        self.client, self.sink, self.config = client, sink, config
        self.state_source, self.wrist_source, self.front_source = (
            state_source, wrist_source, front_source,
        )
        self.cup_source, self.randomize = cup_source, randomize
        self.clock, self.sleep, self.log = clock, sleep, log
        self.task = client.embodiment.contract.task_text

    def observe(self) -> SynriaObservation:
        """The newest sample of every source, with the trained instruction."""
        return SynriaObservation(
            self.state_source.read(), self.wrist_source.read(), self.front_source.read(),
            self.task,
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
        """One trial: randomized reset, approach, policy reset, steps to a lift or the limit."""
        config = self.config
        self.randomize()
        self.sleep(config.settle_s)
        approach_steps = self.approach()
        self.sleep(config.settle_s / 2)
        rest_z = self.cup_source.read().position_m[2]
        scorer = LiftScorer(rest_z, config.lift_m, config.hold_s)
        holds: Counter[str] = Counter()
        steps = 0
        started = self.clock()
        first = True
        try:
            while self.clock() - started < config.max_trial_s:
                tick = self.clock()
                decision = self.client.step(self.observe(), reset=first)
                first = False
                steps += 1
                if decision.hold or decision.action is None:
                    holds[decision.reason] += 1
                    self.sink.hold()
                else:
                    self.sink.offer(decision.action)
                if scorer.feed(self.cup_source.read().position_m[2], self.clock()):
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
            cup_rest_z_m=rest_z, approach_steps=approach_steps,
        )
        verdict = "success" if success else "failure"
        self.log(f"trial {index}: {verdict} after {duration:.1f} s, {steps} steps, "
                 f"{record.holds} holds, max lift {scorer.max_lift_m * 100:.1f} cm")
        return record

    def run(self, policy: dict[str, Any]) -> dict[str, Any]:
        """Run every trial and return the campaign report; the server must be in the sim lane."""
        if training_lane(policy) != SIMULATION_LANE:
            raise ValueError("the policy server is not serving the simulation lane")
        contract = self.client.embodiment.contract
        trials = [self.run_trial(index) for index in range(self.config.trials)]
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
            "score_rule": (f"cup {self.config.lift_m * 100:.1f} cm above its rest height for "
                           f"{self.config.hold_s:.1f} s without a break, from the scene's "
                           f"published cup pose"),
            "trials": [asdict(trial) for trial in trials],
            "successes": successes,
            "success_rate": successes / len(trials),
            "note": "Simulation lane; scored in the Isaac scene; never physical evidence.",
        }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the campaign report once; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
    """Joint targets on the sim-only target topic and reset requests on the randomize topic.

    The target topic is checked against the real arm's topics before any publisher
    exists. ``hold`` republishes the last target so the scene's drive keeps its place;
    before any target it sends nothing.
    """

    def __init__(
        self, *, target_topic: str = SIM_TARGET_TOPIC, randomize_topic: str = SIM_RANDOMIZE_TOPIC,
        node_name: str = "synria_sim_policy_runner",
    ) -> None:
        """Create both publishers inside an owned ROS context."""
        require_sim_only_topic(target_topic)
        require_sim_only_topic(randomize_topic)
        try:
            import rclpy  # type: ignore[import-not-found]
            from rclpy.context import Context  # type: ignore[import-not-found]
            from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
            from std_msgs.msg import Bool  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional ROS dependency
            raise RuntimeError("ROS 2 publisher dependencies are unavailable") from exc
        self._rclpy = rclpy
        self._JointState, self._Bool = JointState, Bool
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(node_name, context=self._context)
        self._targets = self._node.create_publisher(JointState, target_topic, 10)
        self._randomize = self._node.create_publisher(Bool, randomize_topic, 10)
        self._lock = threading.Lock()
        self._last: tuple[float, ...] | None = None
        self.sent = 0
        self.target_topic, self.randomize_topic = target_topic, randomize_topic
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

    def request_randomize(self) -> None:
        """Ask the scene for a Randomize and reset."""
        message = self._Bool()
        message.data = True
        self._randomize.publish(message)

    def close(self) -> None:
        """Destroy the publishers and release the owned context."""
        if not self._closed:
            _close_all(
                _discard_result(lambda: self._node.destroy_publisher(self._targets)),
                _discard_result(lambda: self._node.destroy_publisher(self._randomize)),
                self._node.destroy_node,
                lambda: self._rclpy.shutdown(context=self._context),
            )
            self._closed = True

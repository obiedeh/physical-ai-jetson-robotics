"""Read-only-first Synria session bindings with explicitly gated command resources."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .embodiment import SynriaObservation
from .evaluation import FUNNEL, FixedTaskGrade, OperatorGrade
from .physical_contract import (
    DIRECT_JOINT_COMMAND_TOPIC,
    DRIVER_JOINT_NAMES,
    ActionSource,
    PhysicalDatasetContract,
    PhysicalState,
    require_absolute_topic,
)
from .policy_client import PolicySafetyConfig
from .quality_gates import load_limits
from .recorder import OpenCVFrameSource, RosJointStateSource, _close_all
from .sessions import OperatorAbort
from .task_registry import TaskDefinition
from .turn_executor import TaskMove

TRAJECTORY_ACTION_TYPE = "control_msgs/action/FollowJointTrajectory"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"


def _load_ros() -> Any:
    import rclpy  # type: ignore[import-not-found]
    from control_msgs.action import FollowJointTrajectory  # type: ignore[import-not-found]
    from rclpy.action import ActionClient  # type: ignore[import-not-found]
    from rclpy.action.graph import (  # type: ignore[import-not-found]
        get_action_server_names_and_types_by_node,
    )
    from rclpy.context import Context  # type: ignore[import-not-found]
    from rclpy.executors import SingleThreadedExecutor  # type: ignore[import-not-found]
    from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
    from std_msgs.msg import Bool  # type: ignore[import-not-found]
    from trajectory_msgs.msg import JointTrajectoryPoint  # type: ignore[import-not-found]

    return SimpleNamespace(
        rclpy=rclpy,
        Context=Context,
        Executor=SingleThreadedExecutor,
        ActionClient=ActionClient,
        action_servers=get_action_server_names_and_types_by_node,
        JointState=JointState,
        Bool=Bool,
        Trajectory=FollowJointTrajectory,
        Point=JointTrajectoryPoint,
    )


def validate_adapter_config(config: dict[str, Any], safety: PolicySafetyConfig) -> None:
    routes = [
        config[name]
        for name in (
            "joint_state_topic",
            "policy_target_topic",
            "armed_topic",
            "arm_action",
            "gripper_action",
        )
    ]
    for route in routes:
        require_absolute_topic(route)
    if len(set(routes)) != len(routes):
        raise ValueError("adapter topic/action roles must be distinct")
    unsafe = config["unsafe_command_topics"]
    if not isinstance(unsafe, list):
        raise ValueError("unsafe command topics must be a list")
    for name in unsafe:
        require_absolute_topic(name)
    if set(routes) & {DIRECT_JOINT_COMMAND_TOPIC, *unsafe}:
        raise ValueError("unsafe command topic cannot be used as an adapter route")
    driver = config["standalone_driver_node"]
    if not isinstance(driver, str) or not driver.strip().strip("/"):
        raise ValueError("standalone driver name is required")
    for name in (
        "bridge_move_time_s",
        "discovery_timeout_s",
        "arming_timeout_s",
        "action_timeout_s",
        "shutdown_timeout_s",
        "max_source_age_s",
        "max_header_delay_s",
        "max_header_future_s",
    ):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if safety.command_period_s < config["bridge_move_time_s"]:
        raise ValueError("command period must be at least the configured bridge move time")
    for name in ("image_width", "image_height"):
        if type(config[name]) is not int or not 0 < config[name] <= 512:
            raise ValueError("stored image dimensions must be positive integers at most 512")
    for name in ("wrist_camera", "front_camera"):
        if not isinstance(config[name], str) or not config[name].startswith("/dev/v4l/by-id/"):
            raise ValueError("both cameras require explicit stable by-id paths")
    if config["wrist_camera"] == config["front_camera"]:
        raise ValueError("wrist and front camera sources must be distinct")
    if not isinstance(config["gripper_mapping"], dict):
        raise ValueError("explicit gripper mapping configuration required")


def gripper_target(mapping: dict[str, Any], position_m: float, stroke_m: float) -> float:
    """Operator-verified affine conversion; no simulated finger mapping is inferred."""
    if any(
        not isinstance(mapping.get(key), str) or not mapping[key].strip()
        for key in ("verified_by", "verified_on", "joint_name")
    ):
        raise ValueError("operator-verified gripper joint mapping required")
    if mapping.get("units") not in {"m", "rad"}:
        raise ValueError("verified gripper action units must be m or rad")
    bounds: list[float] = []
    for key in ("open_position", "closed_position"):
        value = mapping.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("finite gripper action endpoints required")
        bounds.append(float(value))
    opened, closed = bounds
    if opened == closed or not math.isfinite(closed - opened):
        raise ValueError("gripper action endpoints must differ")
    if (
        type(stroke_m) not in (int, float)
        or not math.isfinite(stroke_m)
        or stroke_m <= 0
        or type(position_m) not in (int, float)
        or not math.isfinite(position_m)
        or not 0 <= position_m <= stroke_m
    ):
        raise ValueError("gripper state outside the physical contract")
    fraction = position_m / stroke_m
    target = float(opened * (1 - fraction) + closed * fraction)
    if not math.isfinite(target) or not min(bounds) <= target <= max(bounds):
        raise ValueError("gripper conversion exceeds its verified endpoints")
    return target


class RosSessionIO:
    """Never starts drivers, arms a bridge, or publishes direct follower commands."""

    def __init__(
        self,
        session: dict[str, Any],
        *,
        ros: Any = None,
        state_factory: Callable[..., Any] = RosJointStateSource,
        frame_factory: Callable[..., Any] = OpenCVFrameSource,
        input_fn: Callable[[str], str] = input,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = json.loads(Path(session["adapter_config"]).read_text(encoding="utf-8"))
        self.safety = PolicySafetyConfig.load(
            Path(session["policy_config"]),
            command_period_s=session["command_period_s"],
            response_timeout_s=session["response_timeout_s"],
        )
        validate_adapter_config(self.settings, self.safety)
        self.contract = PhysicalDatasetContract(
            session["gripper_type"],
            ActionSource(session["action_source"]),
            session["state_has_velocity"],
            action_lookahead_steps=session["action_lookahead_steps"],
            task_id=session["task_id"],
            task_definition=TaskDefinition.from_metadata(session),
            recording_purpose=session["recording_purpose"],
        )
        self.contract.require_qualifying()
        self.limits = load_limits(Path(session["limits"]))
        self.operator = session["provenance"]["operator"]
        self.output = Path(session["session_output"])
        self.enable_motion = session["enable_motion"] is True
        self.input, self.clock, self.sleep = input_fn, clock, sleep
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._armed: bool | None = None
        self._armed_sequence = 0
        self._armed_stamp = 0.0
        self._sync_sequence = 0
        self._sync_stamp: float | None = None
        self._authorized = self._closed = self._aborted = self._hold_sent = False
        self._publisher: Any = None
        self._gripper: Any = None
        self._context: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._subscription: Any = None
        self._goal: Any = None
        self._goal_future: Any = None
        self._result_future: Any = None
        self._cancel_future: Any = None
        self._cancel_requested = False
        self._cancel_started = self._cancel_confirmed = False
        self._ros_destroyed = self._sources_closed = False
        self._owned_gid: bytes | None = None
        self._sources: list[Any] = []
        self.wrist: Any
        self.front: Any
        self._last_offer: float | None = None
        self._still_index = 0
        self._evidence: dict[str, Any] = {"events": []}
        self.ros = _load_ros() if ros is None else ros
        with ExitStack() as startup:
            startup.callback(self.close)
            self._context = self.ros.Context()
            self.ros.rclpy.init(context=self._context)
            self._name = "synria_session_" + uuid.uuid4().hex
            self._node = self.ros.rclpy.create_node(
                self._name, context=self._context, use_global_arguments=False
            )
            self._executor = self.ros.Executor(context=self._context)
            self._subscription = self._node.create_subscription(
                self.ros.Bool, self.settings["armed_topic"], self._on_armed, 1
            )
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, daemon=True, name=self._name)
            self._thread.start()
            self._graph()
            self.state = state_factory(
                self.settings["joint_state_topic"],
                node_name=self._name + "_state",
                state_has_velocity=self.contract.state_has_velocity,
            )
            self._sources.append(self.state)
            for camera in ("wrist", "front"):
                source = frame_factory(
                    self.settings[camera + "_camera"],
                    width=self.settings["image_width"],
                    height=self.settings["image_height"],
                )
                setattr(self, camera, source)
                self._sources.append(source)
            startup.pop_all()

    def _spin(self) -> None:
        try:
            self._executor.spin()
            if not self._stop.is_set():
                raise RuntimeError("adapter executor stopped unexpectedly")
        except BaseException as error:
            if not self._stop.is_set():
                with self._lock:
                    self._error = error

    def _on_armed(self, message: Any) -> None:
        with self._lock:
            if type(message.data) is not bool:
                self._error = ValueError("armed state must contain an actual boolean")
                self._armed = None
                return
            self._armed = message.data
            self._armed_sequence += 1
            self._armed_stamp = self.clock()

    def _health(self) -> None:
        with self._lock:
            if self._closed or self._error is not None:
                raise RuntimeError(f"adapter is closed or failed: {self._error}")

    def _endpoints(self, topic: str) -> list[Any]:
        endpoints = self._node.get_publishers_info_by_topic(topic)
        count = self._node.count_publishers(topic)
        if type(count) is not int or count < 0 or count != len(endpoints):
            raise RuntimeError(f"inconsistent publisher discovery for {topic}")
        return list(endpoints)

    def _graph(self, *, pin: bool = False) -> bool:
        self._health()
        nodes = self._node.get_node_names_and_namespaces()
        driver = self.settings["standalone_driver_node"].rstrip("/").split("/")[-1]
        if any(name == driver for name, _ in nodes):
            raise RuntimeError("standalone driver detected; ros2_control-only session required")
        if sum(name == self._name for name, _ in nodes) != 1:
            raise RuntimeError("adapter node identity is missing or ambiguous")
        servers: dict[str, list[Any]] = {}
        for name, namespace in nodes:
            for action, types in self.ros.action_servers(self._node, name, namespace):
                if TRAJECTORY_ACTION_TYPE in types:
                    servers.setdefault(action, []).append((name, namespace))
        if any(
            len(servers.get(self.settings[key], [])) != 1
            for key in ("arm_action", "gripper_action")
        ):
            raise RuntimeError("both unique arm and gripper trajectory action servers are required")
        for topic in {DIRECT_JOINT_COMMAND_TOPIC, *self.settings["unsafe_command_topics"]}:
            if self._endpoints(topic):
                raise RuntimeError(f"unsafe command publisher detected on {topic}")
        endpoints = self._endpoints(self.settings["policy_target_topic"])
        if self._publisher is None:
            if endpoints:
                raise RuntimeError("another policy-target publisher is present")
        else:
            if pin and not endpoints:
                return False
            if len(endpoints) != 1:
                raise RuntimeError("policy publisher ownership is missing or ambiguous")
            endpoint = endpoints[0]
            gid = bytes(endpoint.endpoint_gid)
            if (
                endpoint.node_name != self._name
                or endpoint.node_namespace != self._node.get_namespace()
                or endpoint.topic_type != JOINT_STATE_TYPE
                or not gid
                or not any(gid)
            ):
                raise RuntimeError("policy publisher is not the owned endpoint")
            if pin and self._owned_gid is None:
                self._owned_gid = gid
            elif self._owned_gid != gid:
                raise RuntimeError("policy publisher endpoint identity changed")
        self._health()
        return True

    def _fresh_state(self) -> PhysicalState:
        state = self.contract.prepare_state(self.state.read())
        now = self.clock()
        if not 0 <= now - state.monotonic_timestamp_s <= self.settings["max_source_age_s"]:
            raise RuntimeError("follower state is stale or from the future")
        arrival = state.ros_arrival_stamp_s
        if (
            arrival is None
            or not -self.settings["max_header_future_s"]
            <= (arrival - state.ros_header_stamp_s)
            <= self.settings["max_header_delay_s"]
        ):
            raise RuntimeError("follower ROS header is stale or missing arrival evidence")
        current = (*state.joint_positions_rad, state.gripper_m)
        bounds = (
            *self.limits.joint_limits_rad,
            self.limits.gripper_limits_m[self.contract.gripper_type],
        )
        if self.limits.joint_names != DRIVER_JOINT_NAMES or any(
            not low <= value <= high for value, (low, high) in zip(current, bounds, strict=True)
        ):
            raise RuntimeError("follower state or joint ordering is outside configured limits")
        return state

    def observe(self, task: str) -> SynriaObservation:
        self._health()
        state = self._fresh_state()
        wrist, front = self.wrist.read(), self.front.read()
        now = self.clock()
        if any(
            not 0 <= now - sample.monotonic_timestamp_s <= self.settings["max_source_age_s"]
            for sample in (state, wrist, front)
        ):
            raise RuntimeError("observation source is stale or from the future")
        return SynriaObservation(state, wrist, front, task)

    def preflight(self) -> dict[str, Any]:
        self._graph()
        measurement = self.state.measure_rate()
        self.observe("read-only preflight")
        deadline = self.clock() + self.settings["discovery_timeout_s"]
        while self._armed is None and self.clock() < deadline:
            self._health()
            self.sleep(0.01)
        if self._armed is not False:
            raise RuntimeError("preflight requires an observed disarmed bridge state")
        self._graph()
        with self._lock:
            if self._armed is not False:
                raise RuntimeError("bridge armed during read-only preflight")
        self._evidence.update(
            state_source="ros2_control",
            state_source_identity="operator-configured and graph-checked",
            joint_state_topic=self.settings["joint_state_topic"],
            state_rate_measurement=asdict(measurement),
            adapter_config=self.settings,
            command_period_s=self.safety.command_period_s,
            bridge_move_time_s=self.settings["bridge_move_time_s"],
            action_lookahead_steps=self.contract.action_lookahead_steps,
            initial_bridge_armed=False,
            status="implemented, unmeasured",
        )
        return dict(self._evidence)

    def _prompt(self, text: str) -> str:
        try:
            answer = self.input(text).strip()
        except (EOFError, KeyboardInterrupt) as error:
            self._aborted = True
            _close_all(self.hold, self._cancel_owned_goal)
            raise OperatorAbort("operator input closed or interrupted") from error
        if answer.lower() == "abort":
            self._aborted = True
            _close_all(self.hold, self._cancel_owned_goal)
            raise OperatorAbort("operator aborted session")
        return answer

    def authorize_motion(self) -> dict[str, Any]:
        if not self.enable_motion or not self.limits.verified or not self.safety.verified:
            raise ValueError("motion requires opt-in and both verified safety configurations")
        gripper_target(self.settings["gripper_mapping"], 0, self.contract.gripper_stroke_m)
        if self._armed is not False or "initial_bridge_armed" not in self._evidence:
            raise RuntimeError("complete disarmed read-only preflight before authorization")
        if (
            self._prompt("Confirm leader hardware sync OFF by typing sync-off (or abort): ")
            != "sync-off"
        ):
            raise RuntimeError("leader sync OFF was not confirmed")
        with self._lock:
            if self._armed is not False:
                raise RuntimeError("bridge armed before sync-OFF confirmation")
            self._sync_stamp = self.clock()
            self._sync_sequence = self._armed_sequence
        self._evidence["leader_sync_off"] = {
            "operator": self.operator,
            "monotonic_s": self._sync_stamp,
            "utc": datetime.now(timezone.utc).isoformat(),
            "operator_confirmed": True,
        }
        if self._prompt("Manually arm the bridge, then type armed (or abort): ") != "armed":
            raise RuntimeError("manual arming was not confirmed")
        deadline = self.clock() + self.settings["arming_timeout_s"]
        while self.clock() < deadline:
            self._graph()
            with self._lock:
                fresh = self._armed is True and self._armed_sequence > self._sync_sequence
                fresh = fresh and self._armed_stamp >= self._sync_stamp
            if fresh:
                self._authorized = True
                self._evidence["manual_arming_monotonic_s"] = self._armed_stamp
                return dict(self._evidence)
            self.sleep(0.01)
        raise TimeoutError("no fresh manual bridge-arming transition observed")

    def _check_motion_locked(self) -> None:
        if (
            self._closed
            or self._error is not None
            or not self.enable_motion
            or not self._authorized
            or self._sync_stamp is None
            or not self.limits.verified
            or not self.safety.verified
            or self._armed is not True
        ):
            raise RuntimeError("motion gate is not eligible")

    def _eligible(self) -> PhysicalState:
        with self._lock:
            self._check_motion_locked()
        self._graph()
        gripper_target(self.settings["gripper_mapping"], 0, self.contract.gripper_stroke_m)
        state = self._fresh_state()
        with self._lock:
            self._check_motion_locked()
        return state

    def command_sink(self) -> RosSessionIO:
        self._eligible()
        if self._publisher is not None:
            if self._gripper is None or self._aborted:
                raise RuntimeError("command resource startup was incomplete")
            return self
        self._publisher = self._node.create_publisher(
            self.ros.JointState, self.settings["policy_target_topic"], 1
        )
        if self._publisher.topic_name != self.settings["policy_target_topic"]:
            raise RuntimeError("policy publisher topic was unexpectedly remapped")
        deadline = self.clock() + self.settings["discovery_timeout_s"]
        while not self._graph(pin=True):
            if self.clock() >= deadline:
                raise TimeoutError("owned policy publisher discovery timed out")
            self.sleep(0.01)
        self._eligible()
        self._gripper = self.ros.ActionClient(
            self._node, self.ros.Trajectory, self.settings["gripper_action"]
        )
        if not self._gripper.server_is_ready():
            raise RuntimeError("configured gripper action server is not ready")
        return self

    def _publish_arm(self, positions: tuple[float, ...]) -> None:
        message = self.ros.JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = list(DRIVER_JOINT_NAMES)
        message.position = list(positions)
        with self._lock:
            self._check_motion_locked()
            self._publisher.publish(message)
            # A slow graph lookup or publish must not shorten the next command interval.
            self._last_offer = self.clock()

    def _wait(self, future: Any, timeout_s: float, *, motion: bool) -> Any:
        deadline = self.clock() + timeout_s
        while not future.done():
            if motion:
                self._eligible()
            if self.clock() >= deadline:
                raise TimeoutError("gripper action wait timed out")
            self.sleep(min(0.01, deadline - self.clock()))
        if motion:
            self._eligible()
        error = future.exception()
        if error is not None:
            raise error
        return future.result()

    def _goal_ready(self, future: Any) -> None:
        try:
            with self._lock:
                if future is not self._goal_future:
                    return
            handle = future.result()
            if handle is not None and handle.accepted:
                with self._lock:
                    if future is not self._goal_future:
                        return
                    self._goal = handle
                    cancel = self._cancel_requested
                if cancel:
                    self._request_cancel(future, handle)
        except BaseException as error:
            with self._lock:
                if future is self._goal_future:
                    self._error = error

    def _request_cancel(self, owner: Any, handle: Any) -> None:
        with self._lock:
            if owner is not self._goal_future or handle is not self._goal or self._cancel_started:
                return
            self._cancel_started = True
        # Do not hold our lock across a ROS call or an immediately invoked callback.
        future = handle.cancel_goal_async()
        with self._lock:
            if owner is self._goal_future:
                self._cancel_future = future

    def _terminal_goal(self, handle: Any) -> bool:
        terminal = getattr(handle, "status", None) in {4, 5, 6}
        result = self._result_future
        if result is not None and result.done() and result.exception() is None:
            terminal = terminal or result.result().status in {4, 5, 6}
        return terminal

    def _cancel_owned_goal(self) -> None:
        with self._lock:
            self._cancel_requested = True
        if self._goal_future is None:
            return
        timeout = self.settings["action_timeout_s"]
        handle = self._wait(self._goal_future, timeout, motion=False)
        if handle is None or not handle.accepted:
            with self._lock:
                self._goal_future = None
            return
        if self._terminal_goal(handle):
            with self._lock:
                self._goal = self._goal_future = self._result_future = None
            return
        self._goal_ready(self._goal_future)
        deadline = self.clock() + timeout
        while self._cancel_future is None and self.clock() < deadline:
            if self._error is not None:
                raise RuntimeError("owned gripper cancellation failed") from self._error
            self.sleep(0.01)
        if self._cancel_future is None:
            raise TimeoutError("owned gripper cancellation was not started")
        result = self._wait(self._cancel_future, timeout, motion=False)
        if result.return_code != 0 and not self._terminal_goal(handle):
            raise RuntimeError("owned gripper cancellation was not accepted")
        self._cancel_confirmed = True

    def _clamp(self, action: tuple[float, ...], state: PhysicalState) -> tuple[float, ...]:
        current = (*state.joint_positions_rad, state.gripper_m)
        bounds = (
            *self.limits.joint_limits_rad,
            self.limits.gripper_limits_m[self.contract.gripper_type],
        )
        return tuple(
            max(low, value - delta, min(high, value + delta, desired))
            for desired, value, delta, (low, high) in zip(
                action, current, self.safety.max_delta, bounds, strict=True
            )
        )

    def offer(self, action: tuple[float, ...]) -> None:
        try:
            self._eligible()
            if self._publisher is None or self._gripper is None or self._aborted:
                raise RuntimeError("command sink is unavailable or session aborted")
            now = self.clock()
            if (
                self._last_offer is not None
                and now - self._last_offer < self.safety.command_period_s
            ):
                raise RuntimeError("command offer arrived before the configured period")
            if len(action) != 7 or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in action
            ):
                raise ValueError("seven finite nonboolean command values required")
            goal = self.ros.Trajectory.Goal()
            goal.trajectory.joint_names = [self.settings["gripper_mapping"]["joint_name"]]
            point = self.ros.Point()
            duration = self.settings["bridge_move_time_s"]
            point.time_from_start.sec = int(duration)
            point.time_from_start.nanosec = int((duration - int(duration)) * 1_000_000_000)
            goal.trajectory.points = [point]
            target = self._clamp(action, self._eligible())
            point.positions = [
                gripper_target(
                    self.settings["gripper_mapping"], target[6], self.contract.gripper_stroke_m
                )
            ]
            self._hold_sent = False
            self._publish_arm(target[:6])
            # Arm publication and graph checks can take time; use the newest gripper state.
            gripper = self._clamp(action, self._eligible())[6]
            point.positions = [
                gripper_target(
                    self.settings["gripper_mapping"], gripper, self.contract.gripper_stroke_m
                )
            ]
            submit = self._gripper.send_goal_async
            with self._lock:
                self._check_motion_locked()
                future = submit(goal)
                self._goal = self._cancel_future = self._result_future = None
                self._cancel_requested = self._cancel_started = self._cancel_confirmed = False
                self._goal_future = future
            self._goal_future.add_done_callback(self._goal_ready)
            handle = self._wait(self._goal_future, self.settings["action_timeout_s"], motion=True)
            if not handle.accepted:
                raise RuntimeError("gripper goal was rejected")
            self._result_future = handle.get_result_async()
            result = self._wait(self._result_future, self.settings["action_timeout_s"], motion=True)
            if result.status in {4, 5, 6}:
                with self._lock:
                    self._goal = self._goal_future = self._result_future = None
            if result.status != 4 or result.result.error_code != 0:
                raise RuntimeError("gripper action did not succeed")
            with self._lock:
                self._goal = self._goal_future = None
        except BaseException:
            self._aborted = True
            _close_all(self.hold, self._cancel_owned_goal)
            raise

    def hold(self) -> None:
        if self._hold_sent or self._publisher is None:
            return
        try:
            state = self._eligible()
        except Exception as error:
            self._evidence["events"].append({"hold_refused": str(error)})
            return
        self._hold_sent = True
        self._publish_arm(state.joint_positions_rad)
        self._evidence["events"].append({"measured_arm_hold_monotonic_s": self.clock()})

    def completed(self) -> bool:
        return self._prompt("Motion ended? [yes/no/abort]: ") == "yes"

    def roll_completed(self, phase: str) -> bool:
        return self._prompt(f"{phase} motion ended? [yes/no/abort]: ") == "yes"

    def capture_front(self, *, require_native: bool = False) -> tuple[Any, str, float]:
        frame = self.observe("camera evidence").front
        image = frame.native_data
        native = image is not None
        if native:
            if (
                not isinstance(image, np.ndarray)
                or frame.native_resolution is None
                or not isinstance(frame.source_id, str)
                or not frame.source_id.strip()
                or frame.source_id != self.settings["front_camera"]
                or len(frame.native_resolution) != 2
                or any(type(side) is not int or side <= 0 for side in frame.native_resolution)
                or image.shape != (frame.native_resolution[1], frame.native_resolution[0], 3)
            ):
                raise ValueError("native camera dimensions and source identity are required")
        elif require_native:
            raise ValueError("native front-camera pixels are unavailable; no upscaling permitted")
        else:
            image = frame.data
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or (image.shape[2] != 3)
        ):
            raise ValueError("camera evidence must be RGB uint8")
        self._still_index += 1
        destination = self.output / "stills" / f"front_{self._still_index:06d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        encoded, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not encoded:
            raise RuntimeError("front-camera still encoding failed")
        with destination.open("xb") as stream:
            stream.write(buffer.tobytes())
        self._evidence["events"].append(
            {
                "front_still": str(destination),
                "source_monotonic_s": frame.monotonic_timestamp_s,
                "resolution": [int(image.shape[1]), int(image.shape[0])],
                "source_id": frame.source_id,
                "native_resolution_confirmed": native,
            }
        )
        # Legacy perception consumes stored-size pixels; the saved evidence can be native.
        return (
            image if require_native else frame.data,
            str(destination),
            frame.monotonic_timestamp_s,
        )

    def grade_skill(self, task: TaskDefinition) -> FixedTaskGrade:
        self.hold()
        image, still, stamp = self.capture_front(require_native=True)
        label = self._prompt(
            f"{task.task_id}: {task.success_rule} {still} [success/failure/abort]: "
        )
        grade = FixedTaskGrade(
            label,
            self.operator,
            still,
            stamp,
            task,
            (int(image.shape[1]), int(image.shape[0])),
        )
        grade.evidence()
        return grade

    def confirm_skill(self, task: TaskDefinition) -> bool:
        self.hold()
        return self._prompt(
            f"Ready to run {task.task_id}: {task.task_text}; "
            "fixed scene unchanged? [yes/no/abort]: "
        ) == "yes"

    def grade(self, move: TaskMove) -> OperatorGrade:
        _, still, stamp = self.capture_front()
        label = self._prompt(f"{move.source} to {move.target}, {still}: [success/failure/abort]: ")
        reached = int(self._prompt("Highest confirmed funnel stage, 0 to 5 (or abort): "))
        if not 0 <= reached <= len(FUNNEL):
            raise ValueError("funnel stage must be 0 through 5")
        grade = OperatorGrade(
            label, self.operator, still, stamp, *(index < reached for index in range(len(FUNNEL)))
        )
        grade.evidence()
        return grade

    def recover(self, move: TaskMove) -> bool:
        return self._prompt("Scene reset confirmed? [yes/no/abort]: ") == "yes"

    def recover_skill(self, task: TaskDefinition) -> bool:
        self.hold()
        return (
            self._prompt(f"Reset {task.task_id} fixed scene confirmed? [yes/no/abort]: ") == "yes"
        )

    def operator_roll(self, *, still: str | None = None) -> int:
        value = int(self._prompt(f"Operator die value, 1 to 6 (or abort), {still or 'scene'}: "))
        if not 1 <= value <= 6:
            raise ValueError("die value must be 1 through 6")
        return value

    def abort_pending(self) -> bool:
        self._health()
        return self._aborted

    def abort_requested(self) -> bool:
        if self._aborted:
            return True
        try:
            return self._prompt("Continue session? [continue/abort]: ") != "continue"
        except RuntimeError:
            return True

    def power_state_end(self) -> str:
        return (
            "not recorded after abort"
            if self._aborted
            else self._prompt("Observed end power state: ")
        )

    def session_evidence(self) -> dict[str, Any]:
        return dict(self._evidence)

    def _close_ros(self) -> None:
        if self._ros_destroyed:
            return
        if self._goal_future is not None and not self._cancel_confirmed:
            raise RuntimeError(
                "gripper outcome unresolved; ROS resources retained for cancellation"
            )
        self._stop.set()
        shutdown_error = None
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=self.settings["shutdown_timeout_s"])
            except BaseException as error:
                shutdown_error = error
        if self._thread is not None:
            self._thread.join(self.settings["shutdown_timeout_s"])
            if self._thread.is_alive():
                raise RuntimeError("adapter worker did not stop; ROS resources retained")
        callbacks = []
        if self._gripper is not None:
            callbacks.append(self._gripper.destroy)
        if self._publisher is not None:
            callbacks.append(lambda: self._node.destroy_publisher(self._publisher))
        if self._node is not None:
            callbacks.append(self._node.destroy_node)
        if self._context is not None:
            callbacks.append(lambda: self.ros.rclpy.shutdown(context=self._context))
        _close_all(*callbacks)
        self._ros_destroyed = True
        if shutdown_error is not None:
            raise RuntimeError("executor shutdown reported an error") from shutdown_error

    def _close_sources(self) -> None:
        if not self._sources_closed:
            failed = []
            for source in self._sources:
                try:
                    source.close()
                except BaseException as error:
                    failed.append((source, error))
            self._sources = [source for source, _ in failed]
            self._sources_closed = not failed
            if failed:
                raise RuntimeError("recording source cleanup failed; retry required") from failed[
                    0
                ][1]

    def close(self) -> None:
        if self._closed and self._ros_destroyed and self._sources_closed:
            return
        try:
            _close_all(self.hold, self._cancel_owned_goal, self._close_sources, self._close_ros)
        finally:
            self._closed = True


def create_session_io(config: dict[str, Any]) -> RosSessionIO:
    return RosSessionIO(config)

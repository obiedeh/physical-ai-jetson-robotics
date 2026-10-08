from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
from test_task_registry import synthetic_task

from synria_lerobot.evaluation import protocol_digest
from synria_lerobot.physical_contract import ImageFrame, PhysicalState, StateRateMeasurement
from synria_lerobot.policy_client import FakePolicy, PolicyClient
from synria_lerobot.ros_adapter import (
    JOINT_STATE_TYPE,
    TRAJECTORY_ACTION_TYPE,
    RosSessionIO,
    gripper_target,
)
from synria_lerobot.sessions import run_session


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.on_sleep = lambda: None

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration
        self.on_sleep()


class Future:
    def __init__(
        self, value: Any = None, *, done: bool = True, defer_callbacks: bool = False
    ) -> None:
        self.value, self.ready = value, done
        self.defer_callbacks = defer_callbacks
        self.callbacks: list[Any] = []

    def done(self) -> bool:
        return self.ready

    def exception(self) -> None:
        return None

    def result(self) -> Any:
        return self.value

    def add_done_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)
        if self.ready and not self.defer_callbacks:
            callback(self)

    def finish(self) -> None:
        self.ready = True
        if not self.defer_callbacks:
            self.deliver_callbacks()

    def deliver_callbacks(self) -> None:
        for callback in self.callbacks:
            callback(self)


class FakeROS:
    """Only in-memory graph/messages/futures; never imports the ROS runtime."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.closed: list[str] = []
        self.published: list[Any] = []
        self.goals: list[Any] = []
        self.publisher_topics: list[str] = []
        self.action_clients: list[str] = []
        self.nodes = [("controller", "/")]
        self.servers = [
            "/fake_arm/follow_joint_trajectory",
            "/fake_gripper/follow_joint_trajectory",
        ]
        self.initial_armed: Any = False
        self.armed_callback: Any = None
        self.external: dict[str, list[Any]] = {}
        self.publisher: Any = None
        self.query_hook = lambda: None
        self.bad_count: Any = None
        self.cancel_count = 0
        self.cancelled_goals: list[int] = []
        self.defer_callbacks = False
        self.goal_accepted = True
        self.result_status = 4
        self.result_code = 0
        self.ack_delayed = self.result_delayed = False
        self.cancel_code = 0
        self.gripper_ready = True
        self.stop = threading.Event()
        self.executor = NS(add_node=lambda node: None, spin=self.stop.wait, shutdown=self.shutdown)
        self.Context = object
        self.Executor = lambda **kwargs: self.executor
        self.rclpy = NS(
            init=lambda **kwargs: None,
            create_node=self.create_node,
            shutdown=lambda **kwargs: self.closed.append("context"),
        )
        self.JointState = lambda: NS(header=NS(stamp=None), name=[], position=[])
        self.Bool = object
        self.Trajectory = NS(Goal=lambda: NS(trajectory=NS(joint_names=[], points=[])))
        self.Point = lambda: NS(positions=[], time_from_start=NS(sec=0, nanosec=0))
        self.ActionClient = self.action_client

    def shutdown(self, **kwargs: Any) -> bool:
        self.closed.append("executor")
        self.stop.set()
        return True

    def create_node(self, name: str, **kwargs: Any) -> FakeROS:
        assert kwargs["use_global_arguments"] is False
        self.name = name
        self.nodes.insert(0, (name, "/"))
        return self

    def create_subscription(self, kind: Any, topic: str, callback: Any, depth: int) -> object:
        assert kind is self.Bool and topic == "/fake_armed" and depth == 1
        self.armed_callback = callback
        if self.initial_armed is not None:
            callback(NS(data=self.initial_armed))
        return object()

    def emit_armed(self, value: Any) -> None:
        self.armed_callback(NS(data=value))

    def get_namespace(self) -> str:
        return "/"

    def get_node_names_and_namespaces(self) -> list:
        self.query_hook()
        return self.nodes

    def action_servers(self, node: Any, name: str, namespace: str) -> list:
        return (
            [(value, [TRAJECTORY_ACTION_TYPE]) for value in self.servers]
            if name == "controller"
            else []
        )

    def get_publishers_info_by_topic(self, topic: str) -> list:
        result = list(self.external.get(topic, []))
        if self.publisher is not None and self.publisher.topic_name == topic:
            result.append(self.publisher.endpoint)
        return result

    def count_publishers(self, topic: str) -> Any:
        return (
            len(self.get_publishers_info_by_topic(topic))
            if self.bad_count is None
            else self.bad_count
        )

    def create_publisher(self, kind: Any, topic: str, depth: int) -> Any:
        assert kind is self.JointState and topic == "/fake_policy_targets" and depth == 1
        self.publisher_topics.append(topic)
        self.publisher = NS(
            topic_name=topic,
            publish=self.published.append,
            endpoint=NS(
                node_name=self.name,
                node_namespace="/",
                topic_type=JOINT_STATE_TYPE,
                endpoint_gid=b"owned",
            ),
        )
        return self.publisher

    def get_clock(self) -> Any:
        return NS(now=lambda: NS(to_msg=lambda: NS(sec=int(self.clock()), nanosec=0)))

    def action_client(self, node: Any, kind: Any, name: str) -> Any:
        assert name == "/fake_gripper/follow_joint_trajectory" and kind is self.Trajectory
        self.action_clients.append(name)
        return NS(
            server_is_ready=lambda: self.gripper_ready,
            send_goal_async=self.send_goal,
            destroy=lambda: self.closed.append("gripper"),
        )

    def send_goal(self, goal: Any) -> Future:
        self.goals.append(goal)
        result = NS(status=self.result_status, result=NS(error_code=self.result_code))
        self.result_future = Future(result, done=not self.result_delayed)
        index = len(self.goals)
        self.handle = NS(
            accepted=self.goal_accepted,
            get_result_async=lambda: self.result_future,
            cancel_goal_async=lambda: self.cancel(index),
        )
        self.ack_future = Future(
            self.handle, done=not self.ack_delayed, defer_callbacks=self.defer_callbacks
        )
        return self.ack_future

    def cancel(self, index: int) -> Future:
        self.cancel_count += 1
        self.cancelled_goals.append(index)
        return Future(NS(return_code=self.cancel_code))

    def destroy_publisher(self, publisher: Any) -> None:
        self.closed.append("publisher")

    def destroy_node(self) -> None:
        self.closed.append("node")


class Sources:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.state_age = self.image_age = self.header_delay = 0.0
        self.position = 0.0
        self.gripper = 0.0
        self.velocities = (1.0,) * 6

    def state(self, topic: str, **kwargs: Any) -> Any:
        self.opened.append(topic)
        assert kwargs["state_has_velocity"] is False
        return NS(
            read=self.read_state,
            measure_rate=self.measure_rate,
            close=lambda: self.closed.append(topic),
        )

    def read_state(self) -> PhysicalState:
        return PhysicalState(
            (self.position,) * 6,
            self.gripper,
            self.clock() - self.state_age,
            1_800_000_000.0 - self.header_delay,
            self.velocities,
            ros_arrival_stamp_s=1_800_000_000.0,
        )

    def measure_rate(self) -> StateRateMeasurement:
        return StateRateMeasurement(30, 60, 2, self.clock() - 2, self.clock(), 1 / 30)

    def camera(self, device: str, *, width: int, height: int) -> Any:
        self.opened.append(device)
        return NS(
            read=lambda: ImageFrame(
                np.full((height, width, 3), 120, np.uint8), self.clock() - self.image_age
            ),
            close=lambda: self.closed.append(device),
        )


@pytest.fixture
def setup(tmp_path: Path) -> Any:
    config = json.loads(Path("config/synria_session.json").read_text())
    for key in ("limits", "policy_config"):
        content = json.loads(Path(config[key]).read_text())
        content.update(verified_by="synthetic fixture", verified_on="2026-10-08")
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(content))
        config[key] = str(path)
    settings = json.loads(Path("config/synria_ros_adapter.json").read_text())
    settings.update(
        arm_action="/fake_arm/follow_joint_trajectory",
        gripper_action="/fake_gripper/follow_joint_trajectory",
        armed_topic="/fake_armed",
        policy_target_topic="/fake_policy_targets",
        wrist_camera="/dev/v4l/by-id/fake-wrist",
        front_camera="/dev/v4l/by-id/fake-front",
        image_width=8,
        image_height=6,
        discovery_timeout_s=0.03,
        arming_timeout_s=0.03,
        action_timeout_s=0.03,
        shutdown_timeout_s=0.05,
        gripper_mapping=dict(
            joint_name="synthetic_gripper",
            units="m",
            open_position=0.05,
            closed_position=0.0,
            verified_by="synthetic fixture",
            verified_on="2026-10-08",
        ),
    )
    config.update(synthetic_task(20, 30).metadata())
    config.update(
        adapter_config=str(tmp_path / "adapter.json"),
        gripper_type="50mm",
        action_lookahead_steps=2,
        command_period_s=0.4,
        response_timeout_s=0.1,
        session_output=str(tmp_path / "session"),
        enable_motion=True,
    )
    config["provenance"]["operator"] = "synthetic fixture"
    clock = Clock()
    ros, sources = FakeROS(clock), Sources(clock)
    answers: list[str] = []

    def input_fn(prompt: str) -> str:
        answer = answers.pop(0)
        if answer == "armed":
            ros.emit_armed(True)
        return answer

    def create() -> RosSessionIO:
        Path(config["adapter_config"]).write_text(json.dumps(settings))
        return RosSessionIO(
            config,
            ros=ros,
            state_factory=sources.state,
            frame_factory=sources.camera,
            input_fn=input_fn,
            clock=clock,
            sleep=clock.sleep,
        )

    return NS(
        config=config,
        settings=settings,
        clock=clock,
        ros=ros,
        sources=sources,
        answers=answers,
        create=create,
    )


def ready(setup: Any) -> RosSessionIO:
    io = setup.create()
    io.preflight()
    setup.answers.extend(["sync-off", "armed"])
    io.authorize_motion()
    return io.command_sink()


def test_read_only_preflight_has_no_command_resources_and_drops_unused_velocity(setup: Any) -> None:
    setup.config["enable_motion"] = False
    io = setup.create()
    evidence = io.preflight()
    observation = io.observe("fake task")
    assert observation.state.joint_velocities_rad_s is None
    assert observation.front.data.shape == (6, 8, 3)
    assert evidence["state_rate_measurement"]["rate_hz"] == 30
    assert evidence["command_period_s"] == evidence["bridge_move_time_s"] == 0.4
    assert evidence["action_lookahead_steps"] == 2
    with pytest.raises(ValueError, match="opt-in"):
        io.authorize_motion()
    io.close()
    assert not setup.ros.publisher_topics and not setup.ros.action_clients
    assert setup.sources.closed == setup.sources.opened
    assert setup.ros.closed == ["executor", "node", "context"]


@pytest.mark.parametrize(
    "mutation", ["arm", "gripper", "driver", "policy", "direct", "custom", "count", "exception"]
)
def test_graph_preflight_refuses_without_opening_sources(setup: Any, mutation: str) -> None:
    if mutation in {"arm", "gripper"}:
        setup.ros.servers.remove(setup.settings[mutation + "_action"])
    elif mutation == "driver":
        setup.ros.nodes.append(("alicia_d_driver_node", "/unexpected_namespace"))
    elif mutation in {"policy", "direct", "custom"}:
        topic = {
            "policy": "/fake_policy_targets",
            "direct": "/joint_commands",
            "custom": "/unsafe_extra",
        }[mutation]
        setup.settings["unsafe_command_topics"] = ["/unsafe_extra"]
        setup.ros.external[topic] = [object()]
    elif mutation == "count":
        setup.ros.bad_count = True
    else:

        def failed_query() -> None:
            raise RuntimeError("synthetic graph failure")

        setup.ros.query_hook = failed_query
    with pytest.raises(RuntimeError):
        setup.create()
    assert not setup.sources.opened and not setup.ros.publisher_topics
    assert "node" in setup.ros.closed


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm_action", ""),
        ("arm_action", "/fake_gripper/follow_joint_trajectory"),
        ("policy_target_topic", "/joint_commands"),
        ("policy_target_topic", "/fake_armed"),
        ("joint_state_topic", "relative"),
        ("bridge_move_time_s", 0.5),
        ("bridge_move_time_s", True),
        ("shutdown_timeout_s", 0),
        ("image_width", True),
        ("wrist_camera", "/dev/video0"),
        ("unsafe_command_topics", "/unsafe"),
    ],
)
def test_invalid_adapter_settings_fail_before_ros_or_sources(
    setup: Any, field: str, value: Any
) -> None:
    setup.settings[field] = value
    with pytest.raises((ValueError, TypeError)):
        setup.create()
    assert not setup.sources.opened and not hasattr(setup.ros, "name")


@pytest.mark.parametrize("initial", [True, None, 1, "false"])
def test_preflight_requires_actual_disarmed_boolean(setup: Any, initial: Any) -> None:
    setup.ros.initial_armed = initial
    io = None
    try:
        io = setup.create()
        with pytest.raises(RuntimeError):
            io.preflight()
    except RuntimeError:
        assert initial in (1, "false")
    finally:
        if io is not None:
            io.close()
    assert not setup.ros.publisher_topics


@pytest.mark.parametrize("fault", ["no_sync", "early_arm", "no_transition", "abort"])
def test_sync_confirmation_and_fresh_manual_arming_are_required(setup: Any, fault: str) -> None:
    io = setup.create()
    io.preflight()
    setup.answers.extend(["not-confirmed"] if fault == "no_sync" else ["sync-off", "armed"])
    if fault == "early_arm":
        setup.ros.emit_armed(True)
    elif fault == "no_transition":
        io.input = lambda prompt: "sync-off" if "sync OFF" in prompt else "armed"
    elif fault == "abort":
        setup.answers[:] = ["abort"]
    with pytest.raises((RuntimeError, TimeoutError)):
        io.authorize_motion()
    io.close()
    assert not setup.ros.publisher_topics and not setup.ros.action_clients


def test_allowed_command_paths_clamp_speed_separate_gripper_and_hold_once(setup: Any) -> None:
    io = ready(setup)
    io.offer((100.0,) * 7)
    message = setup.ros.published[0]
    assert message.name == [f"Joint{i}" for i in range(1, 7)]
    assert message.position == pytest.approx([0.03] * 6)
    goal = setup.ros.goals[0]
    assert goal.trajectory.joint_names == ["synthetic_gripper"]
    assert goal.trajectory.points[0].positions == pytest.approx([0.048])
    assert goal.trajectory.points[0].time_from_start.nanosec == 400_000_000
    setup.sources.position = 0.012
    io.hold()
    io.hold()
    io.close()
    assert len(setup.ros.published) == 2 and len(setup.ros.goals) == 1
    assert setup.ros.published[-1].position == [0.012] * 6
    assert setup.ros.publisher_topics == ["/fake_policy_targets"]
    assert setup.ros.action_clients == ["/fake_gripper/follow_joint_trajectory"]
    assert setup.ros.closed.index("executor") < setup.ros.closed.index("publisher")
    assert io.session_evidence()["leader_sync_off"]["operator_confirmed"] is True


@pytest.mark.parametrize(
    "fault",
    ["disarm", "during_query", "direct", "competing", "gid", "stale", "header", "malformed"],
)
def test_offer_and_hold_do_not_bypass_any_gate(setup: Any, fault: str) -> None:
    io = ready(setup)
    if fault == "disarm":
        setup.ros.emit_armed(False)
    elif fault == "during_query":
        setup.ros.query_hook = lambda: setup.ros.emit_armed(False)
    elif fault == "direct":
        setup.ros.external["/joint_commands"] = [object()]
    elif fault == "competing":
        setup.ros.external["/fake_policy_targets"] = [object()]
    elif fault == "gid":
        setup.ros.publisher.endpoint.endpoint_gid = b"changed"
    elif fault == "stale":
        setup.sources.state_age = 1
    elif fault == "header":
        setup.sources.header_delay = 1
    else:
        io.settings["gripper_mapping"]["closed_position"] = float("inf")
    with pytest.raises((RuntimeError, ValueError)):
        io.offer((0.01,) * 7)
    io.close()
    assert not setup.ros.published and not setup.ros.goals


@pytest.mark.parametrize("fault", ["rejected", "result_failure", "timeout", "late_ack", "abort"])
def test_faults_hold_and_only_cancel_owned_gripper_goal(setup: Any, fault: str) -> None:
    io = ready(setup)
    if fault == "rejected":
        setup.ros.goal_accepted = False
    elif fault == "result_failure":
        setup.ros.result_code = -1
    elif fault == "timeout":
        setup.ros.result_delayed = True
    elif fault == "late_ack":
        setup.ros.ack_delayed = True
        start = setup.clock()
        setup.clock.on_sleep = lambda: (
            setup.ros.ack_future.finish()
            if setup.clock() - start > 0.035 and not setup.ros.ack_future.done()
            else None
        )
    else:
        setup.answers.append("abort")
    with pytest.raises((RuntimeError, TimeoutError)):
        if fault == "abort":
            io.completed()
        else:
            io.offer((0.01,) * 7)
    io.close()
    assert setup.ros.cancel_count == (1 if fault in {"timeout", "late_ack"} else 0)
    assert len(setup.ros.published) == (1 if fault == "abort" else 2)
    assert setup.ros.published[-1].position == [0] * 6
    assert "node" in setup.ros.closed


def test_cancellation_failure_retains_owned_resources_and_surfaces_error(setup: Any) -> None:
    io = ready(setup)
    setup.ros.result_delayed = True
    setup.ros.cancel_code = 1
    with pytest.raises(TimeoutError):
        io.offer((0.01,) * 7)
    with pytest.raises(RuntimeError, match="cancellation"):
        io.close()
    assert "node" not in setup.ros.closed and setup.ros.cancel_count == 1
    setup.ros.cancel_code = 0
    io._cancel_future.value.return_code = 0
    io.close()
    assert "node" in setup.ros.closed


@pytest.mark.parametrize("status", [4, 5, 6])
@pytest.mark.parametrize("evidence", ["result", "handle_status"])
def test_terminal_goal_disarm_still_fails_attempt_but_needs_no_cancellation(
    setup: Any,
    status: int,
    evidence: str,
) -> None:
    io = ready(setup)
    setup.ros.result_delayed = True
    setup.ros.result_status = status
    setup.ros.cancel_code = 3

    def terminal_then_disarm() -> None:
        if evidence == "result":
            setup.ros.result_future.finish()
        else:
            setup.ros.handle.status = status
        setup.ros.emit_armed(False)

    setup.clock.on_sleep = terminal_then_disarm
    with pytest.raises(RuntimeError, match="gate"):
        io.offer((0.01,) * 7)
    io.close()
    assert setup.ros.cancel_count == 0 and "node" in setup.ros.closed
    assert len(setup.ros.published) == len(setup.ros.goals) == 1


def test_deferred_callback_from_completed_goal_cannot_cancel_or_mutate_new_goal(setup: Any) -> None:
    io = ready(setup)
    setup.ros.defer_callbacks = True
    io.offer((0.01,) * 7)
    first_ack = setup.ros.ack_future
    setup.clock.sleep(0.5)
    setup.ros.result_delayed = True

    def deliver_old_callback() -> None:
        current_result = setup.ros.ack_future.result

        def result_with_deferred_callback() -> Any:
            if io._cancel_requested:
                first_ack.deliver_callbacks()
            return current_result()

        setup.ros.ack_future.result = result_with_deferred_callback
        setup.clock.on_sleep = lambda: None

    setup.clock.on_sleep = deliver_old_callback
    with pytest.raises(TimeoutError):
        io.offer((0.01,) * 7)
    io.close()
    assert setup.ros.cancelled_goals == [2]
    assert "node" in setup.ros.closed


def test_terminal_result_during_cancellation_is_not_reported_as_unknown(setup: Any) -> None:
    io = ready(setup)
    setup.ros.result_delayed = True

    def completed_during_cancel(index: int) -> Future:
        setup.ros.result_future.finish()
        return Future(NS(return_code=3))

    setup.ros.cancel = completed_during_cancel
    with pytest.raises(TimeoutError):
        io.offer((0.01,) * 7)
    io.close()
    assert "node" in setup.ros.closed


def test_fault_hold_precedes_bounded_wait_for_unknown_gripper_acceptance(setup: Any) -> None:
    io = ready(setup)
    setup.ros.ack_delayed = True
    cancellation_waits = []

    def observe_cancel_wait() -> None:
        if io._cancel_requested:
            cancellation_waits.append(setup.clock())
            assert len(setup.ros.published) == 2
            assert setup.ros.published[-1].position == [0] * 6

    setup.clock.on_sleep = observe_cancel_wait
    with pytest.raises(TimeoutError):
        io.offer((0.01,) * 7)
    assert cancellation_waits and not setup.ros.cancel_count
    setup.clock.on_sleep = lambda: None
    setup.ros.ack_future.finish()
    io.close()
    assert setup.ros.cancelled_goals == [1] and "node" in setup.ros.closed


@pytest.mark.parametrize("position,target", [(0, 0.05), (0.0125, 0.025), (0.025, 0)])
def test_verified_gripper_mapping_respects_driver_sense(
    setup: Any, position: float, target: float
) -> None:
    assert gripper_target(setup.settings["gripper_mapping"], position, 0.025) == pytest.approx(
        target
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified_by", ""),
        ("verified_on", None),
        ("units", "invented"),
        ("joint_name", ""),
        ("closed_position", True),
        ("closed_position", float("nan")),
    ],
)
def test_unverified_gripper_mapping_creates_no_command_resources(
    setup: Any, field: str, value: Any
) -> None:
    setup.settings["gripper_mapping"][field] = value
    io = setup.create()
    io.preflight()
    with pytest.raises(ValueError):
        io.authorize_motion()
    io.close()
    assert not setup.ros.publisher_topics and not setup.ros.action_clients


def test_gripper_conversion_rejects_overflow_and_boolean_position(setup: Any) -> None:
    mapping = setup.settings["gripper_mapping"]
    with pytest.raises(ValueError):
        gripper_target(mapping, True, 0.025)
    mapping.update(open_position=-1e308, closed_position=1e308)
    with pytest.raises(ValueError):
        gripper_target(mapping, 0, 0.025)


def test_too_fast_second_offer_stops_with_single_fresh_hold(setup: Any) -> None:
    io = ready(setup)
    io.offer((0.01,) * 7)
    with pytest.raises(RuntimeError, match="period"):
        io.offer((0.01,) * 7)
    io.close()
    assert len(setup.ros.goals) == 1 and len(setup.ros.published) == 2


@pytest.mark.parametrize("key", ["limits", "policy_config"])
def test_adapter_itself_refuses_unverified_motion_before_sink(setup: Any, key: str) -> None:
    path = Path(setup.config[key])
    content = json.loads(path.read_text())
    content["verified_by"] = ""
    path.write_text(json.dumps(content))
    io = setup.create()
    io.preflight()
    with pytest.raises(ValueError, match="both verified"):
        io.authorize_motion()
    with pytest.raises(RuntimeError, match="gate"):
        io.command_sink()
    io.close()
    assert not setup.ros.publisher_topics and not setup.ros.action_clients


def test_stale_camera_refuses_observation_and_closes_sources(setup: Any) -> None:
    io = setup.create()
    setup.sources.image_age = 1
    with pytest.raises(RuntimeError, match="stale"):
        io.preflight()
    io.close()
    assert setup.sources.closed == setup.sources.opened


@pytest.mark.parametrize("fault", ["duplicate_node", "duplicate_server", "wrong_type", "empty_gid"])
def test_discovery_identity_must_be_unambiguous(setup: Any, fault: str) -> None:
    io = setup.create()
    io.preflight()
    setup.answers.extend(["sync-off", "armed"])
    io.authorize_motion()
    if fault == "duplicate_node":
        setup.ros.nodes.append((setup.ros.name, "/other"))
    elif fault == "duplicate_server":
        setup.ros.servers.append(setup.ros.servers[0])
    else:
        create = setup.ros.create_publisher

        def corrupt_endpoint(*args: Any) -> Any:
            publisher = create(*args)
            if fault == "wrong_type":
                publisher.endpoint.topic_type = "std_msgs/msg/Bool"
            else:
                publisher.endpoint.endpoint_gid = b"\x00" * 24
            return publisher

        setup.ros.create_publisher = corrupt_endpoint
    with pytest.raises(RuntimeError):
        io.command_sink()
    io.close()
    assert not setup.ros.published and not setup.ros.goals


def test_guard_rechecks_terminal_worker_failure_after_graph_query(setup: Any) -> None:
    io = ready(setup)

    def fail_during_query() -> None:
        io._error = RuntimeError("synthetic terminal graph failure")

    setup.ros.query_hook = fail_during_query
    with pytest.raises(RuntimeError):
        io.offer((0.01,) * 7)
    io.close()
    assert not setup.ros.published and not setup.ros.goals


def test_hold_restarts_minimum_cadence_before_next_offer(setup: Any) -> None:
    io = ready(setup)
    io.offer((0.01,) * 7)
    setup.clock.sleep(0.5)
    io.hold()
    with pytest.raises(RuntimeError, match="period"):
        io.offer((0.01,) * 7)
    io.close()
    assert len(setup.ros.goals) == 1 and len(setup.ros.published) == 2


def test_command_cadence_starts_after_slow_graph_and_successful_publication(setup: Any) -> None:
    io = ready(setup)
    queries = []

    def slow_final_graph_check() -> None:
        queries.append(True)
        if len(queries) == 2:
            setup.clock.sleep(0.3)

    publish = setup.ros.publisher.publish

    def slow_publish(message: Any) -> None:
        setup.clock.sleep(0.1)
        publish(message)

    setup.ros.query_hook = slow_final_graph_check
    setup.ros.publisher.publish = slow_publish
    started = setup.clock()
    io.offer((0.01,) * 7)
    assert io._last_offer == pytest.approx(started + 0.4)
    setup.clock.sleep(0.1)
    with pytest.raises(RuntimeError, match="period"):
        io.offer((0.01,) * 7)
    assert io._last_offer == pytest.approx(setup.clock())
    io.close()
    assert len(setup.ros.goals) == 1 and len(setup.ros.published) == 2


@pytest.mark.parametrize("operation", ["offer", "hold"])
@pytest.mark.parametrize("fault", ["disarm", "worker_failure"])
def test_terminal_arm_submission_gate_runs_after_message_stamp_creation(
    setup: Any,
    operation: str,
    fault: str,
) -> None:
    io = ready(setup)

    def clock_with_terminal_fault() -> Any:
        if fault == "disarm":
            setup.ros.emit_armed(False)
        else:
            io._error = RuntimeError("synthetic terminal worker failure")
        return NS(now=lambda: NS(to_msg=lambda: NS(sec=100, nanosec=0)))

    setup.ros.get_clock = clock_with_terminal_fault
    with pytest.raises(RuntimeError, match="gate"):
        io.offer((0.01,) * 7) if operation == "offer" else io.hold()
    io.close()
    assert not setup.ros.published and not setup.ros.goals


def test_terminal_gripper_submission_gate_refuses_known_disarm(setup: Any) -> None:
    io = ready(setup)
    original = io._gripper

    class DisarmingClient:
        @property
        def send_goal_async(self) -> Any:
            setup.ros.emit_armed(False)
            return original.send_goal_async

        def destroy(self) -> None:
            original.destroy()

    io._gripper = DisarmingClient()
    with pytest.raises(RuntimeError, match="gate"):
        io.offer((0.01,) * 7)
    io.close()
    assert len(setup.ros.published) == 1 and not setup.ros.goals


def test_clamps_use_final_arm_snapshot_and_newest_pre_gripper_snapshot(setup: Any) -> None:
    io = ready(setup)
    queries = []

    def changing_state_during_graph() -> None:
        queries.append(True)
        if len(queries) == 2:
            setup.sources.position, setup.sources.gripper = 0.2, 0.01
        elif len(queries) == 3:
            setup.sources.position, setup.sources.gripper = 0.3, 0.02

    setup.ros.query_hook = changing_state_during_graph
    io.offer((-1.0,) * 7)
    assert setup.ros.published[0].position == pytest.approx([0.17] * 6)
    assert setup.ros.goals[0].trajectory.points[0].positions == pytest.approx([0.012])
    io.close()


def test_partial_camera_startup_closes_already_owned_resources(setup: Any) -> None:
    camera = setup.sources.camera

    def fail_front(device: str, **kwargs: Any) -> Any:
        if device.endswith("front"):
            raise RuntimeError("synthetic front startup failure")
        return camera(device, **kwargs)

    setup.sources.camera = fail_front
    with pytest.raises(RuntimeError, match="front startup"):
        setup.create()
    assert len(setup.sources.closed) == 2 and "node" in setup.ros.closed


def test_sticky_worker_error_refuses_every_command(setup: Any) -> None:
    io = ready(setup)
    io._error = RuntimeError("synthetic executor failure")
    with pytest.raises(RuntimeError, match="gate"):
        io.offer((0.01,) * 7)
    io.close()
    assert not setup.ros.published and not setup.ros.goals


def test_source_cleanup_failure_can_be_retried_after_other_cleanup(setup: Any) -> None:
    io = setup.create()
    closed = []

    def retry_close() -> None:
        closed.append(True)
        if len(closed) == 1:
            raise RuntimeError("synthetic worker is still stopping")

    io.state.close = retry_close
    with pytest.raises(RuntimeError, match="source cleanup"):
        io.close()
    assert "node" in setup.ros.closed
    io.close()
    assert len(closed) == 2 and setup.ros.closed.count("node") == 1


def test_stuck_executor_retains_ros_resources_until_join_succeeds(setup: Any) -> None:
    io = setup.create()
    setup.ros.executor.shutdown = lambda **kwargs: False
    with pytest.raises(RuntimeError, match="worker did not stop"):
        io.close()
    assert "node" not in setup.ros.closed
    setup.ros.stop.set()
    io.close()
    assert "node" in setup.ros.closed


def test_shutdown_error_still_joins_and_destroys_stopped_resources(setup: Any) -> None:
    io = setup.create()

    def failed_shutdown(**kwargs: Any) -> None:
        setup.ros.stop.set()
        raise RuntimeError("synthetic shutdown error")

    setup.ros.executor.shutdown = failed_shutdown
    with pytest.raises(RuntimeError, match="shutdown reported"):
        io.close()
    assert "node" in setup.ros.closed


def test_interrupt_during_pending_action_cancels_only_owned_goal(setup: Any) -> None:
    io = ready(setup)
    setup.ros.result_delayed = True

    def interrupted() -> None:
        setup.clock.on_sleep = lambda: None
        raise KeyboardInterrupt

    setup.clock.on_sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        io.offer((0.01,) * 7)
    io.close()
    assert setup.ros.cancel_count == 1 and len(setup.ros.goals) == 1
    assert setup.ros.published[-1].position == [0] * 6


def test_twenty_graded_trials_use_adapter_fake_ros_and_stub_policy(
    setup: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = setup.config
    squares = {"A": [0, 0, 0], "B": [0.1, 0, 0]}
    for key, values in (("calibration", squares), ("reachable", list(squares))):
        path = tmp_path / (key + ".json")
        path.write_text(
            json.dumps(
                dict(
                    frame="synthetic",
                    squares=values,
                    verified_by="synthetic fixture",
                    verified_on="2026-10-08",
                )
            )
        )
        config[key] = str(path)
    trials = [
        dict(source="A", target="B", piece="red:0", condition=f"synthetic-{i}") for i in range(20)
    ]
    repository = tmp_path / "protocol_repo"
    repository.mkdir()
    protocol = repository / "protocol.md"
    registration = dict(
        protocol_sha256="", trials=20, success_threshold=14, max_attempts=1, scene_schedule=trials
    )
    text = "# Synthetic protocol\n\n```json\n" + json.dumps(registration, indent=2) + "\n```\n"
    digest = protocol_digest(text)
    protocol.write_text(text.replace('"protocol_sha256": ""', f'"protocol_sha256": "{digest}"'))
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "add", "protocol.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Operator",
            "-c",
            "user.email=operator@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Register synthetic protocol",
        ],
        cwd=repository,
        check=True,
    )
    config.update(
        protocol=str(protocol),
        protocol_sha256=digest,
        d2_trials=trials,
        max_steps=1,
        max_turns=20,
        max_attempts=1,
        policy_endpoint="http://unused.test",
    )
    config["provenance"] = {key: "synthetic fixture" for key in config["provenance"]}
    io = setup.create()
    setup.answers.extend(["sync-off", "armed"])
    for index in range(20):
        setup.answers.extend(
            ["continue", "yes", "success" if index < 14 else "failure", "5" if index < 14 else "3"]
        )
    setup.answers.append("synthetic powered state")
    original_input = io.input

    def paced_input(prompt: str) -> str:
        if "Continue session" in prompt:
            setup.clock.sleep(0.5)
        return original_input(prompt)

    io.input = paced_input
    requests = []

    class StubPolicy(FakePolicy):
        def request(self, payload: dict, timeout_s: float) -> dict:
            requests.append(payload)
            return super().request(payload, timeout_s)

    monkeypatch.setattr(
        "synria_lerobot.sessions.HttpPolicyTransport", lambda endpoint: StubPolicy((0.01,) * 7)
    )
    monkeypatch.setattr(
        "synria_lerobot.sessions.PolicyClient", lambda *args: PolicyClient(*args, clock=setup.clock)
    )
    output = Path(config["session_output"])
    stats = run_session(
        config, "d2", repository, output, enable_motion=True, factory=lambda conf: io
    )
    assert stats["threshold_met"] and stats["first_try"]["rate"] == 0.7
    assert stats["stage_status"] == "planned"
    assert len(requests) == len(setup.ros.goals) == 20
    assert all(request["reset"] and request["action_lookahead_steps"] == 2 for request in requests)
    assert len(list((output / "stills").glob("*.ppm"))) == 20
    provenance = json.loads((output / "provenance.json").read_text())
    source = provenance["source_preflight"]
    assert (
        source["state_source"] == "ros2_control"
        and source["state_rate_measurement"]["rate_hz"] == 30
    )
    assert source["leader_sync_off"]["operator_confirmed"] is True
    assert source["command_period_s"] == source["bridge_move_time_s"] == 0.4
    assert len(source["events"]) == 20
    assert (output / "EvalLog.jsonl").is_file() and (output / "session_end.json").is_file()
    assert "node" in setup.ros.closed and not setup.answers

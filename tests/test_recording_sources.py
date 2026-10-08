"""Threaded source tests use fake executors and synthetic camera arrays only."""

from __future__ import annotations

import queue
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from synria_lerobot import recorder


def fake_ros(monkeypatch: pytest.MonkeyPatch, *, fail_subscription: bool = False) -> Any:
    control = SimpleNamespace(
        messages=queue.Queue(),
        stopped=threading.Event(),
        processed=threading.Event(),
        release=threading.Event(),
        events=[],
        callback=None,
        qos=None,
        publishers=0,
        count=0,
        hang=False,
        callback_thread=None,
    )

    class Node:
        def create_subscription(self, kind: Any, topic: str, callback: Any, qos: int) -> object:
            if fail_subscription:
                raise OSError("fake subscription failure")
            control.callback, control.qos = callback, qos
            return object()

        def create_publisher(self, *args: Any) -> None:
            control.publishers += 1

        def get_clock(self) -> Any:
            return SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=1_800_000_000_135_000_000)
            )

        def destroy_subscription(self, subscription: object) -> None:
            control.events.append("destroy_subscription")

        def destroy_node(self) -> None:
            control.events.append("destroy_node")

    class Executor:
        def __init__(self, *, context: object) -> None:
            assert context is control.context

        def add_node(self, node: object) -> None:
            control.events.append("add_node")

        def spin(self) -> None:
            while not control.stopped.is_set():
                message = control.messages.get()
                if message is None:
                    return
                if control.hang:
                    control.processed.set()
                    control.release.wait()
                    return
                if isinstance(message, BaseException):
                    raise message
                control.callback_thread = threading.get_ident()
                control.callback(message)
                control.count += 1
                if control.count >= 2:
                    control.processed.set()

        def shutdown(self, *, timeout_sec: float) -> bool:
            control.events.append("shutdown_executor")
            control.stopped.set()
            control.messages.put(None)
            return True

        def remove_node(self, node: object) -> None:
            control.events.append("remove_node")

    def initialize(*, context: object) -> None:
        control.context = context
        control.events.append("initialize_context")

    def create_node(name: str, *, context: object) -> Node:
        assert context is control.context
        return Node()

    def shutdown(*, context: object) -> None:
        assert context is control.context
        control.events.append("shutdown_context")

    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        SimpleNamespace(init=initialize, create_node=create_node, shutdown=shutdown),
    )
    monkeypatch.setitem(sys.modules, "rclpy.context", SimpleNamespace(Context=object))
    monkeypatch.setitem(
        sys.modules,
        "rclpy.executors",
        SimpleNamespace(
            SingleThreadedExecutor=Executor,
            ExternalShutdownException=type("Shutdown", (Exception,), {}),
        ),
    )
    monkeypatch.setitem(sys.modules, "sensor_msgs", ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=object))
    return control


def joint_message(value: float) -> Any:
    return SimpleNamespace(
        name=[*[f"Joint{i}" for i in range(1, 7)], "Gripper"],
        position=[value] * 6 + [0.01],
        velocity=[],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1_800_000_000, nanosec=125_000_000)),
    )


def test_ros_caches_newest_depth_one_message_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = fake_ros(monkeypatch)
    monkeypatch.setattr(recorder.time, "monotonic", lambda: 123.5)
    source = recorder.RosJointStateSource("/unused", node_name="fake", timeout_s=0.5)
    control.messages.put(joint_message(0.1))
    control.messages.put(joint_message(0.2))
    assert control.processed.wait(1)
    state = source.read()
    assert state.joint_positions_rad == (0.2,) * 6
    assert state.monotonic_timestamp_s == 123.5
    assert state.ros_header_stamp_s == 1_800_000_000.125
    assert state.ros_arrival_stamp_s == 1_800_000_000.135
    assert source.read() is state
    assert control.callback_thread != threading.get_ident()
    assert control.qos == 1 and control.publishers == 0
    source.close()
    assert not source._thread.is_alive()
    assert control.events[-5:] == [
        "shutdown_executor",
        "remove_node",
        "destroy_subscription",
        "destroy_node",
        "shutdown_context",
    ]
    source.close()


@pytest.mark.parametrize("failure", ["callback", "executor"])
def test_ros_worker_errors_are_sticky(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    control = fake_ros(monkeypatch)
    source = recorder.RosJointStateSource("/unused", node_name="fake", timeout_s=0.5)
    message = joint_message(0.1)
    if failure == "callback":
        message.position = []
    else:
        message = OSError("executor failure")
    control.messages.put(message)
    source._thread.join(1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="background source failed"):
            source.read()
    source.close()


def test_ros_refuses_to_destroy_resources_until_worker_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = fake_ros(monkeypatch)
    control.hang = True
    source = recorder.RosJointStateSource("/unused", node_name="fake", timeout_s=0.01)
    control.messages.put(joint_message(0.1))
    assert control.processed.wait(1)
    with pytest.raises(RuntimeError, match="resources retained"):
        source.close()
    assert "destroy_node" not in control.events
    control.release.set()
    source._thread.join(1)
    source.close()
    assert control.events[-1] == "shutdown_context"


def test_failed_subscription_closes_only_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    control = fake_ros(monkeypatch, fail_subscription=True)
    with pytest.raises(OSError, match="fake subscription failure"):
        recorder.RosJointStateSource("/unused", node_name="fake")
    assert control.events[-3:] == ["shutdown_executor", "destroy_node", "shutdown_context"]


def fake_camera(monkeypatch: pytest.MonkeyPatch) -> Any:
    control = SimpleNamespace(
        frames=queue.Queue(),
        delivered=threading.Event(),
        release=threading.Event(),
        reads=0,
        released=False,
        last=None,
    )

    class Capture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, Any]:
            if control.reads < 2:
                value = control.frames.get()
                if isinstance(value, BaseException):
                    raise value
                control.last = value
                control.reads += 1
                return True, value
            control.delivered.set()
            control.release.wait()
            return True, control.last

        def release(self) -> None:
            control.released = True

    def resize(frame: Any, resolution: tuple[int, int], **kwargs: Any) -> Any:
        width, height = resolution
        return np.broadcast_to(frame[0, 0], (height, width, 3)).copy()

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda path: Capture(),
            COLOR_BGR2RGB=1,
            INTER_AREA=2,
            cvtColor=lambda frame, code: frame[:, :, ::-1],
            resize=resize,
        ),
    )
    return control


def test_camera_grabber_caches_latest_and_retains_arrival_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = fake_camera(monkeypatch)
    control.frames.put(np.full((8, 10, 3), (0, 0, 10), dtype=np.uint8))
    control.frames.put(np.full((8, 10, 3), (0, 0, 20), dtype=np.uint8))
    source = recorder.OpenCVFrameSource("/dev/v4l/by-id/fake", width=4, height=6, timeout_s=0.01)
    assert control.delivered.wait(1)
    frame = source.read()
    assert frame.data.shape == (6, 4, 3)
    assert frame.data[0, 0].tolist() == [20, 0, 0]
    assert source.read() is frame
    with pytest.raises(RuntimeError, match="resources retained"):
        source.close()
    assert not control.released
    control.release.set()
    source._thread.join(1)
    source.close()
    assert control.released and not source._thread.is_alive()


def test_camera_worker_failure_is_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    control = fake_camera(monkeypatch)
    control.frames.put(OSError("camera failure"))
    source = recorder.OpenCVFrameSource("/dev/v4l/by-id/fake")
    source._thread.join(1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="camera failure"):
            source.read()
    source.close()
    assert control.released

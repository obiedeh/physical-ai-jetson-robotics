from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from synria_lerobot.physical_contract import (
    ActionSource as ActionSourceKind,
)
from synria_lerobot.physical_contract import (
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.recorder import (
    ActionSample,
    EpisodeRecorder,
    NextStateActionSource,
    OperatorLabel,
    PhysicalEpisodeRecorder,
    PhysicalRecorderConfig,
    RecordedPhysicalEpisode,
    RecorderState,
    RosJointStateSource,
    run_operator_loop,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeStateSource:
    def __init__(self, states: list[PhysicalState]) -> None:
        self.states = iter(states)
        self.closed = False

    def read(self) -> PhysicalState:
        return next(self.states)

    def close(self) -> None:
        self.closed = True


class FakeLeaderActionSource:
    kind = ActionSourceKind.LEADER

    def __init__(self, values: list[tuple[float, ...]]) -> None:
        self.values = iter(values)
        self.closed = False

    def read(self, follower_state: PhysicalState) -> ActionSample:
        return ActionSample(next(self.values), follower_state.monotonic_timestamp_s)

    def close(self) -> None:
        self.closed = True


class FakeFrameSource:
    def __init__(self, label: str, clock: FakeClock) -> None:
        self.label = label
        self.clock = clock
        self.count = 0
        self.closed = False

    def read(self) -> ImageFrame:
        self.count += 1
        return ImageFrame(
            data=f"{self.label}-{self.count}".encode(),
            monotonic_timestamp_s=self.clock(),
        )

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    next_episode_index = 0

    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes: list[RecordedPhysicalEpisode] = []
        self.finalized = 0

    def save_final_still(self, episode_index: int, frame: ImageFrame) -> Path:
        path = self.root / f"episode_{episode_index:06d}_final.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame.data)
        return path

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None:
        self.episodes.append(episode)

    def finalize(self) -> None:
        self.finalized += 1


def _state(value: float, timestamp_s: float) -> PhysicalState:
    return PhysicalState(
        joint_positions_rad=(value,) * 6,
        gripper_m=min(value, 0.025),
        monotonic_timestamp_s=timestamp_s,
        ros_header_stamp_s=timestamp_s + 100.0,
    )


def _recorder(
    tmp_path: Path,
    *,
    action_source: ActionSourceKind,
    states: list[PhysicalState],
    clock: FakeClock,
    smoke: bool = False,
    initial_episode_index: int = 0,
) -> tuple[PhysicalEpisodeRecorder, FakeWriter]:
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=action_source,
        state_has_velocity=False,
    )
    config = PhysicalRecorderConfig(
        dataset_path=tmp_path / "dataset",
        repo_id="local/synria-d1",
        contract=contract,
        min_episode_s=20.0,
        max_episode_s=30.0,
        hard_cap_s=30.0,
        smoke=smoke,
    )
    writer = FakeWriter(tmp_path)
    writer.next_episode_index = initial_episode_index
    source = FakeStateSource(states)
    actions = (
        FakeLeaderActionSource([(0.01,) * 7 for _ in states])
        if action_source is ActionSourceKind.LEADER
        else NextStateActionSource()
    )
    return (
        PhysicalEpisodeRecorder(
            config=config,
            state_source=source,
            action_source=actions,
            wrist_source=FakeFrameSource("wrist", clock),
            front_source=FakeFrameSource("front", clock),
            writer=writer,
            clock=clock,
        ),
        writer,
    )


@pytest.mark.parametrize("label", [OperatorLabel.SUCCESS, OperatorLabel.FAILURE])
def test_operator_can_record_label_and_save_final_still(
    tmp_path: Path, label: OperatorLabel
) -> None:
    clock = FakeClock()
    recorder, writer = _recorder(
        tmp_path,
        action_source=ActionSourceKind.LEADER,
        states=[_state(0.0, 0.0), _state(0.01, 20.0)],
        clock=clock,
    )
    recorder.start()
    recorder.capture_once()
    clock.now = 20.0
    recorder.capture_once()
    recorder.stop()
    episode = (
        recorder.mark_success()
        if label is OperatorLabel.SUCCESS
        else recorder.mark_failure()
    )
    assert episode.operator_label is label
    assert episode.duration_s == pytest.approx(20.0)
    assert episode.final_still_path is not None
    assert episode.final_still_path.read_bytes() == b"front-2"
    assert writer.episodes == [episode]
    assert recorder.state is RecorderState.IDLE


def test_next_state_actions_are_shifted_one_frame(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0), _state(0.01, 20.0)],
        clock=clock,
    )
    recorder.start()
    recorder.capture_once()
    clock.now = 20.0
    recorder.capture_once()
    recorder.stop()
    episode = recorder.mark_success()
    assert episode.frames[0].action == pytest.approx((0.01,) * 6 + (0.01,))
    assert episode.frames[1].action == pytest.approx((0.01,) * 6 + (0.01,))


def test_recorder_continues_episode_index_from_writer_metadata(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0)],
        clock=clock,
        initial_episode_index=7,
    )
    recorder.start()
    recorder.capture_once()
    clock.now = 20.0
    recorder.stop()
    assert recorder.mark_success().episode_index == 7


def test_hard_cap_stops_before_collecting_late_frame(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0)],
        clock=clock,
    )
    recorder.start()
    clock.now = 30.0
    assert recorder.capture_once() is False
    assert recorder.state is RecorderState.STOPPED


def test_discard_writes_nothing(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, writer = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0)],
        clock=clock,
    )
    recorder.start()
    recorder.capture_once()
    recorder.discard()
    assert writer.episodes == []
    assert recorder.state is RecorderState.IDLE


def test_smoke_episode_does_not_advance_count(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0), _state(0.01, 20.0), _state(0.02, 21.0)],
        clock=clock,
        smoke=True,
    )
    for end in (20.0, 21.0):
        recorder.start()
        recorder.capture_once()
        clock.now = end
        recorder.stop()
        episode = recorder.mark_failure()
        assert episode.episode_index == 0
        assert episode.smoke is True


def test_dataset_path_must_be_outside_repository(tmp_path: Path) -> None:
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=ActionSourceKind.LEADER,
        state_has_velocity=False,
    )
    config = PhysicalRecorderConfig(
        dataset_path=tmp_path / "repo" / "data",
        repo_id="local/synria-d1",
        contract=contract,
    )
    with pytest.raises(ValueError, match="outside"):
        config.require_outside_repository(tmp_path / "repo")


def test_legacy_hardware_path_is_implemented_but_requires_configuration() -> None:
    with pytest.raises(RuntimeError, match="hardware_recorder"):
        EpisodeRecorder(mock=False).record()


def test_ros_state_source_creates_subscription_and_no_publishers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"subscriptions": 0, "publishers": 0}

    class FakeNode:
        def create_subscription(self, *args: Any, **kwargs: Any) -> object:
            del args, kwargs
            calls["subscriptions"] += 1
            return object()

        def create_publisher(self, *args: Any, **kwargs: Any) -> object:
            del args, kwargs
            calls["publishers"] += 1
            return object()

        def destroy_subscription(self, subscription: object) -> None:
            del subscription

        def destroy_node(self) -> None:
            return None

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: True  # type: ignore[attr-defined]
    rclpy.create_node = lambda name: FakeNode()  # type: ignore[attr-defined]
    rclpy.init = lambda: None  # type: ignore[attr-defined]
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    source = RosJointStateSource("/joint_states", node_name="test_state")
    assert calls == {"subscriptions": 1, "publishers": 0}
    source.close()


def test_failed_ros_subscription_destroys_new_node(monkeypatch: pytest.MonkeyPatch) -> None:
    destroyed = []

    def fail_subscription(*args: Any) -> None:
        raise OSError("fake subscription failure")

    node = SimpleNamespace(
        create_subscription=fail_subscription,
        destroy_node=lambda: destroyed.append(True),
    )
    rclpy = SimpleNamespace(ok=lambda: True, create_node=lambda name: node)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=object))
    with pytest.raises(OSError, match="fake subscription failure"):
        RosJointStateSource("/unused", node_name="fake_test")
    assert destroyed == [True]


def test_failed_camera_open_releases_fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from synria_lerobot.recorder import OpenCVFrameSource

    released = []
    capture = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda path: capture))
    with pytest.raises(RuntimeError, match="cannot open camera source"):
        OpenCVFrameSource("/dev/v4l/by-id/fake-test")
    assert released == [True]


def test_camera_capture_resizes_rgb_before_returning(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from synria_lerobot.recorder import OpenCVFrameSource

    raw = np.full((480, 640, 3), (10, 20, 230), dtype=np.uint8)
    resized_inputs = []

    def resize(rgb: Any, resolution: tuple[int, int], *, interpolation: int) -> Any:
        resized_inputs.append((rgb[0, 0].tolist(), resolution, interpolation))
        width, height = resolution
        return np.broadcast_to(rgb[0, 0], (height, width, 3)).copy()

    capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, raw), release=lambda: None)
    fake_cv = SimpleNamespace(
        VideoCapture=lambda path: capture, COLOR_BGR2RGB=1, INTER_AREA=2,
        cvtColor=lambda frame, code: frame[:, :, ::-1], resize=resize,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv)
    source = OpenCVFrameSource("/dev/v4l/by-id/fake-test", width=48, height=32)
    frame = source.read()
    source.close()
    assert frame.data.shape == (32, 48, 3)
    assert frame.data[0, 0].tolist() == [230, 20, 10]
    assert frame.native_resolution == (640, 480)
    assert frame.source_id == "/dev/v4l/by-id/fake-test"
    assert resized_inputs == [([230, 20, 10], (48, 32), 2)]
    raw[:] = 0
    assert frame.data[0, 0].tolist() == [230, 20, 10]


@pytest.mark.parametrize("width,height", [(0, 224), (224, -1), (1.5, 224), (True, 224)])
def test_invalid_image_size_is_rejected_before_capture(width: Any, height: Any) -> None:
    from synria_lerobot.recorder import OpenCVFrameSource

    with pytest.raises(ValueError, match="positive integers"):
        OpenCVFrameSource("/dev/v4l/by-id/not-opened", width=width, height=height)


@pytest.mark.parametrize("exit_kind", ["success", "quit", "eof", "interrupt", "capture_error"])
def test_operator_loop_finalizes_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    import select

    clock = FakeClock()
    recorder, writer = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0)],
        clock=clock,
    )
    commands = {
        "success": "start\nstop\nsuccess\n",
        "quit": "quit\n",
        "eof": "",
        "interrupt": "",
        "capture_error": "start\n",
    }
    monkeypatch.setattr(sys, "stdin", StringIO(commands[exit_kind]))

    def ready(*args: Any) -> tuple[list[Any], list[Any], list[Any]]:
        if exit_kind == "interrupt":
            raise KeyboardInterrupt
        return [sys.stdin], [], []

    monkeypatch.setattr(select, "select", ready)
    if exit_kind == "capture_error":
        def fail_capture() -> PhysicalState:
            raise OSError("fake state failure")

        monkeypatch.setattr(recorder.state_source, "read", fail_capture)
    if exit_kind == "success":
        run_operator_loop(recorder)
    else:
        expected_error = {
            "quit": RuntimeError,
            "eof": RuntimeError,
            "interrupt": KeyboardInterrupt,
            "capture_error": OSError,
        }[exit_kind]
        with pytest.raises(expected_error):
            run_operator_loop(recorder)
    recorder.close()
    assert writer.finalized == 1
    assert recorder.state_source.closed
    assert recorder.wrist_source.closed
    assert recorder.front_source.closed


def test_cleanup_attempts_finalization_after_source_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, writer = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[],
        clock=FakeClock(),
    )

    def fail_close() -> None:
        raise OSError("fake close failure")

    monkeypatch.setattr(recorder.state_source, "close", fail_close)
    with pytest.raises(RuntimeError, match="session cleanup failed"):
        recorder.close()
    assert recorder.wrist_source.closed
    assert recorder.front_source.closed
    assert writer.finalized == 1
    recorder.close()
    assert writer.finalized == 1


def test_cleanup_failure_does_not_hide_original_session_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import select

    recorder, writer = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[],
        clock=FakeClock(),
    )

    def fail_close() -> None:
        raise OSError("fake close failure")

    monkeypatch.setattr(recorder.state_source, "close", fail_close)
    monkeypatch.setattr(sys, "stdin", StringIO(""))
    monkeypatch.setattr(select, "select", lambda *args: ([sys.stdin], [], []))
    with pytest.raises(RuntimeError, match="operator input closed"):
        run_operator_loop(recorder)
    assert writer.finalized == 1
    assert "fake close failure" in capsys.readouterr().err


@pytest.mark.parametrize("failure_at", ["follower", "wrist", "front", "recorder", "loop"])
def test_main_cleans_partial_startup_and_session_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    from synria_lerobot import recorder

    closed = []
    args = SimpleNamespace(
        dataset_path=tmp_path / "dataset", repo_id="local/test", gripper_type="50mm",
        action_source="next_state", state_has_velocity=False, smoke=False, fps=15,
        image_width=224, image_height=224,
        follower_topic="follower", leader_topic="leader",
        wrist_camera="wrist", front_camera="front",
    )
    monkeypatch.setattr(recorder, "_parse_physical_args", lambda: args)
    monkeypatch.setattr(
        recorder, "LeRobotDatasetWriter",
        lambda config: SimpleNamespace(
            next_episode_index=0, finalize=lambda: closed.append("writer")
        ),
    )

    def source(name: str, **kwargs: Any) -> SimpleNamespace:
        if name == failure_at:
            raise OSError(f"fake {name} failure")
        return SimpleNamespace(close=lambda: closed.append(name))

    monkeypatch.setattr(recorder, "RosJointStateSource", source)
    monkeypatch.setattr(recorder, "OpenCVFrameSource", source)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"fake {failure_at} failure")

    if failure_at == "recorder":
        monkeypatch.setattr(recorder, "PhysicalEpisodeRecorder", fail)
    monkeypatch.setattr(recorder, "run_operator_loop", fail)
    with pytest.raises(OSError, match=f"fake {failure_at} failure"):
        recorder.physical_main()
    assert closed.count("writer") == 1
    order = ["follower", "wrist", "front", "recorder", "loop"]
    for name in ("follower", "wrist", "front"):
        assert closed.count(name) == int(order.index(name) < order.index(failure_at))

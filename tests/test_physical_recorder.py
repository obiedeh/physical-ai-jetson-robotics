from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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
        return ActionSample(
            next(self.values), follower_state.monotonic_timestamp_s,
            follower_state.ros_header_stamp_s, follower_state.ros_arrival_stamp_s,
        )

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
    recovery_blocked = False

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
        episode.final_still_path = self.save_final_still(
            episode.episode_index, episode.frames[-1].front
        )
        self.episodes.append(episode)

    def finalize(self) -> None:
        self.finalized += 1


def _state(value: float, timestamp_s: float) -> PhysicalState:
    return PhysicalState(
        joint_positions_rad=(value,) * 6,
        gripper_m=min(value, 0.025),
        monotonic_timestamp_s=timestamp_s,
        ros_header_stamp_s=timestamp_s + 100.0,
        ros_arrival_stamp_s=timestamp_s + 100.0,
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
    assert episode.frames[0].action_monotonic_timestamp_s == 20.0
    assert episode.frames[0].action_ros_header_stamp_s == 120.0
    assert episode.frames[1].action_monotonic_timestamp_s == 20.0
    assert episode.achieved_sample_rate_hz == pytest.approx(0.05)


def test_sample_time_is_taken_after_source_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0)], clock=clock,
    )
    original_read = recorder.front_source.read

    def delayed_read() -> ImageFrame:
        clock.now = 0.01
        return original_read()

    monkeypatch.setattr(recorder.front_source, "read", delayed_read)
    recorder.start()
    recorder.capture_once()
    pending = recorder._pending[0]
    assert pending.state.monotonic_timestamp_s == 0
    assert pending.wrist.monotonic_timestamp_s == 0
    assert pending.front.monotonic_timestamp_s == 0.01
    assert pending.sample_monotonic_s == 0.01
    recorder.close()


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


def test_smoke_dataset_still_uses_unique_episode_indices(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path,
        action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.0, 0.0), _state(0.01, 20.0), _state(0.02, 21.0)],
        clock=clock,
        smoke=True,
    )
    for index, end in enumerate((20.0, 21.0)):
        recorder.start()
        recorder.capture_once()
        clock.now = end
        recorder.stop()
        episode = recorder.mark_failure()
        assert episode.episode_index == index
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
    assert resized_inputs and all(item == ([230, 20, 10], (48, 32), 2) for item in resized_inputs)
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
        "success": "start\nstop\nsuccess\nquit\n",
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
    if exit_kind in {"success", "quit"}:
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
            next_episode_index=0, recovery_blocked=False, finalize=lambda: closed.append("writer")
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


def test_session_saves_many_episodes_and_returns_no_frame_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import select

    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 0.0), _state(0.02, 0.0)], clock=FakeClock(),
    )
    monkeypatch.setattr(
        sys, "stdin", StringIO("start\nstop\nsuccess\nstart\nstop\nfailure\nquit\n")
    )
    monkeypatch.setattr(select, "select", lambda *args: ([sys.stdin], [], []))
    result = EpisodeRecorder(mock=False, hardware_recorder=recorder).record()
    assert result.saved_episode_count == 2
    assert result.last_episode_index == 1
    assert not hasattr(result, "frames")
    assert [episode.episode_index for episode in writer.episodes] == [0, 1]
    assert recorder._pending == []


@pytest.mark.parametrize("choice", ["retry", "discard"])
def test_failed_save_retains_frames_until_explicit_retry_or_discard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], choice: str
) -> None:
    import select

    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 0.0)], clock=FakeClock(),
    )
    original = writer.write_episode
    attempts = []

    def fail_once(episode: RecordedPhysicalEpisode) -> None:
        attempts.append(episode.frames[0].wrist.data)
        if len(attempts) == 1:
            raise OSError("injected save failure")
        original(episode)

    monkeypatch.setattr(writer, "write_episode", fail_once)
    commands = f"start\nstop\nsuccess\nstart\nquit\n{choice}\nquit\n"
    monkeypatch.setattr(sys, "stdin", StringIO(commands))
    monkeypatch.setattr(select, "select", lambda *args: ([sys.stdin], [], []))
    result = run_operator_loop(recorder)
    assert result.failed_save_count == 1
    assert result.saved_episode_count == int(choice == "retry")
    assert result.discarded_episode_count == int(choice == "discard")
    if choice == "retry":
        assert attempts[0] is attempts[1]
    output = capsys.readouterr().out
    assert "Frames retained" in output
    assert "Command refused" in output
    assert "explicitly discard before quit" in output


def test_smoke_session_exits_after_exactly_one_saved_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import select

    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 0.0)], clock=FakeClock(), smoke=True,
    )
    monkeypatch.setattr(sys, "stdin", StringIO("start\nstop\nsuccess\nstart\n"))
    monkeypatch.setattr(select, "select", lambda *args: ([sys.stdin], [], []))
    result = run_operator_loop(recorder)
    assert result.smoke and result.saved_episode_count == 1
    assert len(writer.episodes) == 1


def test_blocked_recovery_prevents_discard_or_new_capture(tmp_path: Path) -> None:
    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 0.0)], clock=FakeClock(),
    )
    recorder.start()
    recorder.capture_once()
    recorder.stop()
    writer.recovery_blocked = True
    with pytest.raises(RuntimeError, match="recovery is blocked"):
        recorder.discard()
    with pytest.raises(RuntimeError, match="recovery is blocked"):
        recorder.start()
    assert len(recorder._pending) == 1

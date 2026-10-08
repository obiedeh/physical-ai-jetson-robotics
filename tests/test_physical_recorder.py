from __future__ import annotations

import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_task_registry import synthetic_registry, synthetic_task

from synria_lerobot.physical_contract import (
    ActionSource as ActionSourceKind,
)
from synria_lerobot.physical_contract import (
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
    StateRateMeasurement,
    StateSourceProvenance,
)
from synria_lerobot.recorder import (
    ActionSample,
    EpisodeRecorder,
    LeaderActionSource,
    NextStateActionSource,
    OperatorLabel,
    PhysicalEpisodeRecorder,
    PhysicalRecorderConfig,
    PhysicalSessionResult,
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
            episode.episode_index, episode.final_front_still or episode.frames[-1].front
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
    state_has_velocity: bool = False,
    action_lookahead_steps: int = 1,
) -> tuple[PhysicalEpisodeRecorder, FakeWriter]:
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=action_source,
        state_has_velocity=state_has_velocity,
        action_lookahead_steps=action_lookahead_steps,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
        recording_purpose="disposable_smoke" if smoke else "qualifying",
    )
    config = PhysicalRecorderConfig(
        dataset_path=tmp_path / "dataset",
        repo_id="local/synria-d1",
        contract=contract,
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


@pytest.mark.parametrize("steps", [0, 1, 2, 9])
@pytest.mark.parametrize("source", list(ActionSourceKind))
def test_configured_lookahead_shifts_only_follower_actions_and_clamps_tail(
    tmp_path: Path, steps: int, source: ActionSourceKind
) -> None:
    clock = FakeClock()
    states = [_state((index + 1) / 1000, float(index)) for index in range(3)]
    recorder, _ = _recorder(
        tmp_path, action_source=source, states=states, clock=clock, action_lookahead_steps=steps
    )
    recorder.start()
    for index in range(3):
        clock.now = float(index)
        recorder.capture_once()
    clock.now = 20
    recorder.stop()
    episode = recorder.mark_success()
    recorder.close()
    assert episode.action_lookahead_steps == steps
    for index, frame in enumerate(episode.frames):
        target = min(index + steps, 2) if source is ActionSourceKind.NEXT_STATE else index
        expected = (
            states[target].observation_vector()
            if source is ActionSourceKind.NEXT_STATE else (0.01,) * 7
        )
        assert frame.action == expected
        assert frame.action_monotonic_timestamp_s == states[target].monotonic_timestamp_s
        assert frame.action_ros_header_stamp_s == states[target].ros_header_stamp_s


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


def test_missing_required_velocity_refuses_start_without_capture_or_save(tmp_path: Path) -> None:
    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE, states=[_state(0.01, 0.0)],
        clock=FakeClock(), state_has_velocity=True,
    )
    with pytest.raises(ValueError, match="requires six reported joint velocities"):
        recorder.start()
    assert recorder.state is RecorderState.IDLE
    assert recorder._pending == [] and writer.episodes == []
    assert recorder.wrist_source.count == recorder.front_source.count == 0
    recorder.close()
    assert writer.finalized == 1


@pytest.mark.parametrize("required", [False, True])
def test_state_contract_is_applied_at_capture_and_preflight_never_replaces_latest(
    tmp_path: Path, required: bool
) -> None:
    initial = replace(_state(0.01, 0), joint_velocities_rad_s=(0.1,) * 6)
    latest = replace(_state(0.02, 0), joint_velocities_rad_s=(0.2,) * 6)
    # Required-velocity preflight consumes one explicit read; capture reads the newer state.
    states = [initial, latest] if required else [latest]
    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE, states=states,
        clock=clock, state_has_velocity=required,
    )
    recorder.start()
    recorder.capture_once()
    assert recorder._pending[0].state.joint_positions_rad == (0.02,) * 6
    clock.now = 20
    recorder.stop()
    episode = recorder.mark_success()
    assert len(episode.frames[0].state.observation_vector()) == (13 if required else 7)
    assert len(episode.frames[0].action) == 7
    recorder.close()


def test_velocity_loss_after_preflight_fails_before_a_frame_is_accepted(tmp_path: Path) -> None:
    states = [replace(_state(0.01, 0), joint_velocities_rad_s=(0.1,) * 6), _state(0.02, 0)]
    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE, states=states,
        clock=FakeClock(), state_has_velocity=True,
    )
    recorder.start()
    with pytest.raises(ValueError, match="requires six reported joint velocities"):
        recorder.capture_once()
    assert recorder._pending == [] and writer.episodes == []
    assert recorder.wrist_source.count == recorder.front_source.count == 0
    recorder.close()


def test_leader_actions_never_include_reported_velocities() -> None:
    leader = replace(_state(0.02, 1.0), joint_velocities_rad_s=(0.2,) * 6)
    source = FakeStateSource([leader])
    actions = LeaderActionSource(source)
    sample = actions.read(_state(0.01, 1.0))
    assert sample.values == (0.02,) * 7
    assert sample.monotonic_timestamp_s == 1.0
    actions.close()
    assert source.closed


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
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
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
    assert frame.native_data.shape == (480, 640, 3)
    assert not frame.native_data.flags.writeable
    assert resized_inputs and all(item == ([230, 20, 10], (48, 32), 2) for item in resized_inputs)
    raw[:] = 0
    assert frame.data[0, 0].tolist() == [230, 20, 10]
    assert frame.native_data[0, 0].tolist() == [230, 20, 10]


def test_recorder_pins_only_final_native_front_through_failed_save_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    clock = FakeClock()
    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, float(index)) for index in range(8)], clock=clock,
    )
    native = np.zeros((480, 640, 3), dtype=np.uint8)

    def read() -> ImageFrame:
        native[:] = (int(clock()), 30, 220)
        return ImageFrame(
            np.full((224, 224, 3), native[0, 0], dtype=np.uint8), clock(),
            (640, 480), "synthetic-front", native,
        )

    monkeypatch.setattr(recorder.front_source, "read", read)
    monkeypatch.setattr(recorder.wrist_source, "read", read)
    written = []

    def save_still(index: int, frame: ImageFrame) -> Path:
        written.append(frame)
        path = tmp_path / f"still-{index}.jpg"
        path.write_bytes(b"synthetic still")
        return path

    monkeypatch.setattr(writer, "save_final_still", save_still)
    recorder.start()
    for index in range(8):
        clock.now = float(index)
        assert recorder.capture_once()
    assert all(
        pending.front.native_data is None and pending.wrist.native_data is None
        and pending.front.data.shape == (224, 224, 3)
        for pending in recorder._pending
    )
    pinned = recorder._pending_final_front
    assert pinned is not None and pinned.data.shape == (480, 640, 3)
    assert not pinned.data.flags.writeable
    recorder.stop()
    original_save = writer.write_episode

    def fail_save(episode: RecordedPhysicalEpisode) -> None:
        assert episode.final_front_still is pinned
        raise OSError("synthetic save failure")

    monkeypatch.setattr(writer, "write_episode", fail_save)
    with pytest.raises(OSError, match="synthetic save failure"):
        recorder.mark_success()
    assert recorder._pending_final_front is pinned
    clock.now = 99.0
    read()  # Reuse the backend buffer after stop and before retry.
    monkeypatch.setattr(writer, "write_episode", original_save)
    episode = recorder.retry()
    assert written == [pinned]
    assert pinned.data[0, 0].tolist() == [7, 30, 220]
    assert pinned.monotonic_timestamp_s == episode.frames[-1].front.monotonic_timestamp_s == 7
    assert recorder._pending_final_front is None
    assert recorder._pending == []


def test_hard_cap_rejected_grab_cannot_replace_pinned_native_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    clock = FakeClock()
    recorder, _ = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 1), _state(0.01, 29)], clock=clock,
    )

    def read() -> ImageFrame:
        stamp = clock()
        if stamp == 29:
            clock.now = 30
        return ImageFrame(
            np.zeros((224, 224, 3), dtype=np.uint8), stamp, (640, 480),
            "synthetic-front", np.full((480, 640, 3), int(stamp), dtype=np.uint8),
        )

    monkeypatch.setattr(recorder.front_source, "read", read)
    recorder.start()
    clock.now = 1
    assert recorder.capture_once()
    pinned = recorder._pending_final_front
    clock.now = 29
    assert not recorder.capture_once()
    assert recorder.state is RecorderState.STOPPED
    assert recorder._pending_final_front is pinned
    assert len(recorder._pending) == 1
    assert pinned is not None and pinned.monotonic_timestamp_s == 1
    recorder.discard()
    assert recorder._pending_final_front is None


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
        task_id="die_into_cup",
        task_registry=synthetic_registry(tmp_path / "tasks.json", 20, 30),
        dataset_path=tmp_path / "dataset", repo_id="local/test", gripper_type="50mm",
        action_source="next_state", state_has_velocity=False, smoke=False, fps=15,
        image_width=224, image_height=224, action_lookahead_steps=1,
        follower_topic="/follower", leader_topic="/leader",
        state_source="standalone_driver", guard_command_topic=[],
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
        name = name.lstrip("/")
        if name == failure_at:
            raise OSError(f"fake {name} failure")
        return SimpleNamespace(
            close=lambda: closed.append(name),
            measure_rate=lambda: StateRateMeasurement(50, 100, 2, 0, 2, 0.02),
            read=lambda: _state(0.01, 0),
            require_no_command_publishers=lambda topics: None,
        )

    monkeypatch.setattr(recorder, "RosJointStateSource", source)
    monkeypatch.setattr(recorder, "OpenCVFrameSource", source)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"fake {failure_at} failure")

    if failure_at == "recorder":
        monkeypatch.setattr(recorder, "PhysicalEpisodeRecorder", fail)
    monkeypatch.setattr(recorder, "run_operator_loop", fail)
    with pytest.raises(OSError, match=f"fake {failure_at} failure"):
        recorder.physical_main()
    assert closed.count("writer") == int(failure_at != "follower")
    order = ["follower", "wrist", "front", "recorder", "loop"]
    for name in ("follower", "wrist", "front"):
        assert closed.count(name) == int(order.index(name) < order.index(failure_at))


@pytest.mark.parametrize("action_source", [None, "leader"])
@pytest.mark.parametrize("task_id", ["die_into_cup", "roll_and_dump", "cup_return"])
def test_cli_defaults_to_follower_only_and_keeps_optional_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action_source: str | None, task_id: str,
) -> None:
    from synria_lerobot import recorder

    arguments = [
        "recorder", "--dataset-path", str(tmp_path / "dataset"),
        "--task-id", task_id,
        "--task-registry", str(synthetic_registry(tmp_path / "tasks.json", 20, 30)),
        "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake-wrist", "--front-camera", "fake-front",
        "--state-has-velocity",
        "--action-lookahead-steps", "2",
        "--fps", "15",
        "--state-source", "standalone_driver",
    ]
    if action_source is not None:
        arguments.extend(["--action-source", action_source])
    monkeypatch.setattr(sys, "argv", arguments)
    assert recorder._parse_physical_args().action_source == (action_source or "next_state")
    sources = []
    configs = []

    def state_source(topic: str, **kwargs: Any) -> SimpleNamespace:
        sources.append((topic, kwargs["state_has_velocity"]))
        return SimpleNamespace(
            close=lambda: None,
            measure_rate=lambda: StateRateMeasurement(50, 100, 2, 0, 2, 0.02),
            read=lambda: replace(_state(0.01, 0), joint_velocities_rad_s=(0.1,) * 6),
            require_no_command_publishers=lambda topics: None,
        )

    def writer(config: PhysicalRecorderConfig) -> FakeWriter:
        configs.append(config)
        return FakeWriter(tmp_path)

    monkeypatch.setattr(recorder, "RosJointStateSource", state_source)
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", writer)
    monkeypatch.setattr(
        recorder, "OpenCVFrameSource", lambda *args, **kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(recorder, "run_operator_loop", lambda _: PhysicalSessionResult())
    assert recorder.physical_main() == 0
    expected = [("/joint_states", True)]
    if action_source == "leader":
        expected.append(("/leader/joint_states", False))
    assert sources == expected
    assert configs[0].contract.action_source.value == (action_source or "next_state")
    assert configs[0].contract.action_lookahead_steps == 2
    assert configs[0].state_rate_measurement.rate_hz == 50
    assert configs[0].contract.task_id == task_id
    assert configs[0].task == synthetic_task(20, 30, task_id).task_text
    assert (configs[0].min_episode_s, configs[0].max_episode_s) == (20, 30)
    assert configs[0].contract.recording_purpose == "qualifying"


@pytest.mark.parametrize("task_id", [None, "unknown"])
def test_cli_requires_known_task_before_sources(
    monkeypatch: pytest.MonkeyPatch, task_id: str | None,
) -> None:
    from synria_lerobot import recorder

    arguments = [
        "recorder", "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake", "--front-camera", "fake", "--fps", "15",
        "--state-source", "ros2_control", "--smoke",
    ]
    if task_id is not None:
        arguments.extend(["--task-id", task_id])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        recorder, "RosJointStateSource", lambda *a, **k: pytest.fail("source opened")
    )
    with pytest.raises(SystemExit) as error:
        recorder.physical_main()
    assert error.value.code == 2


@pytest.mark.parametrize("fault", ["unset", "malformed", "goal", "override"])
def test_registry_refusal_precedes_temporary_directory_and_all_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    import json

    from test_task_registry import REGISTRY

    from synria_lerobot import recorder

    path = tmp_path / "tasks.json"
    payload = json.loads(REGISTRY.read_text())
    if fault == "goal":
        payload["tasks"][0]["requires_unobserved_goal"] = True
    path.write_text("{" if fault == "malformed" else json.dumps(payload))
    arguments = [
        "recorder", "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake", "--front-camera", "fake", "--fps", "15",
        "--state-source", "ros2_control", "--task-id", "die_into_cup",
        "--task-registry", str(path), "--dataset-path", str(tmp_path / "dataset"),
    ]
    if fault != "unset":
        arguments.append("--smoke")
    if fault == "override":
        arguments.extend(["--min-episode-s", "1", "--max-episode-s", "60"])
    monkeypatch.setattr(sys, "argv", arguments)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("registry refusal must precede any resource creation")

    for name in ("RosJointStateSource", "OpenCVFrameSource", "LeRobotDatasetWriter"):
        monkeypatch.setattr(recorder, name, forbidden)
    monkeypatch.setattr(recorder.tempfile, "TemporaryDirectory", forbidden)
    with pytest.raises((ValueError, SystemExit)):
        recorder.physical_main()
    assert not (tmp_path / "dataset").exists()


def test_recorder_cli_requires_explicit_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    from synria_lerobot import recorder

    monkeypatch.setattr(sys, "argv", [
        "recorder", "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake-wrist", "--front-camera", "fake-front", "--smoke",
        "--state-source", "standalone_driver",
        "--task-id", "die_into_cup",
    ])
    with pytest.raises(SystemExit) as error:
        recorder._parse_physical_args()
    assert error.value.code == 2


@pytest.mark.parametrize("state_source", [None, "unknown", "standalone_driver", "ros2_control"])
def test_recorder_cli_requires_declared_source_and_retains_default_guard_topics(
    monkeypatch: pytest.MonkeyPatch, state_source: str | None
) -> None:
    from synria_lerobot import recorder
    from synria_lerobot.physical_contract import guarded_command_topics

    arguments = [
        "recorder", "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake", "--front-camera", "fake", "--fps", "15", "--smoke",
        "--guard-command-topic", "/extra/one", "--guard-command-topic", "/extra/two",
        "--task-id", "die_into_cup",
    ]
    if state_source is not None:
        arguments.extend(["--state-source", state_source])
    monkeypatch.setattr(sys, "argv", arguments)
    if state_source in {None, "unknown"}:
        with pytest.raises(SystemExit) as error:
            recorder._parse_physical_args()
        assert error.value.code == 2
    else:
        args = recorder._parse_physical_args()
        assert args.state_source == state_source
        assert guarded_command_topics(tuple(args.guard_command_topic)) == (
            "/joint_commands", "/policy_joint_targets", "/extra/one", "/extra/two"
        )


@pytest.mark.parametrize("topic", ["relative", "/", "/bad//topic", "/1bad", " /joint_commands"])
def test_source_declaration_rejects_nonabsolute_or_invalid_topics(topic: str) -> None:
    from synria_lerobot.physical_contract import guarded_command_topics

    with pytest.raises(ValueError, match="absolute"):
        StateSourceProvenance("standalone_driver", topic)
    with pytest.raises(ValueError, match="absolute"):
        guarded_command_topics((topic,))


@pytest.mark.parametrize("failure", ["publisher", "graph"])
def test_command_guard_refusal_precedes_dataset_and_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from synria_lerobot import recorder

    monkeypatch.setattr(sys, "argv", [
        "recorder", "--dataset-path", str(tmp_path / "dataset"),
        "--repo-id", "local/fake", "--gripper-type", "50mm", "--state-source", "ros2_control",
        "--task-id", "die_into_cup",
        "--task-registry", str(synthetic_registry(tmp_path / "tasks.json", 20, 30)),
        "--wrist-camera", "fake", "--front-camera", "fake", "--fps", "15",
    ])
    events = []

    def measure() -> StateRateMeasurement:
        events.append("measured")
        return StateRateMeasurement(50, 100, 2, 0, 2, 0.02)

    def guard(topics: tuple[str, ...]) -> None:
        events.append("guarded")
        assert topics == ("/joint_commands", "/policy_joint_targets")
        raise RuntimeError(f"fake {failure} refusal")

    source = SimpleNamespace(
        measure_rate=measure, read=lambda: _state(0.01, 0),
        require_no_command_publishers=guard, close=lambda: events.append("closed"),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("guard refusal must precede dataset and camera construction")

    monkeypatch.setattr(recorder, "RosJointStateSource", lambda *args, **kwargs: source)
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", forbidden)
    monkeypatch.setattr(recorder, "OpenCVFrameSource", forbidden)
    with pytest.raises(RuntimeError, match=failure):
        recorder.physical_main()
    assert events == ["measured", "guarded", "closed"]
    assert not (tmp_path / "dataset").exists()


def test_late_publisher_refuses_next_episode_without_losing_pending_state(tmp_path: Path) -> None:
    clock = FakeClock()
    recorder, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE,
        states=[_state(0.01, 0)], clock=clock,
    )
    counts = [0]
    queries = []

    def guard(topics: tuple[str, ...]) -> None:
        queries.append(topics)
        if counts[0]:
            raise RuntimeError("recording refused: command publishers on /joint_commands")

    recorder._command_publisher_guard = guard
    recorder.start()
    recorder.capture_once()
    recorder.stop()
    recorder.mark_success()
    counts[0] = 1
    snapshot = (recorder._started, recorder._ended, recorder._next_episode_index, recorder._pending)
    with pytest.raises(RuntimeError, match="/joint_commands"):
        recorder.start()
    assert recorder.state is RecorderState.IDLE
    assert snapshot == (
        recorder._started, recorder._ended, recorder._next_episode_index, recorder._pending
    )
    assert len(writer.episodes) == 1
    assert recorder.wrist_source.count == recorder.front_source.count == 1
    assert len(queries) == 2
    counts[0] = 0
    recorder.start()
    recorder.stop()
    with pytest.raises(RuntimeError, match="pending episode"):
        recorder.start()
    assert len(queries) == 3
    recorder.discard()
    recorder.close()


def test_source_guard_cannot_be_replaced_by_an_injected_noop(tmp_path: Path) -> None:
    clock = FakeClock()
    existing, writer = _recorder(
        tmp_path, action_source=ActionSourceKind.NEXT_STATE, states=[], clock=clock,
    )
    config = replace(existing.config, state_source_provenance=StateSourceProvenance(
        "standalone_driver", "/joint_states"
    ))

    def refuse(topics: tuple[str, ...]) -> None:
        raise RuntimeError("source graph refusal")

    source = SimpleNamespace(require_no_command_publishers=refuse, close=lambda: None)
    arguments = dict(
        config=config, action_source=NextStateActionSource(), writer=writer,
        wrist_source=existing.wrist_source, front_source=existing.front_source, clock=clock,
    )
    recorder = PhysicalEpisodeRecorder(
        **arguments, state_source=source, command_publisher_guard=lambda topics: None
    )
    with pytest.raises(RuntimeError, match="source graph refusal"):
        recorder.start()
    assert recorder.state is RecorderState.IDLE
    assert existing.wrist_source.count == 0
    with pytest.raises(ValueError, match="requires a command publisher guard"):
        PhysicalEpisodeRecorder(**arguments, state_source=FakeStateSource([]))
    fake = PhysicalEpisodeRecorder(
        **arguments, state_source=FakeStateSource([]), command_publisher_guard=lambda topics: None
    )
    fake.start()
    assert fake.state is RecorderState.RECORDING
    recorder.close()


@pytest.mark.parametrize("failure", ["rate", "missing", "stale", "worker", "velocities"])
def test_failed_state_preflight_never_opens_dataset_or_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from synria_lerobot import recorder

    monkeypatch.setattr(sys, "argv", [
        "recorder", "--dataset-path", str(tmp_path / "dataset"),
        "--repo-id", "local/fake", "--gripper-type", "50mm",
        "--wrist-camera", "fake-wrist", "--front-camera", "fake-front", "--fps", "30",
        "--state-source", "standalone_driver",
        *( ["--state-has-velocity"] if failure == "velocities" else [] ),
        "--task-id", "die_into_cup",
        "--task-registry", str(synthetic_registry(tmp_path / "tasks.json", 20, 30)),
    ])
    closed = []

    def measure() -> StateRateMeasurement:
        if failure in {"missing", "stale", "worker"}:
            raise RuntimeError(f"fake {failure} preflight")
        rate = 15 if failure == "rate" else 50
        return StateRateMeasurement(rate, 2 * rate, 2, 0, 2, 1 / rate)

    source = SimpleNamespace(
        measure_rate=measure, read=lambda: _state(0.01, 0),
        close=lambda: closed.append("follower"),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight must precede dataset and camera construction")

    monkeypatch.setattr(recorder, "RosJointStateSource", lambda *args, **kwargs: source)
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", forbidden)
    monkeypatch.setattr(recorder, "OpenCVFrameSource", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        recorder.physical_main()
    assert closed == ["follower"]
    assert not (tmp_path / "dataset").exists()


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

from __future__ import annotations

import sys
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
    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes: list[RecordedPhysicalEpisode] = []

    def save_final_still(self, episode_index: int, frame: ImageFrame) -> Path:
        path = self.root / f"episode_{episode_index:06d}_final.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame.data)
        return path

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None:
        self.episodes.append(episode)


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

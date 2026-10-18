"""Goal channels: the contract, the frame, the embodiment, the recorder, the writer, the server."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_lerobot_dataset_writer import synthetic_episode, writer_config
from test_physical_recorder import (
    FakeClock,
    FakeFrameSource,
    FakeLeaderActionSource,
    FakeStateSource,
    FakeWriter,
    _state,
)
from test_synria_policy_server import StubModel, fixture, payload
from test_task_registry import synthetic_task

from synria_lerobot.act_training import FEATURE_KEYS, feature_keys
from synria_lerobot.embodiment import SynriaEmbodiment, SynriaObservation
from synria_lerobot.physical_contract import (
    GOAL_KEY,
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalFrame,
    PhysicalState,
)
from synria_lerobot.policy_codec import decode_policy_response, encode_policy_request
from synria_lerobot.policy_server import CheckpointIdentity, FixedScenePolicyService
from synria_lerobot.recorder import (
    GoalSample,
    LeRobotDatasetWriter,
    PhysicalEpisodeRecorder,
    PhysicalRecorderConfig,
    parse_goal_channels,
)
from synria_lerobot.task_registry import SIMULATION


def contract(channels: tuple[str, ...] = ("object", "target")) -> PhysicalDatasetContract:
    return PhysicalDatasetContract("50mm", ActionSource.NEXT_STATE, False, task_id="die_into_cup",
                                   task_definition=synthetic_task(0.1, 1), goal_channels=channels)


class FakeGoalSource:
    """Serves one fixed goal; counts reads and closes."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values, self.reads, self.closed = values, 0, False

    def read(self) -> GoalSample:
        self.reads += 1
        return GoalSample(self.values, 0.5)

    def close(self) -> None:
        self.closed = True


def test_contract_declares_channels_and_round_trips_only_when_present() -> None:
    """Six values for two channels; the payload gains the key only with channels."""
    two = contract()
    assert two.goal_size == 6 and two.as_dict()["goal_channels"] == ["object", "target"]
    assert PhysicalDatasetContract.from_dict(two.as_dict(fps=15)) == two
    none = contract(())
    assert "goal_channels" not in none.as_dict() and none.goal_size == 0
    assert none.check_goal(None) is None
    assert two.check_goal((1, 2, 3, 4, 5, 6)) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    with pytest.raises(ValueError, match="6 values"):
        two.check_goal((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="no goal channels"):
        none.check_goal((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="distinct names"):
        contract(("object", "object"))
    with pytest.raises(ValueError, match="distinct names"):
        contract(("cup",))
    assert feature_keys(two.goal_channels) == (*FEATURE_KEYS, GOAL_KEY)
    assert feature_keys(()) == FEATURE_KEYS


def test_frame_and_embodiment_carry_the_goal() -> None:
    """A frame holds a finite goal; the embodiment emits it exactly when declared."""
    state = PhysicalState((0.0,) * 6, 0.01, 1.0, 10.0)
    image = ImageFrame(np.zeros((4, 4, 3), np.uint8), 1.0)
    frame = PhysicalFrame(state, (0.0,) * 7, 1.0, image, image, goal=(0.1, 0.2, 0.0))
    assert frame.goal == (0.1, 0.2, 0.0)
    with pytest.raises(ValueError, match="three values"):
        PhysicalFrame(state, (0.0,) * 7, 1.0, image, image, goal=(0.1, 0.2))
    with pytest.raises(ValueError, match="goal"):
        PhysicalFrame(state, (0.0,) * 7, 1.0, image, image, goal=(0.1, 0.2, float("nan")))
    embodiment = SynriaEmbodiment(contract(("object",)))
    request = embodiment.observation(SynriaObservation(state, image, image, "t", (0.3, 0.2, 0.0)))
    assert request[GOAL_KEY] == [0.3, 0.2, 0.0] and request["goal_channels"] == ["object"]
    with pytest.raises(ValueError, match="3 values"):
        embodiment.observation(SynriaObservation(state, image, image, "t", None))
    plain = SynriaEmbodiment(contract(()))
    assert GOAL_KEY not in plain.observation(SynriaObservation(state, image, image, "t"))
    with pytest.raises(ValueError, match="no goal channels"):
        plain.observation(SynriaObservation(state, image, image, "t", (0.3, 0.2, 0.0)))


def test_recorder_requires_and_reads_a_goal_source_exactly_when_declared(tmp_path: Path) -> None:
    """Each capture reads the goal; the frames carry it; the source is closed."""
    clock = FakeClock()
    config = PhysicalRecorderConfig(dataset_path=tmp_path / "dataset", repo_id="local/synria",
                                    contract=replace(contract(("object",)),
                                                     action_source=ActionSource.LEADER))
    states = [_state(0.0, 0.0), _state(0.01, 1.0)]
    goal = FakeGoalSource((0.3, -0.1, 0.0))
    writer = FakeWriter(tmp_path)

    def build(goal_source: FakeGoalSource | None, recorder_config: PhysicalRecorderConfig):
        return PhysicalEpisodeRecorder(
            config=recorder_config, state_source=FakeStateSource(states),
            action_source=FakeLeaderActionSource([(0.01,) * 7 for _ in states]),
            wrist_source=FakeFrameSource("wrist", clock),
            front_source=FakeFrameSource("front", clock),
            writer=writer, clock=clock, goal_source=goal_source,
        )

    with pytest.raises(ValueError, match="goal source is required"):
        build(None, config)
    with pytest.raises(ValueError, match="goal source is required"):
        build(goal, replace(config, contract=replace(config.contract, goal_channels=())))
    recorder = build(goal, config)
    recorder.start()
    recorder.capture_once()
    clock.now = 0.5
    recorder.capture_once()
    recorder.stop()
    episode = recorder.mark_success()
    assert goal.reads == 2
    assert [frame.goal for frame in episode.frames] == [(0.3, -0.1, 0.0)] * 2
    recorder.close()
    assert goal.closed
    bad = FakeGoalSource((0.3, -0.1))
    recorder = build(bad, config)
    recorder.start()
    with pytest.raises(ValueError, match="3 values"):
        recorder.capture_once()


def test_writer_declares_and_stores_the_environment_state_feature(tmp_path: Path) -> None:
    """A dataset with channels carries the feature and its values; without, no feature."""
    config = replace(writer_config(tmp_path / "with"), image_width=32, image_height=32)
    config = replace(config, contract=replace(config.contract, goal_channels=("object", "target"),
                                              recording_purpose=SIMULATION))
    writer = LeRobotDatasetWriter(config)
    episode = synthetic_episode(width=32, height=32)
    episode = replace(episode, declared_recording_purpose=SIMULATION)
    episode.frames[:] = [replace(frame, goal=(0.3, 0.1, 0.0, 0.2, 0.05, 0.0))
                         for frame in episode.frames]
    writer.write_episode(episode)
    writer.finalize()
    info = json.loads((tmp_path / "with" / "meta" / "info.json").read_text())
    assert info["features"][GOAL_KEY]["shape"] == [6]
    assert json.loads((tmp_path / "with" / "physical_contract.json").read_text())[
        "goal_channels"] == ["object", "target"]
    import pyarrow.parquet as pq

    table = next(iter(sorted((tmp_path / "with" / "data").rglob("*.parquet"))))
    rows = pq.read_table(table, columns=[GOAL_KEY]).to_pylist()
    expected = [0.3, 0.1, 0.0, 0.2, 0.05, 0.0]
    assert [list(np.round(row[GOAL_KEY], 3)) for row in rows] == [expected] * 3
    missing = synthetic_episode(width=32, height=32)
    missing = replace(missing, declared_recording_purpose=SIMULATION, episode_index=1)
    again = LeRobotDatasetWriter(config)
    with pytest.raises(ValueError, match="6 values"):
        again.write_episode(missing)
    again.finalize()
    plain = replace(writer_config(tmp_path / "without"), image_width=32, image_height=32)
    writer = LeRobotDatasetWriter(plain)
    writer.write_episode(synthetic_episode(width=32, height=32))
    writer.finalize()
    info = json.loads((tmp_path / "without" / "meta" / "info.json").read_text())
    assert GOAL_KEY not in info["features"]


def test_goal_channel_options_parse_in_order() -> None:
    """NAME=TOPIC entries; duplicates, relative topics and malformed entries are refused."""
    assert parse_goal_channels(["object=/isaac/pick_objects/cup/pose",
                                "target=/isaac/pick_objects/place_target/pose"]) == {
        "object": "/isaac/pick_objects/cup/pose",
        "target": "/isaac/pick_objects/place_target/pose",
    }
    with pytest.raises(ValueError, match="NAME=TOPIC"):
        parse_goal_channels(["object"])
    with pytest.raises(ValueError, match="twice"):
        parse_goal_channels(["object=/a", "object=/b"])
    with pytest.raises(ValueError):
        parse_goal_channels(["object=relative/topic"])


def test_server_requires_the_goal_in_requests_when_the_checkpoint_declares_it(
    tmp_path: Path,
) -> None:
    """A goal-conditioned checkpoint refuses a request without its goal and accepts one with it."""
    identity, _, _ = fixture(tmp_path)
    channels = ("object",)
    contract_payload = {**identity.manifest["physical_contract"], "goal_channels": list(channels)}
    manifest = {
        **identity.manifest, "physical_contract": contract_payload,
        "declared_input_conditioning": [*FEATURE_KEYS, GOAL_KEY],
        "goal_channels": list(channels),
        "model_feature_shapes": {**identity.manifest["model_feature_shapes"], GOAL_KEY: [3]},
    }
    goal_identity = CheckpointIdentity(
        identity.checkpoint, identity.content_sha256, manifest,
        PhysicalDatasetContract.from_dict(contract_payload), identity.lane,
    )
    service = FixedScenePolicyService(goal_identity, StubModel())
    without = payload(goal_identity, reset=True)
    with pytest.raises(ValueError, match="request contract/task identity"):
        service.request(encode_policy_request(without))
    with_goal = {**without, GOAL_KEY: [0.3, -0.1, 0.0]}
    response = decode_policy_response(service.request(encode_policy_request(with_goal)))
    assert len(response["action"]) == 7
    with pytest.raises(ValueError, match="goal differs"):
        service.request(encode_policy_request({**without, GOAL_KEY: [0.3, -0.1]}))
    assert goal_identity.metadata()["physical_contract"]["goal_channels"] == ["object"]

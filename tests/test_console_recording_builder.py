"""Shared construction and synthetic-purpose boundaries without ROS or device access."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_physical_recorder import FakeClock, FakeWriter
from test_synria_policy_server import fixture
from test_task_registry import synthetic_registry

from synria_lerobot import recorder
from synria_lerobot.checkpoint_eval import strict_content_hash
from synria_lerobot.console.demo import SyntheticFrameSource, SyntheticStateSource
from synria_lerobot.evaluation import FIXED_D2_KIND, validate_fixed_protocol
from synria_lerobot.physical_contract import PhysicalDatasetContract, StateRateMeasurement
from synria_lerobot.policy_server import CheckpointIdentity
from synria_lerobot.quality_gates import EpisodeQualityRecord, episode_quality_record
from synria_lerobot.task_registry import SYNTHETIC_DEMO, recording_purpose


def settings(tmp_path: Path, *, demo: bool = False) -> argparse.Namespace:
    """Provide the same explicit configuration accepted by the reference CLI."""
    return argparse.Namespace(
        task_id="die_into_cup", task_registry=synthetic_registry(tmp_path / "tasks.json", 1, 30),
        dataset_path=tmp_path / "dataset", repo_id="local/console-test", gripper_type="50mm",
        action_source="next_state", state_has_velocity=False, smoke=False, fps=15,
        image_width=32, image_height=24, action_lookahead_steps=1,
        follower_topic="/joint_states", leader_topic="/leader/joint_states",
        state_startup_timeout_s=10.0, state_source=SYNTHETIC_DEMO if demo else "ros2_control",
        guard_command_topic=[], wrist_camera="fake-wrist", front_camera="fake-front",
    )


def test_builder_matches_config_and_reports_actual_preflights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console and terminal share configuration, source order and graph checks."""
    args = settings(tmp_path)
    expected = recorder.recording_config_from_args(args)
    clock = FakeClock()
    demo_config = replace(expected, contract=replace(
        expected.contract, recording_purpose=SYNTHETIC_DEMO,
    ), state_source_provenance=None)
    state = SyntheticStateSource(demo_config, clock=clock)
    measurement = StateRateMeasurement(50, 100, 2, 0, 2, 0.02)
    guards: list[tuple[str, ...]] = []
    source = SimpleNamespace(
        read=state.read, close=state.close, measure_rate=lambda: measurement,
        require_no_command_publishers=guards.append,
    )
    monkeypatch.setattr(recorder, "RosJointStateSource", lambda *a, **kw: source)
    monkeypatch.setattr(recorder, "OpenCVFrameSource", lambda *a, **kw: SimpleNamespace(
        close=lambda: None,
    ))
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", lambda config: FakeWriter(tmp_path))
    checks: dict[str, Any] = {}
    with recorder.build_recording_session(args, preflight=checks.__setitem__) as session:
        assert session.config == replace(expected, state_rate_measurement=measurement)
        assert guards == [("/joint_commands", "/policy_joint_targets")]
        assert checks["state_rate_evidence"]["rate_hz"] == 50
        assert all(checks[name]["passed"] for name in (
            "configuration", "velocity", "command_publishers", "dataset_lock_and_recovery",
            "wrist_camera", "front_camera", "command_topic:/joint_commands",
        ))
    assert state.closed


def test_demo_purpose_round_trips_without_physical_source_or_registry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained demos have explicit synthetic purpose rather than a physical smoke disguise."""
    pytest.importorskip("cv2")
    args = settings(tmp_path, demo=True)
    registry = json.loads(args.task_registry.read_text(encoding="utf-8"))
    for task in registry["tasks"]:
        task["episode_window"] = {"min_episode_s": None, "max_episode_s": None}
    args.task_registry.write_text(json.dumps(registry), encoding="utf-8")
    original = args.task_registry.read_bytes()
    clock = FakeClock()
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", lambda config: FakeWriter(tmp_path))
    monkeypatch.setattr(recorder, "RosJointStateSource", lambda *a, **kw: pytest.fail("ROS used"))
    monkeypatch.setattr(recorder, "OpenCVFrameSource", lambda *a, **kw: pytest.fail("camera used"))
    with recorder.build_recording_session(args, synthetic_demo=True, clock=clock) as session:
        contract = session.config.contract
        assert contract.recording_purpose == SYNTHETIC_DEMO
        assert (contract.min_episode_s, contract.max_episode_s) == (1, 30)
        assert contract.task_definition.min_episode_s is None
        assert session.config.state_source_provenance is None
        assert session.config.state_rate_measurement is None
        assert PhysicalDatasetContract.from_dict(contract.as_dict(fps=15)) == contract
        session.start()
        session.capture_once()
        clock.now = 1
        session.capture_once()
        session.stop()
        episode = session.mark_success()
        quality = episode_quality_record(episode, fps=15)
        assert episode.recording_purpose == quality.recording_purpose == SYNTHETIC_DEMO
        assert EpisodeQualityRecord.from_dict(quality.as_dict()) == quality
        assert episode.final_front_still.data.shape == (480, 640, 3)
        assert episode.frames[0].front.data.shape == (24, 32, 3)
    assert args.task_registry.read_bytes() == original


def test_synthetic_preview_reads_do_not_advance_sample(tmp_path: Path) -> None:
    """Repeated preview reads leave source time, frame index and trajectory unchanged."""
    pytest.importorskip("cv2")
    config = recorder.recording_config_from_args(settings(tmp_path, demo=True), synthetic_demo=True)
    clock = FakeClock()
    frames = SyntheticFrameSource("front", config, clock=clock)
    states = SyntheticStateSource(config, clock=clock)
    first = frames.read()
    assert all(frames.read() is first for _ in range(10))
    assert states.read() == states.read()
    assert first.native_resolution == (640, 480)
    assert not first.data.flags.writeable
    clock.now = 1
    assert frames.read().monotonic_timestamp_s == 1
    assert frames.read() is not first
    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames.read()


@pytest.mark.parametrize("consumer", ["server", "d2"])
def test_synthetic_contract_is_refused_by_public_physical_consumers(
    tmp_path: Path, consumer: str,
) -> None:
    """An otherwise hash-valid checkpoint or D2 protocol cannot legitimize demo data."""
    identity, receipt, registry = fixture(tmp_path)
    contract = replace(identity.contract, recording_purpose=SYNTHETIC_DEMO).as_dict(fps=30)
    if consumer == "d2":
        with pytest.raises(ValueError, match="synthetic demo"):
            validate_fixed_protocol({
                "kind": FIXED_D2_KIND, "task_id": "die_into_cup", "physical_contract": contract,
            })
        return
    path = identity.checkpoint / "physical_policy_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["physical_contract"] = contract
    manifest["configuration"]["expected_contract"] = copy.deepcopy(contract)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    record = {**manifest, "path": str(identity.checkpoint), "evaluation_status": "evaluated",
              "checkpoint_content_sha256": strict_content_hash(identity.checkpoint)}
    receipt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic demo"):
        CheckpointIdentity.load(identity.checkpoint, receipt, registry)


def test_purpose_cannot_mix_smoke_demo_or_physical_declaration(tmp_path: Path) -> None:
    """Keep the explicitly selected synthetic path separate from every physical mode."""
    args = settings(tmp_path)
    with pytest.raises(ValueError, match="physical state source"):
        recorder.recording_config_from_args(args, synthetic_demo=True)
    with pytest.raises(ValueError, match="smoke flag"):
        recording_purpose(True, SYNTHETIC_DEMO)
    args.smoke = True
    with pytest.raises(ValueError, match="distinct purposes"):
        recorder.recording_config_from_args(args, synthetic_demo=True)


def test_console_sources_contain_no_command_capabilities() -> None:
    """Enforce read-only ownership at source level without initializing any runtime."""
    package = Path(recorder.__file__).parent / "console"
    prohibited = {"create_publisher", "create_client", "ActionClient", "Serial", "VideoCapture"}
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else ""
                )
                assert name not in prohibited, (path, name)

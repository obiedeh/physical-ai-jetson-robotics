from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from test_task_registry import synthetic_task

from synria_lerobot.physical_contract import (
    CONTRACT_VERSION,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
    StateRateMeasurement,
    StateSourceProvenance,
    action_timing_metadata,
)
from synria_lerobot.physical_contract import (
    ActionSource as ActionSourceKind,
)
from synria_lerobot.quality_gates import (
    GATE_NAMES,
    OBJECT_SUCCESS_LIMITATION,
    EpisodeQualityRecord,
    FrameQualityRecord,
    GateConfig,
    ImageDiagnostic,
    episode_quality_record,
    evaluate_episode,
    load_limits,
    write_aggregate_summary,
    write_episode_records,
    write_session_artifacts,
)
from synria_lerobot.recorder import (
    ActionSample,
    NextStateActionSource,
    PhysicalEpisodeRecorder,
    PhysicalRecorderConfig,
    RecordedPhysicalEpisode,
)

LIMITS_PATH = Path(__file__).resolve().parents[1] / "config" / "synria_limits.yaml"


def _image(prefix: str, index: int) -> ImageDiagnostic:
    return ImageDiagnostic(True, 40.0 + index, f"{prefix}-{index}")


def _baseline_episode() -> EpisodeQualityRecord:
    frames = tuple(
        FrameQualityRecord(
            state=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01),
            action=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01),
            timestamps={
                "state_monotonic_s": index + 0.00,
                "state_ros_header_s": 1000.0 + index,
                "action_monotonic_s": index + 0.01,
                "wrist_monotonic_s": index + 0.02,
                "front_monotonic_s": index + 0.03,
                "sample_monotonic_s": index + 0.04,
                "state_ros_arrival_s": 1000.0 + index + 0.01,
                "action_ros_header_s": 1000.0 + index,
                "action_ros_arrival_s": 1000.0 + index + 0.01,
            },
            wrist=_image("wrist", index),
            front=_image("front", index),
        )
        for index in range(20)
    )
    return EpisodeQualityRecord(
        episode_index=0,
        fps=1.0,
        duration_s=20.0,
        operator_label="success",
        action_source="leader",
        contract_version=CONTRACT_VERSION,
        gripper_type="50mm",
        final_still="episode_000000_final.jpg",
        smoke=False,
        frames=frames,
        **action_timing_metadata(ActionSourceKind.LEADER, 1, 1.0),
        state_rate_measurement=asdict(StateRateMeasurement(50, 100, 2, 0, 2, 0.02)),
        task_definition=synthetic_task(20, 30),
    )


def _failed_gates(episode: EpisodeQualityRecord) -> list[str]:
    report = evaluate_episode(episode, load_limits(LIMITS_PATH), GateConfig(20, 30))
    assert tuple(report.gates) == GATE_NAMES
    return report.failed_gates


def _replace_frame(
    episode: EpisodeQualityRecord, index: int, frame: FrameQualityRecord
) -> EpisodeQualityRecord:
    frames = list(episode.frames)
    frames[index] = frame
    return replace(episode, frames=tuple(frames))


def test_baseline_passes_all_gates() -> None:
    assert _failed_gates(_baseline_episode()) == []


def test_corrupt_frame_count_fails_exact_gate() -> None:
    episode = _baseline_episode()
    assert _failed_gates(replace(episode, frames=episode.frames[:-4])) == ["frame_count"]


def test_corrupt_nan_fails_exact_gate() -> None:
    episode = _baseline_episode()
    frame = replace(episode.frames[0], state=(float("nan"), *episode.frames[0].state[1:]))
    assert _failed_gates(_replace_frame(episode, 0, frame)) == ["finite_state_action"]


def test_corrupt_missing_camera_fails_exact_gate() -> None:
    episode = _baseline_episode()
    frame = replace(episode.frames[0], wrist=ImageDiagnostic(False, 0.0, ""))
    assert _failed_gates(_replace_frame(episode, 0, frame)) == ["cameras_present"]


def test_corrupt_limit_fails_exact_gate() -> None:
    episode = _baseline_episode()
    frame = replace(episode.frames[0], action=(9.0, *episode.frames[0].action[1:]))
    assert _failed_gates(_replace_frame(episode, 0, frame)) == ["state_action_limits"]


def test_corrupt_timestamp_skew_fails_exact_gate() -> None:
    episode = _baseline_episode()
    timestamps = dict(episode.frames[0].timestamps)
    timestamps["front_monotonic_s"] = 0.1
    timestamps["sample_monotonic_s"] = 0.11
    frame = replace(episode.frames[0], timestamps=timestamps)
    assert _failed_gates(_replace_frame(episode, 0, frame)) == ["timestamp_skew"]


@pytest.mark.parametrize(
    "corruption", ["old_sources", "header_delay", "future_header", "leader_age", "missing_sample",
                   "missing_header_arrival", "repeated_sample", "backwards_sample"]
)
def test_corrupt_source_freshness_fails_exact_gate(corruption: str) -> None:
    episode = _baseline_episode()
    index = 1 if corruption in {"repeated_sample", "backwards_sample"} else 0
    timestamps = dict(episode.frames[index].timestamps)
    if corruption == "old_sources":
        for name in (
            "state_monotonic_s", "action_monotonic_s", "wrist_monotonic_s", "front_monotonic_s"
        ):
            timestamps[name] -= 1
    elif corruption == "header_delay":
        timestamps["state_ros_arrival_s"] += 1
    elif corruption == "future_header":
        timestamps["state_ros_header_s"] += 1
    elif corruption == "leader_age":
        timestamps.update(state_monotonic_s=0.04, wrist_monotonic_s=0.04, front_monotonic_s=0.04,
                          action_monotonic_s=0.0, sample_monotonic_s=0.21)
    elif corruption == "missing_sample":
        timestamps.pop("sample_monotonic_s")
    elif corruption == "missing_header_arrival":
        timestamps.pop("state_ros_arrival_s")
    else:
        timestamps["sample_monotonic_s"] = 0.04 if corruption == "repeated_sample" else 0.03
    frame = replace(episode.frames[index], timestamps=timestamps)
    assert _failed_gates(_replace_frame(episode, index, frame)) == ["source_staleness"]


@pytest.mark.parametrize(
    "corruption", [None, "missing", "unrelated", "nonfinite", "unknown_source"]
)
@pytest.mark.parametrize("steps", [0, 1, 2, 25])
def test_derived_actions_require_true_next_state_timestamp_lineage(
    corruption: str | None, steps: int
) -> None:
    episode = _baseline_episode()
    frames = []
    for index, frame in enumerate(episode.frames):
        target = episode.frames[min(index + steps, len(episode.frames) - 1)].timestamps
        timestamps = {
            **frame.timestamps, "action_monotonic_s": target["state_monotonic_s"],
            "action_ros_header_s": target["state_ros_header_s"],
            "action_ros_arrival_s": target["state_ros_arrival_s"],
        }
        frames.append(replace(frame, timestamps=timestamps))
    if corruption == "missing":
        frames[0].timestamps.pop("action_ros_arrival_s")
    elif corruption == "unrelated":
        frames[0].timestamps["action_monotonic_s"] += 1
    elif corruption == "nonfinite":
        frames[0].timestamps["action_ros_header_s"] = float("nan")
    episode = replace(
        episode, frames=tuple(frames), action_source=(
            "unknown" if corruption == "unknown_source" else "next_state"
        ), **action_timing_metadata(ActionSourceKind.NEXT_STATE, steps, 1.0),
    )
    assert _failed_gates(episode) == ([] if corruption is None else ["source_staleness"])


def test_corrupt_duration_fails_exact_gate() -> None:
    assert _failed_gates(replace(_baseline_episode(), duration_s=19.0)) == [
        "episode_length"
    ]


def test_corrupt_gripper_dimension_fails_exact_gate() -> None:
    episode = _baseline_episode()
    frame = replace(episode.frames[0], state=episode.frames[0].state[:6])
    assert _failed_gates(_replace_frame(episode, 0, frame)) == [
        "gripper_dimensionality"
    ]


def test_corrupt_visual_stream_fails_exact_gate() -> None:
    episode = _baseline_episode()
    frames = tuple(
        replace(frame, wrist=ImageDiagnostic(True, 40.0, "frozen"))
        for frame in episode.frames
    )
    assert _failed_gates(replace(episode, frames=frames)) == ["visual_sanity"]


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class _StateSource:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.count = 0

    def read(self) -> PhysicalState:
        value = 0.1 + self.count * 0.001
        self.count += 1
        return PhysicalState(
            joint_positions_rad=(value,) * 6,
            gripper_m=0.01,
            monotonic_timestamp_s=self.clock(),
            ros_header_stamp_s=1000.0 + self.clock(),
            ros_arrival_stamp_s=1000.0 + self.clock(),
        )

    def close(self) -> None:
        return None


class _LeaderActions:
    kind = ActionSourceKind.LEADER

    def read(self, follower_state: PhysicalState) -> ActionSample:
        return ActionSample(
            (*follower_state.joint_positions_rad, follower_state.gripper_m),
            follower_state.monotonic_timestamp_s,
            follower_state.ros_header_stamp_s,
            follower_state.ros_arrival_stamp_s,
        )

    def close(self) -> None:
        return None


class _Frames:
    def __init__(self, clock: _Clock, offset: int) -> None:
        self.clock = clock
        self.offset = offset
        self.count = 0

    def read(self) -> ImageFrame:
        self.count += 1
        value = (self.offset + self.count) % 250 + 2
        return ImageFrame(
            data=np.full((4, 4, 3), value, dtype=np.uint8),
            monotonic_timestamp_s=self.clock(),
        )

    def close(self) -> None:
        return None


class _Writer:
    next_episode_index = 0
    recovery_blocked = False

    def finalize(self) -> None:
        return None

    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes: list[RecordedPhysicalEpisode] = []
        root.mkdir(parents=True, exist_ok=True)

    def save_final_still(self, episode_index: int, frame: ImageFrame) -> Path:
        path = self.root / f"episode_{episode_index:06d}_final.bin"
        path.write_bytes(np.asarray(frame.data).tobytes())
        return path

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None:
        episode.final_still_path = self.save_final_still(
            episode.episode_index, episode.frames[-1].front
        )
        self.episodes.append(episode)
        (self.root / f"episode_{episode.episode_index:06d}.json").write_text(
            json.dumps({"frames": len(episode.frames), "label": episode.operator_label.value})
        )


def _record_two_episodes(
    root: Path, action_source: ActionSourceKind
) -> list[RecordedPhysicalEpisode]:
    clock = _Clock()
    contract = PhysicalDatasetContract(
        gripper_type="50mm",
        action_source=action_source,
        state_has_velocity=False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    )
    recorder = PhysicalEpisodeRecorder(
        config=PhysicalRecorderConfig(
            dataset_path=root,
            repo_id=f"local/{action_source.value}",
            contract=contract,
            fps=1.0,
        ),
        state_source=_StateSource(clock),
        action_source=(
            _LeaderActions()
            if action_source is ActionSourceKind.LEADER
            else NextStateActionSource()
        ),
        wrist_source=_Frames(clock, 10),
        front_source=_Frames(clock, 100),
        writer=_Writer(root),
        clock=clock,
    )
    episodes = []
    for episode_number in range(2):
        start = episode_number * 40.0
        clock.now = start
        recorder.start()
        for frame_index in range(20):
            clock.now = start + frame_index
            recorder.capture_once()
        clock.now = start + 20.0
        recorder.stop()
        episodes.append(recorder.mark_success())
    return episodes


def _provenance(action_source: str) -> dict[str, object]:
    return {
        "follower_serial": "ADF-unverified",
        "leader_serial": "ADL-unverified",
        "host": "fake-host",
        "git_sha": "test-sha",
        "utc_date": "2026-10-07",
        "operator": "test-operator",
        "power_state_start": "test-only",
        "power_state_end": "test-only",
        "scene": "synthetic test scene",
        "camera_ids": {"wrist": "fake-wrist", "front": "fake-front"},
        "resolution": {"width": 4, "height": 4},
        "rate_hz": 1.0,
        **synthetic_task(20, 30).metadata(),
        "contract_version": CONTRACT_VERSION,
        "gripper_type": "50mm",
        "action_source": action_source,
        **action_timing_metadata(ActionSourceKind(action_source), 1, 1.0),
    }


@pytest.mark.parametrize("layer", ["provenance", "episode", "capture"])
def test_session_refuses_a_different_valid_task_snapshot_before_writes(
    tmp_path: Path, layer: str,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    original = synthetic_task(20, 30)
    other = synthetic_task(10, 80, "cup_return")
    contract = PhysicalDatasetContract(
        "50mm", ActionSourceKind.LEADER, False,
        task_id=original.task_id, task_definition=original,
    )
    (dataset / "physical_contract.json").write_text(json.dumps(contract.as_dict(fps=1)))
    declaration = StateSourceProvenance("standalone_driver", "/joint_states").as_dict()
    provenance = {**_provenance("leader"), "state_source_provenance": declaration}
    episode = replace(_baseline_episode(), state_source_provenance=declaration)
    capture = {
        "episode_index": 0, "camera_ids": provenance["camera_ids"],
        "native_resolution": {"wrist": {"width": 4, "height": 4},
                              "front": {"width": 4, "height": 4}},
        "stored_resolution": provenance["resolution"], "stored_color_space": "RGB",
        "achieved_sample_rate_hz": None, "state_rate_measurement": episode.state_rate_measurement,
        "state_source_provenance": declaration, **contract.as_dict(fps=1),
    }
    if layer == "episode":
        episode = replace(episode, task_definition=other)
    else:
        {"provenance": provenance, "capture": capture}[layer].update(other.metadata())
    (dataset / "physical_capture_provenance.jsonl").write_text(json.dumps(capture) + "\n")
    session = tmp_path / "reports" / "session"
    with pytest.raises(ValueError, match="differs"):
        write_session_artifacts(
            session_dir=session, dataset_path=dataset, provenance=provenance, episodes=[episode],
            limits=load_limits(LIMITS_PATH), gate_config=GateConfig(20, 30),
        )
    assert not session.exists()


def test_aggregate_keeps_distinct_task_windows_per_dataset(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    tasks = [synthetic_task(20, 30), synthetic_task(10, 80, "cup_return")]
    for index, task in enumerate(tasks):
        dataset = tmp_path / f"dataset-{index}"
        dataset.mkdir()
        write_session_artifacts(
            session_dir=reports / f"session-{index}", dataset_path=dataset,
            provenance={**_provenance("leader"), **task.metadata()},
            episodes=[replace(_baseline_episode(), task_definition=task)],
            limits=load_limits(LIMITS_PATH), gate_config=GateConfig(**task.require_configured()),
        )
    aggregate = write_aggregate_summary(
        data_root=reports, output_path=reports / "aggregate.json",
        timeline_path=tmp_path / "timeline.jsonl",
    )
    rows = aggregate["recording_contracts"]
    assert [row["task_id"] for row in rows] == [task.task_id for task in tasks]
    assert [row["max_episode_s"] for row in rows] == [30, 80]
    summary_path = reports / "session-0" / "session_summary.json"
    payload = json.loads(summary_path.read_text())
    payload.pop("task_definition")
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task definition"):
        write_aggregate_summary(
            data_root=reports, output_path=reports / "not-written.json",
            timeline_path=tmp_path / "not-written.jsonl",
        )
    assert not (reports / "not-written.json").exists()


def test_fake_source_end_to_end_for_both_action_sources(tmp_path: Path) -> None:
    limits = load_limits(LIMITS_PATH)
    data_root = tmp_path / "reports" / "data"
    for source in (ActionSourceKind.LEADER, ActionSourceKind.NEXT_STATE):
        dataset = tmp_path / "datasets" / source.value
        episodes = _record_two_episodes(dataset, source)
        records = [episode_quality_record(episode, fps=1.0) for episode in episodes]
        write_episode_records(records, dataset / "physical_quality_records.jsonl")
        summary = write_session_artifacts(
            session_dir=data_root / source.value,
            dataset_path=dataset,
            provenance=_provenance(source.value),
            episodes=records,
            limits=limits,
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
        assert summary["episode_count"] == 2
        assert summary["quality_valid_episode_count"] == 2, summary["gate_results"]
        assert summary["object_success_limitation"] == OBJECT_SUCCESS_LIMITATION
        assert summary["limits_status"] == "limits unverified by operator"

    aggregate = write_aggregate_summary(
        data_root=data_root,
        output_path=data_root / "D1_dataset_summary.json",
        timeline_path=tmp_path / "timeline.jsonl",
        now_utc="2026-10-07T00:00:00+00:00",
    )
    assert aggregate["quality_valid_episode_count"] == 4
    assert aggregate["qualifying_episode_count"] == 0
    assert aggregate["status"] == "planned"
    assert aggregate["action_sources"] == ["leader", "next_state"]
    assert [item["nominal_action_lookahead_s"] for item in aggregate["recording_contracts"]] == [
        0.0, 1.0
    ]
    assert (tmp_path / "timeline.jsonl").read_text().count("d1_dataset_summary") == 1


@pytest.mark.parametrize("configured", [False, True])
def test_smoke_is_diagnosed_but_never_qualifies_or_affects_demo_success(
    tmp_path: Path, configured: bool,
) -> None:
    from test_task_registry import REGISTRY

    from synria_lerobot.task_registry import load_task_registry

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    task = synthetic_task(20, 30) if configured else load_task_registry(REGISTRY)["die_into_cup"]
    limits = replace(load_limits(LIMITS_PATH), verified_by="test", verified_on="2026-10-08")
    summary = write_session_artifacts(
        session_dir=tmp_path / "reports" / "session", dataset_path=dataset,
        provenance={**_provenance("leader"),
                    **task.metadata("disposable_smoke")},
        episodes=[replace(_baseline_episode(), smoke=True, task_definition=task)], limits=limits,
        gate_config=GateConfig(min_episode_s=20, max_episode_s=20),
    )
    assert summary["recorded_episode_count"] == 1
    assert summary["smoke_episode_count"] == 1
    assert summary["episode_count"] == 0
    assert summary["quality_valid_episode_count"] == 0
    assert summary["qualifying_episode_count"] == 0
    assert summary["demonstration_success_rate"] == 0
    assert len(summary["gate_results"]) == 1
    aggregate = write_aggregate_summary(
        data_root=tmp_path / "reports", output_path=tmp_path / "aggregate.json",
        timeline_path=tmp_path / "timeline.jsonl",
    )
    assert aggregate["qualifying_episode_count"] == 0
    assert aggregate["quality_valid_episode_count"] == 0
    recorded = aggregate["recording_contracts"][0]
    assert recorded["recording_purpose"] == "disposable_smoke"
    assert recorded["task_definition"] == task.as_dict()
    assert recorded["task_definition_sha256"] == task.sha256


@pytest.mark.parametrize("smoke", [False, 0, 1, "true", None])
def test_quality_record_cannot_relabel_disposable_episode(smoke: object) -> None:
    payload = replace(_baseline_episode(), smoke=True).as_dict()
    payload["smoke"] = smoke
    with pytest.raises(ValueError, match="smoke"):
        EpisodeQualityRecord.from_dict(payload)


@pytest.mark.parametrize("field", [
    "episode_count", "quality_valid_episode_count", "qualifying_episode_count",
    "episode_indices", "operator_labels", "demonstration_success_rate",
])
def test_aggregate_refuses_smoke_counts_tampering(tmp_path: Path, field: str) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data = tmp_path / "reports"
    task = synthetic_task(20, 30)
    summary = write_session_artifacts(
        session_dir=data / "smoke", dataset_path=dataset,
        provenance={**_provenance("leader"), **task.metadata("disposable_smoke")},
        episodes=[replace(_baseline_episode(), smoke=True)], limits=load_limits(LIMITS_PATH),
        gate_config=GateConfig(min_episode_s=20, max_episode_s=20),
    )
    summary[field] = [0] if field in {"episode_indices", "operator_labels"} else 1
    (data / "smoke" / "session_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="disposable smoke summary"):
        write_aggregate_summary(
            data_root=data, output_path=tmp_path / "aggregate.json",
            timeline_path=tmp_path / "timeline.jsonl",
        )
    assert not (tmp_path / "aggregate.json").exists()
    assert not (tmp_path / "timeline.jsonl").exists()


def test_resumed_dataset_reuses_summary_and_cannot_be_counted_twice(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reports = tmp_path / "reports"
    limits = load_limits(LIMITS_PATH)
    kwargs = dict(
        dataset_path=dataset, provenance=_provenance("leader"),
        episodes=[_baseline_episode()], limits=limits,
    )
    write_session_artifacts(session_dir=reports / "session", **kwargs,
        gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
    )
    with pytest.raises(ValueError, match="already summarized"):
        write_session_artifacts(session_dir=reports / "duplicate", **kwargs,
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
    assert not (reports / "duplicate").exists()
    kwargs["episodes"] = [_baseline_episode(), replace(_baseline_episode(), episode_index=1)]
    summary = write_session_artifacts(session_dir=reports / "session", **kwargs,
        gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
    )
    assert summary["episode_count"] == 2
    duplicate = reports / "duplicate"
    duplicate.mkdir()
    (duplicate / "session_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="double counting"):
        write_aggregate_summary(
            data_root=reports, output_path=reports / "aggregate.json",
            timeline_path=tmp_path / "timeline.jsonl",
        )


@pytest.mark.parametrize("bad_index", [0, -1, 0.5, True, "1"])
def test_session_refuses_duplicate_or_invalid_indices_before_writing(
    tmp_path: Path, bad_index: object
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    session = tmp_path / "reports" / "session"
    limits = replace(load_limits(LIMITS_PATH), verified_by="test", verified_on="2026-10-08")
    episodes = [
        _baseline_episode(),
        replace(_baseline_episode(), episode_index=bad_index, smoke=True),
    ]
    with pytest.raises(ValueError, match="episode indices"):
        write_session_artifacts(
            session_dir=session, dataset_path=dataset,
            provenance=_provenance("leader"), episodes=episodes, limits=limits,
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
    assert not session.exists()


@pytest.mark.parametrize("bad_index", [-1, 0.5, True, "1"])
def test_quality_record_parser_does_not_coerce_invalid_episode_indices(bad_index: object) -> None:
    payload = _baseline_episode().as_dict()
    payload["episode_index"] = bad_index
    with pytest.raises(ValueError, match="episode indices"):
        EpisodeQualityRecord.from_dict(payload)


@pytest.mark.parametrize("layer", ["contract", "provenance", "episode", "capture"])
@pytest.mark.parametrize(
    "field,value",
    [("action_source", "next_state"), ("contract_version", "unknown"),
     ("gripper_type", "100mm"), ("requested_rate_hz", 2), ("action_lookahead_steps", 2),
     ("action_lookahead_steps", True), ("effective_action_lookahead_steps", 1),
     ("nominal_action_lookahead_s", 0.5)],
)
def test_summary_refuses_relabelled_contract_or_episode_evidence(
    tmp_path: Path, layer: str, field: str, value: object
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    contract = PhysicalDatasetContract("50mm", ActionSourceKind.LEADER, False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    ).as_dict(fps=1)
    provenance = _provenance("leader")
    declaration = StateSourceProvenance("standalone_driver", "/joint_states").as_dict()
    provenance["state_source_provenance"] = declaration
    episode = replace(_baseline_episode(), state_source_provenance=declaration)
    capture = {
        "episode_index": 0, "camera_ids": provenance["camera_ids"],
        "native_resolution": {"wrist": {"width": 4, "height": 4},
                              "front": {"width": 4, "height": 4}},
        "stored_resolution": provenance["resolution"], "stored_color_space": "RGB",
        "achieved_sample_rate_hz": None,
        "state_rate_measurement": episode.state_rate_measurement,
        "state_source_provenance": declaration,
        "action_source": "leader", "contract_version": CONTRACT_VERSION, "gripper_type": "50mm",
        **action_timing_metadata(ActionSourceKind.LEADER, 1, 1),
        **synthetic_task(20, 30).metadata(),
    }
    if layer == "episode":
        episode = replace(episode, **{field: value})
    else:
        target = {"contract": contract, "provenance": provenance, "capture": capture}[layer]
        target[field] = value
    (dataset / "physical_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (dataset / "physical_capture_provenance.jsonl").write_text(
        json.dumps(capture) + "\n", encoding="utf-8"
    )
    session = tmp_path / "reports" / "session"
    with pytest.raises(ValueError):
        write_session_artifacts(
            session_dir=session, dataset_path=dataset, provenance=provenance,
            episodes=[episode], limits=load_limits(LIMITS_PATH),
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
    assert not session.exists()


@pytest.mark.parametrize("measurement", [None, {"rate_hz": 50}, {
    "rate_hz": 0.5, "message_count": 1, "duration_s": 2,
    "started_monotonic_s": 0, "ended_monotonic_s": 2, "max_callback_gap_s": 2,
}])
def test_physical_summary_requires_complete_sufficient_incoming_rate_evidence(
    tmp_path: Path, measurement: dict[str, object] | None
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    contract = PhysicalDatasetContract("50mm", ActionSourceKind.LEADER, False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    ).as_dict(fps=1)
    (dataset / "physical_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    session = tmp_path / "reports" / "session"
    declaration = StateSourceProvenance("standalone_driver", "/joint_states").as_dict()
    with pytest.raises((ValueError, TypeError)):
        write_session_artifacts(
            session_dir=session, dataset_path=dataset,
            provenance={**_provenance("leader"), "state_source_provenance": declaration},
            episodes=[replace(_baseline_episode(), state_rate_measurement=measurement,
                              state_source_provenance=declaration)],
            limits=load_limits(LIMITS_PATH),
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
    assert not session.exists()


@pytest.mark.parametrize("layer", ["provenance", "episode", "capture"])
@pytest.mark.parametrize("corruption", ["missing", "kind", "topic", "guards", "declaration"])
def test_physical_summary_refuses_missing_or_conflicting_source_evidence(
    tmp_path: Path, layer: str, corruption: str
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    contract = PhysicalDatasetContract("50mm", ActionSourceKind.LEADER, False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    ).as_dict(fps=1)
    (dataset / "physical_contract.json").write_text(json.dumps(contract))
    declaration = StateSourceProvenance("standalone_driver", "/joint_states").as_dict()
    provenance = {**_provenance("leader"), "state_source_provenance": declaration}
    episode = replace(_baseline_episode(), state_source_provenance=declaration)
    capture = {
        "episode_index": 0, "camera_ids": provenance["camera_ids"],
        "native_resolution": {"wrist": {"width": 4, "height": 4},
                              "front": {"width": 4, "height": 4}},
        "stored_resolution": provenance["resolution"], "stored_color_space": "RGB",
        "achieved_sample_rate_hz": None, "state_rate_measurement": episode.state_rate_measurement,
        "state_source_provenance": declaration,
        "action_source": "leader", "contract_version": CONTRACT_VERSION, "gripper_type": "50mm",
        **action_timing_metadata(ActionSourceKind.LEADER, 1, 1),
        **synthetic_task(20, 30).metadata(),
    }
    changed = dict(declaration)
    if corruption == "missing":
        broken = None
    else:
        name, value = {
            "kind": ("state_source", "ros2_control"),
            "topic": ("follower_topic", "/different_states"),
            "guards": ("guarded_command_topics", ["/joint_commands"]),
            "declaration": ("declaration", "automatically-identified"),
        }[corruption]
        changed[name] = value
        broken = changed
    if layer == "episode":
        episode = replace(episode, state_source_provenance=broken)
    else:
        {"provenance": provenance, "capture": capture}[layer]["state_source_provenance"] = broken
    (dataset / "physical_capture_provenance.jsonl").write_text(json.dumps(capture) + "\n")
    session = tmp_path / "reports" / "session"
    with pytest.raises(ValueError, match="state source|command topics"):
        write_session_artifacts(
            session_dir=session, dataset_path=dataset, provenance=provenance,
            episodes=[episode], limits=load_limits(LIMITS_PATH),
            gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
        )
    assert not session.exists()


@pytest.mark.parametrize("corruption", [
    None, "legacy", "resolution", "timestamp", "source", "color", "null", "bool_resolution",
])
def test_summary_validates_native_still_evidence_and_marks_legacy_unknown(
    tmp_path: Path, corruption: str | None,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episode = _baseline_episode()
    provenance = _provenance("leader")
    capture = {
        "episode_index": 0, "camera_ids": provenance["camera_ids"],
        "native_resolution": {"wrist": {"width": 640, "height": 480},
                              "front": {"width": 640, "height": 480}},
        "stored_resolution": provenance["resolution"], "stored_color_space": "RGB",
        "achieved_sample_rate_hz": None, "state_rate_measurement": episode.state_rate_measurement,
        "state_source_provenance": None, "action_source": "leader",
        "contract_version": CONTRACT_VERSION, "gripper_type": "50mm",
        **action_timing_metadata(ActionSourceKind.LEADER, 1, 1),
        **synthetic_task(20, 30).metadata(),
    }
    still = {
        "resolution": {"width": 640, "height": 480}, "color_space": "RGB",
        "source_id": provenance["camera_ids"]["front"],
        "source_monotonic_s": episode.frames[-1].timestamps["front_monotonic_s"],
    }
    changes = {
        "resolution": ("resolution", {"width": 224, "height": 224}),
        "timestamp": ("source_monotonic_s", 999), "source": ("source_id", "other"),
        "color": ("color_space", "BGR"),
    }
    if corruption in changes:
        name, value = changes[corruption]
        still[name] = value
    if corruption != "legacy":
        capture["final_still_capture"] = None if corruption == "null" else still
    if corruption == "bool_resolution":
        capture["native_resolution"]["front"]["width"] = 1
        still["resolution"]["width"] = True
    (dataset / "physical_capture_provenance.jsonl").write_text(json.dumps(capture) + "\n")
    session = tmp_path / "reports" / "session"
    if corruption not in (None, "legacy"):
        with pytest.raises(ValueError, match="final still capture"):
            write_session_artifacts(
                session_dir=session, dataset_path=dataset, provenance=provenance,
                episodes=[episode], limits=load_limits(LIMITS_PATH),
                gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
            )
        assert not session.exists()
        return
    summary = write_session_artifacts(
        session_dir=session, dataset_path=dataset, provenance=provenance,
        episodes=[episode], limits=load_limits(LIMITS_PATH),
        gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
    )
    expected = [{
        "episode_index": 0,
        "status": (
            "unknown: legacy capture has no native still evidence" if corruption else "native"
        ),
        "capture": None if corruption else still,
    }]
    assert summary["final_still_captures"] == expected
    assert json.loads((session / "provenance.json").read_text())["final_still_captures"] == expected

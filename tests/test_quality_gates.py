from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from synria_lerobot.physical_contract import (
    CONTRACT_VERSION,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.physical_contract import (
    ActionSource as ActionSourceKind,
)
from synria_lerobot.quality_gates import (
    GATE_NAMES,
    OBJECT_SUCCESS_LIMITATION,
    EpisodeQualityRecord,
    FrameQualityRecord,
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
    )


def _failed_gates(episode: EpisodeQualityRecord) -> list[str]:
    report = evaluate_episode(episode, load_limits(LIMITS_PATH))
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
    timestamps["front_monotonic_s"] = 1.0
    frame = replace(episode.frames[0], timestamps=timestamps)
    assert _failed_gates(_replace_frame(episode, 0, frame)) == ["timestamp_skew"]


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
        )

    def close(self) -> None:
        return None


class _LeaderActions:
    kind = ActionSourceKind.LEADER

    def read(self, follower_state: PhysicalState) -> ActionSample:
        return ActionSample(
            (*follower_state.joint_positions_rad, follower_state.gripper_m),
            follower_state.monotonic_timestamp_s,
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

    def __init__(self, root: Path) -> None:
        self.root = root
        self.episodes: list[RecordedPhysicalEpisode] = []
        root.mkdir(parents=True, exist_ok=True)

    def save_final_still(self, episode_index: int, frame: ImageFrame) -> Path:
        path = self.root / f"episode_{episode_index:06d}_final.bin"
        path.write_bytes(np.asarray(frame.data).tobytes())
        return path

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None:
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
        "contract_version": CONTRACT_VERSION,
        "gripper_type": "50mm",
        "action_source": action_source,
    }


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
    assert (tmp_path / "timeline.jsonl").read_text().count("d1_dataset_summary") == 1

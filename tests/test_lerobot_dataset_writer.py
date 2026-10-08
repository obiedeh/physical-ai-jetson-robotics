"""Dataset-boundary regressions; all images and state are synthetic."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from synria_lerobot.physical_contract import (
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalFrame,
    PhysicalState,
    StateRateMeasurement,
    StateSourceProvenance,
    action_timing_metadata,
)
from synria_lerobot.recorder import (
    LeRobotDatasetWriter,
    NextStateActionSource,
    OperatorLabel,
    PhysicalEpisodeRecorder,
    PhysicalRecorderConfig,
    PhysicalSessionResult,
    RecordedPhysicalEpisode,
    RecorderState,
)


def writer_config(root: Path) -> PhysicalRecorderConfig:
    return PhysicalRecorderConfig(
        dataset_path=root,
        repo_id="local/synria-test",
        contract=PhysicalDatasetContract(
            gripper_type="50mm",
            action_source=ActionSource.NEXT_STATE,
            state_has_velocity=False,
        ),
        fps=15,
    )


def image_features() -> dict[str, dict[str, object]]:
    features: dict[str, dict[str, object]] = {
        f"observation.images.{name}": {"dtype": "video", "shape": (3, 224, 224)}
        for name in ("wrist", "front")
    }
    features["observation.state"] = {"dtype": "float32", "shape": (7,)}
    features["action"] = {"dtype": "float32", "shape": (7,)}
    for name in (
        "state_monotonic_s", "state_ros_header_s", "action_monotonic_s",
        "wrist_monotonic_s", "front_monotonic_s", "sample_monotonic_s",
        "state_ros_arrival_s", "action_ros_header_s", "action_ros_arrival_s",
    ):
        features[name] = {"dtype": "float64", "shape": (1,)}
    return features


def _install_fake_library(monkeypatch: pytest.MonkeyPatch, library: type) -> None:
    package = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    datasets.DEFAULT_EPISODES_PATH = (  # type: ignore[attr-defined]
        "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    module = ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = library  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


def test_create_owns_root_and_contract_is_written_after_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = writer_config(tmp_path / "nested" / "dataset")

    class Dataset:
        @staticmethod
        def create(**kwargs: Any) -> SimpleNamespace:
            root = kwargs["root"]
            assert not root.exists()
            root.mkdir(parents=True, exist_ok=False)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            assert not (root / "physical_contract.json").exists()
            return SimpleNamespace(
                meta=SimpleNamespace(total_episodes=0, total_frames=0), finalize=lambda: None
            )

    _install_fake_library(monkeypatch, Dataset)
    writer = LeRobotDatasetWriter(config)
    assert writer.next_episode_index == 0
    assert json.loads((config.dataset_path / "physical_contract.json").read_text()) == (
        config.contract.as_dict(fps=config.fps)
    )
    writer.finalize()


def test_resume_uses_dataset_metadata_and_refuses_contract_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = writer_config(tmp_path / "dataset")
    (config.dataset_path / "meta").mkdir(parents=True)
    (config.dataset_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    contract_path = config.dataset_path / "physical_contract.json"
    contract_path.write_text(json.dumps(config.contract.as_dict(fps=config.fps)), encoding="utf-8")
    calls = []

    class Dataset:
        @staticmethod
        def resume(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                meta=SimpleNamespace(
                    total_episodes=7, features=image_features(), episodes=[], fps=15
                ),
                finalize=lambda: None,
            )

    _install_fake_library(monkeypatch, Dataset)
    monkeypatch.setattr(LeRobotDatasetWriter, "_validate_segments", lambda self: None)
    writer = LeRobotDatasetWriter(config)
    assert writer.next_episode_index == 7
    assert calls == [{"repo_id": config.repo_id, "root": config.dataset_path}]
    writer.finalize()
    with pytest.raises(ValueError, match="image features differ"):
        LeRobotDatasetWriter(replace(config, image_width=48))
    contract_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="contract differs"):
        LeRobotDatasetWriter(config)
    assert len(calls) == 2


def test_existing_empty_root_is_rejected_without_opening_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_library(monkeypatch, type("UnusedDataset", (), {}))
    with pytest.raises(ValueError, match="must contain a physical LeRobot dataset"):
        LeRobotDatasetWriter(writer_config(tmp_path))


def test_smoke_uses_nonexisting_child_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from synria_lerobot import recorder

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recorder",
            "--repo-id", "local/smoke",
            "--gripper-type", "50mm",
            "--action-source", "next_state",
            "--wrist-camera", "/unused/wrist",
            "--front-camera", "/unused/front",
            "--image-width", "48",
            "--image-height", "32",
            "--smoke",
            "--fps", "15",
            "--state-source", "standalone_driver",
        ],
    )
    roots = []

    def fake_writer(config: PhysicalRecorderConfig) -> object:
        roots.append(config.dataset_path)
        assert config.dataset_path.parent.is_dir()
        assert not config.dataset_path.exists()
        assert (config.image_width, config.image_height) == (48, 32)
        return SimpleNamespace(finalize=lambda: None)

    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", fake_writer)
    monkeypatch.setattr(
        recorder, "RosJointStateSource", lambda *args, **kwargs: SimpleNamespace(
            close=lambda: None,
            measure_rate=lambda: StateRateMeasurement(50, 100, 2, 0, 2, 0.02),
            read=lambda: PhysicalState((0.01,) * 6, 0.01, 2, 1000),
            require_no_command_publishers=lambda topics: None,
        )
    )
    monkeypatch.setattr(
        recorder, "OpenCVFrameSource", lambda *args, **kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(
        recorder, "PhysicalEpisodeRecorder", lambda **kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(
        recorder,
        "run_operator_loop",
        lambda recorder: PhysicalSessionResult(
            saved_episode_count=1, last_episode_index=0, smoke=True
        ),
    )
    assert recorder.physical_main() == 0
    assert len(roots) == 1
    assert not roots[0].parent.exists()


def real_dataset_type() -> Any:
    """Skip only when the optional supported upstream library is absent."""
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        pytest.skip("optional upstream LeRobot 0.6 is not installed")
    if installed.split(".")[:2] != ["0", "6"]:
        pytest.skip("optional integration tests require upstream LeRobot 0.6")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def test_real_writer_creates_nonexisting_root_and_resumes_empty_dataset(tmp_path: Path) -> None:
    real_dataset_type()
    config = writer_config(tmp_path / "new-dataset")
    writer = LeRobotDatasetWriter(config)
    try:
        assert writer.next_episode_index == 0
        assert json.loads((config.dataset_path / "physical_contract.json").read_text()) == (
            config.contract.as_dict(fps=config.fps)
        )
    finally:
        writer.finalize()
    resumed = LeRobotDatasetWriter(config)
    try:
        assert resumed.next_episode_index == 0
    finally:
        resumed.finalize()


def test_real_writer_resumes_persisted_episode_index(tmp_path: Path) -> None:
    real_dataset_type()
    config = writer_config(tmp_path / "existing-dataset")
    writer = LeRobotDatasetWriter(config)
    try:
        for index in range(2):
            writer.write_episode(synthetic_episode(index))
    finally:
        writer.finalize()
    resumed = LeRobotDatasetWriter(config)
    try:
        assert resumed.next_episode_index == 2
        assert resumed._dataset.meta.total_episodes == 2
    finally:
        resumed.finalize()


def synthetic_episode(
    index: int = 0, *, width: int = 224, height: int = 224,
    action_source: ActionSource = ActionSource.NEXT_STATE, lookahead_steps: int = 1,
) -> RecordedPhysicalEpisode:
    import numpy as np

    frames = []
    for number in range(3):
        stamp = number / 15
        frames.append(
            PhysicalFrame(
                state=PhysicalState(
                    (0.01,) * 6, 0.01, stamp, stamp + 1000,
                    ros_arrival_stamp_s=stamp + 1000,
                ),
                action=(0.01,) * 7,
                action_monotonic_timestamp_s=stamp,
                sample_monotonic_timestamp_s=stamp + 0.001,
                action_ros_header_stamp_s=stamp + 1000,
                action_ros_arrival_stamp_s=stamp + 1000,
                wrist=ImageFrame(
                    np.full((height, width, 3), 30 + number, dtype=np.uint8),
                    stamp, (width, height), "synthetic-wrist",
                ),
                front=ImageFrame(
                    np.full((height, width, 3), 90 + number, dtype=np.uint8),
                    stamp, (width, height), "synthetic-front",
                ),
            )
        )
    for frame_index, frame in enumerate(frames):
        offset = lookahead_steps if action_source is ActionSource.NEXT_STATE else 0
        target = frames[min(frame_index + offset, len(frames) - 1)].state
        frames[frame_index] = replace(
            frame, action_monotonic_timestamp_s=target.monotonic_timestamp_s,
            action_ros_header_stamp_s=target.ros_header_stamp_s,
            action_ros_arrival_stamp_s=target.ros_arrival_stamp_s,
        )
    return RecordedPhysicalEpisode(
        episode_index=index,
        frames=frames,
        started_monotonic_s=0,
        ended_monotonic_s=0.2,
        operator_label=OperatorLabel.SUCCESS,
        action_source=action_source,
        action_lookahead_steps=lookahead_steps,
        contract_version="synria_physical_v1",
        gripper_type="50mm",
    )


def test_writer_passes_task_in_frame_and_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_frames = []
    calls = {"saved": 0, "finalized": 0}

    class Dataset:
        meta = SimpleNamespace(
            total_episodes=0, total_frames=0, episodes=[], features=image_features(), fps=15
        )

        @staticmethod
        def create(**kwargs: Any) -> Dataset:
            kwargs["root"].mkdir()
            (kwargs["root"] / "meta").mkdir()
            (kwargs["root"] / "meta" / "info.json").write_text("{}", encoding="utf-8")
            return Dataset()

        @staticmethod
        def resume(**kwargs: Any) -> Dataset:
            return Dataset()

        def add_frame(self, frame: dict[str, object]) -> None:
            saved_frames.append(frame)

        def save_episode(self, **kwargs: Any) -> None:
            calls["saved"] += 1
            self.meta.total_episodes += 1

        def finalize(self) -> None:
            calls["finalized"] += 1

    _install_fake_library(monkeypatch, Dataset)
    config = writer_config(tmp_path / "dataset")
    writer = LeRobotDatasetWriter(config)
    monkeypatch.setattr(writer, "save_final_still", lambda index, frame: tmp_path / "fake.jpg")
    writer.write_episode(synthetic_episode())
    writer.finalize()
    writer.finalize()
    assert calls == {"saved": 1, "finalized": 2}
    assert len(saved_frames) == 3
    assert all(frame["task"] == config.task for frame in saved_frames)
    with pytest.raises(RuntimeError, match="finalized"):
        writer.write_episode(synthetic_episode())


def test_contract_write_failure_finalizes_created_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalized = []

    class Dataset:
        @staticmethod
        def create(**kwargs: Any) -> SimpleNamespace:
            kwargs["root"].mkdir()
            return SimpleNamespace(finalize=lambda: finalized.append(True))

    _install_fake_library(monkeypatch, Dataset)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("contract write failure")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="contract write failure"):
        LeRobotDatasetWriter(writer_config(tmp_path / "dataset"))
    assert finalized == [True]


def test_real_writer_saves_task_and_finalizes_for_reload(tmp_path: Path) -> None:
    library = real_dataset_type()
    config = writer_config(tmp_path / "dataset")
    writer = LeRobotDatasetWriter(config)
    try:
        writer.write_episode(synthetic_episode())
    finally:
        writer.finalize()
    writer.finalize()
    reloaded = library(config.repo_id, root=config.dataset_path, video_backend="pyav")
    assert reloaded.num_episodes == 1
    assert reloaded.num_frames == 3
    assert reloaded[0]["task"] == config.task
    assert tuple(reloaded[0]["observation.images.wrist"].shape) == (3, 224, 224)


def test_resume_rejects_image_shape_changes_before_writing(tmp_path: Path) -> None:
    real_dataset_type()
    config = writer_config(tmp_path / "dataset")
    writer = LeRobotDatasetWriter(config)
    writer.finalize()
    with pytest.raises(ValueError, match="image features differ"):
        LeRobotDatasetWriter(replace(config, image_width=320))


def test_real_capture_stores_resized_rgb_and_correct_final_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = real_dataset_type()
    import cv2
    import numpy as np

    from synria_lerobot.recorder import OpenCVFrameSource

    raw = np.full((240, 320, 3), (5, 30, 220), dtype=np.uint8)
    capture = SimpleNamespace(
        isOpened=lambda: True, read=lambda: (True, raw.copy()), release=lambda: None
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    source = OpenCVFrameSource("/dev/v4l/by-id/synthetic-camera", width=48, height=32)
    try:
        captured = source.read()
    finally:
        source.close()
    assert captured.data.shape == (32, 48, 3)
    assert captured.data[0, 0].tolist() == [220, 30, 5]
    assert captured.native_resolution == (320, 240)
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    episode = synthetic_episode(width=48, height=32)
    episode.frames = [
        replace(
            frame,
            wrist=replace(captured, monotonic_timestamp_s=frame.wrist.monotonic_timestamp_s),
            front=replace(captured, monotonic_timestamp_s=frame.front.monotonic_timestamp_s),
        )
        for frame in episode.frames
    ]
    writer = LeRobotDatasetWriter(config)
    try:
        writer.write_episode(episode)
    finally:
        writer.finalize()
    stored = library(config.repo_id, root=config.dataset_path, video_backend="pyav")
    rgb = stored[0]["observation.images.wrist"]
    assert tuple(rgb.shape) == (3, 32, 48)
    assert rgb[:, 0, 0].tolist() == pytest.approx([220 / 255, 30 / 255, 5 / 255], abs=0.03)
    bgr = cv2.imread(str(episode.final_still_path))
    assert bgr[0, 0].tolist() == pytest.approx([5, 30, 220], abs=3)
    capture_record = json.loads(
        (config.dataset_path / "physical_capture_provenance.jsonl").read_text()
    )
    assert capture_record["native_resolution"]["wrist"] == {"width": 320, "height": 240}
    assert capture_record["stored_resolution"] == {"width": 48, "height": 32}


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("fault_stage", ["data", "metadata", "sidecars"])
def test_real_late_save_failure_rolls_back_then_retries_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_stage: str
) -> None:
    library = real_dataset_type()
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_writer import DatasetWriter

    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    writer = LeRobotDatasetWriter(config)
    for index in range(2):
        writer.write_episode(synthetic_episode(index, width=48, height=32))
    baseline = _file_hashes(config.dataset_path)
    episode = synthetic_episode(2, width=48, height=32)
    target, method = {
        "data": (DatasetWriter, "_save_episode_data"),
        "metadata": (LeRobotDatasetMetadata, "save_episode"),
        "sidecars": (LeRobotDatasetWriter, "_write_episode_data"),
    }[fault_stage]
    original = getattr(target, method)

    def fail_after_write(self: Any, *args: Any, **kwargs: Any) -> Any:
        original(self, *args, **kwargs)
        raise OSError(f"injected failure after {fault_stage}")

    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail_after_write)
        with pytest.raises(OSError, match="injected failure"):
            writer.write_episode(episode)
    assert writer.next_episode_index == 2
    assert not writer.recovery_blocked
    assert len(episode.frames) == 3
    assert episode.final_still_path is None
    assert _file_hashes(config.dataset_path) == baseline
    writer.write_episode(episode)
    writer.finalize()
    restored = library(config.repo_id, root=config.dataset_path, video_backend="pyav")
    assert restored.num_episodes == 3
    assert restored.num_frames == 9
    assert sorted({int(row["episode_index"]) for row in restored.hf_dataset}) == [0, 1, 2]
    for name in ("physical_episode_metadata.jsonl", "physical_quality_records.jsonl",
                 "physical_capture_provenance.jsonl"):
        rows = [json.loads(line) for line in (config.dataset_path / name).read_text().splitlines()]
        assert [row["episode_index"] for row in rows] == [0, 1, 2]
    assert len(list((config.dataset_path / "final_stills").glob("*.jpg"))) == 3


@pytest.mark.parametrize("source", list(ActionSource))
def test_real_two_episodes_close_resume_third_and_reload(
    tmp_path: Path, source: ActionSource
) -> None:
    library = real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    config = replace(config, contract=replace(
        config.contract, action_source=source, action_lookahead_steps=2
    ))
    writer = LeRobotDatasetWriter(config)
    for index in range(2):
        writer.write_episode(synthetic_episode(
            index, width=48, height=32, action_source=source, lookahead_steps=2
        ))
    writer.finalize()
    resumed = LeRobotDatasetWriter(config)
    assert resumed.next_episode_index == 2
    resumed.write_episode(synthetic_episode(
        2, width=48, height=32, action_source=source, lookahead_steps=2
    ))
    resumed.finalize()
    reloaded = library(config.repo_id, root=config.dataset_path, video_backend="pyav")
    assert (reloaded.num_episodes, reloaded.num_frames) == (3, 9)
    assert tuple(reloaded[8]["observation.images.front"].shape) == (3, 32, 48)
    assert tuple(reloaded[0]["observation.images.wrist"].shape) == (3, 32, 48)
    timing = action_timing_metadata(source, 2, 15)
    contract = json.loads((config.dataset_path / "physical_contract.json").read_text())
    assert all(contract[name] == value for name, value in timing.items())
    for name in ("physical_episode_metadata.jsonl", "physical_quality_records.jsonl",
                 "physical_capture_provenance.jsonl"):
        records = [
            json.loads(line) for line in (config.dataset_path / name).read_text().splitlines()
        ]
        assert len(records) == 3
        for record in records:
            assert all(record[key] == value for key, value in timing.items())
            assert record["action_source"] == source.value
    raw = reloaded.hf_dataset.with_format(None)[0]
    assert raw["action_monotonic_s"] == (2 / 15 if source is ActionSource.NEXT_STATE else 0)


@pytest.mark.parametrize("changed", ["lookahead", "rate", "stored_rate"])
def test_real_resume_refuses_changed_horizon_or_rate(tmp_path: Path, changed: str) -> None:
    real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    LeRobotDatasetWriter(config).finalize()
    if changed == "lookahead":
        config = replace(config, contract=replace(config.contract, action_lookahead_steps=2))
    elif changed == "rate":
        config = replace(config, fps=30)
    else:
        path = config.dataset_path / "meta" / "info.json"
        metadata = json.loads(path.read_text())
        metadata["fps"] = 30
        path.write_text(json.dumps(metadata), encoding="utf-8")
    before = _file_hashes(config.dataset_path)
    with pytest.raises(ValueError, match="contract"):
        LeRobotDatasetWriter(config)
    assert _file_hashes(config.dataset_path) == before


@pytest.mark.parametrize("fault", ["orphan", "missing", "total_episodes", "total_frames"])
def test_real_resume_rejects_incomplete_or_orphan_metadata(tmp_path: Path, fault: str) -> None:
    real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    writer = LeRobotDatasetWriter(config)
    writer.write_episode(synthetic_episode(width=48, height=32))
    writer.finalize()
    if fault == "orphan":
        (config.dataset_path / "data" / "orphan.parquet").write_bytes(b"unrelated")
    elif fault == "missing":
        next((config.dataset_path / "data").rglob("*.parquet")).unlink()
    else:
        path = config.dataset_path / "meta" / "info.json"
        info = json.loads(path.read_text())
        info[fault] += 1
        path.write_text(json.dumps(info), encoding="utf-8")
    baseline = _file_hashes(config.dataset_path)
    with pytest.raises(RuntimeError, match="inconsistent|missing|orphan"):
        LeRobotDatasetWriter(config)
    assert _file_hashes(config.dataset_path) == baseline


@pytest.mark.parametrize("blocked_at", ["finalize", "rollback"])
def test_real_failed_recovery_retains_frames_journal_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    writer = LeRobotDatasetWriter(config)
    writer.write_episode(synthetic_episode(width=48, height=32))
    baseline = _file_hashes(config.dataset_path)
    episode = synthetic_episode(1, width=48, height=32)
    original_write = writer._write_episode_data

    def fail_after_write(*args: Any) -> None:
        original_write(*args)
        raise OSError("injected save fault")

    def fail_recovery() -> None:
        raise OSError("injected recovery fault")

    with monkeypatch.context() as patch:
        patch.setattr(writer, "_write_episode_data", fail_after_write)
        if blocked_at == "finalize":
            patch.setattr(writer, "_close_dataset", fail_recovery)
        else:
            patch.setattr(writer._transaction, "rollback", fail_recovery)
        with pytest.raises(RuntimeError, match="recovery blocked"):
            writer.write_episode(episode)
    assert writer.recovery_blocked
    assert len(episode.frames) == 3
    assert writer._transaction.journal.is_file()
    assert (tmp_path / ".dataset.recording.lock").is_file()
    with pytest.raises(RuntimeError, match="recovery is blocked"):
        writer.write_episode(episode)
    with pytest.raises(RuntimeError, match="retained for recovery"):
        writer.finalize()
    # Explicit test-only recovery after verifying that automatic continuation is blocked.
    writer._close_dataset()
    writer._transaction.rollback()
    writer.recovery_blocked = False
    writer.finalize()
    assert _file_hashes(config.dataset_path) == baseline


def test_real_camera_setting_changes_are_rejected_before_dataset_mutation(tmp_path: Path) -> None:
    real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    writer = LeRobotDatasetWriter(config)
    writer.write_episode(synthetic_episode(width=48, height=32))
    baseline = _file_hashes(config.dataset_path)
    changed = synthetic_episode(1, width=48, height=32)
    changed.frames = [
        replace(frame, wrist=replace(frame.wrist, source_id="changed-camera"))
        for frame in changed.frames
    ]
    with pytest.raises(ValueError, match="start a separate dataset"):
        writer.write_episode(changed)
    writer.finalize()
    assert _file_hashes(config.dataset_path) == baseline


@pytest.mark.parametrize("stale", [False, True])
def test_real_timestamp_evidence_preserves_precision_and_fails_stale_sources(
    tmp_path: Path, stale: bool
) -> None:
    dataset_type = real_dataset_type()
    import pyarrow as pa
    import pyarrow.parquet as pq

    from synria_lerobot.quality_gates import (
        EpisodeQualityRecord,
        GateConfig,
        evaluate_episode,
        load_limits,
    )

    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    episode = synthetic_episode(width=48, height=32)
    for index, frame in enumerate(episode.frames):
        arrival = 123_456 + index / config.fps
        header = 1_800_000_000.125 + index / config.fps
        episode.frames[index] = replace(
            frame,
            state=replace(
                frame.state, monotonic_timestamp_s=arrival, ros_header_stamp_s=header,
                ros_arrival_stamp_s=header + (1 if stale and index == 1 else 0.01),
            ),
            wrist=replace(frame.wrist, monotonic_timestamp_s=arrival),
            front=replace(frame.front, monotonic_timestamp_s=arrival),
            sample_monotonic_timestamp_s=arrival + 0.001,
        )
    for index, frame in enumerate(episode.frames):
        target = episode.frames[min(index + 1, len(episode.frames) - 1)].state
        episode.frames[index] = replace(
            frame, action_monotonic_timestamp_s=target.monotonic_timestamp_s,
            action_ros_header_stamp_s=target.ros_header_stamp_s,
            action_ros_arrival_stamp_s=target.ros_arrival_stamp_s,
        )
    writer = LeRobotDatasetWriter(config)
    try:
        writer.write_episode(episode)
    finally:
        writer.finalize()
    reloaded = dataset_type(config.repo_id, root=config.dataset_path)
    raw = reloaded.hf_dataset.with_format(None)[0]
    table = pq.read_table(next((config.dataset_path / "data").rglob("*.parquet")))
    for key, expected in episode.frames[0].timestamps().items():
        assert table.schema.field(key).type == pa.float64()
        assert raw[key] == expected
    payload = json.loads((config.dataset_path / "physical_quality_records.jsonl").read_text())
    record = EpisodeQualityRecord.from_dict(payload)
    assert record.achieved_sample_rate_hz == pytest.approx(15)
    report = evaluate_episode(
        record, load_limits(Path("config/synria_limits.yaml")),
        GateConfig(min_episode_s=0.1, max_episode_s=1),
    )
    assert report.failed_gates == (["source_staleness"] if stale else [])
    capture = json.loads(
        (config.dataset_path / "physical_capture_provenance.jsonl").read_text()
    )
    assert capture["achieved_sample_rate_hz"] == pytest.approx(15)


@pytest.mark.parametrize("required", [False, True])
def test_real_dataset_persists_exact_contract_state_vector(
    tmp_path: Path, required: bool
) -> None:
    dataset_type = real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    config = replace(config, contract=replace(config.contract, state_has_velocity=required))
    frames = synthetic_episode(width=48, height=32).frames
    states = [replace(frame.state, joint_velocities_rad_s=(0.2,) * 6) for frame in frames]
    # Preflight is an explicit additional latest-state read only in required-velocity mode.
    state_reads = iter(([states[0]] if required else []) + states)
    wrist_reads = iter(frame.wrist for frame in frames)
    front_reads = iter(frame.front for frame in frames)
    clock = SimpleNamespace(now=0.0)
    writer = LeRobotDatasetWriter(config)
    recorder = PhysicalEpisodeRecorder(
        config=config,
        state_source=SimpleNamespace(read=lambda: next(state_reads), close=lambda: None),
        action_source=NextStateActionSource(),
        wrist_source=SimpleNamespace(read=lambda: next(wrist_reads), close=lambda: None),
        front_source=SimpleNamespace(read=lambda: next(front_reads), close=lambda: None),
        writer=writer, clock=lambda: clock.now,
    )
    try:
        recorder.start()
        for index in range(3):
            clock.now = index / config.fps + 0.001
            recorder.capture_once()
        clock.now = 0.2
        recorder.stop()
        episode = recorder.mark_success()
    finally:
        recorder.close()
    reloaded = dataset_type(config.repo_id, root=config.dataset_path)
    raw = reloaded.hf_dataset.with_format(None)[0]
    assert len(raw["observation.state"]) == (13 if required else 7)
    assert len(raw["action"]) == 7
    assert len(episode.frames[0].state.observation_vector()) == (13 if required else 7)
    if required:
        assert raw["observation.state"][-6:] == pytest.approx([0.2] * 6)
    quality = json.loads((config.dataset_path / "physical_quality_records.jsonl").read_text())
    assert len(quality["frames"][0]["state"]) == (13 if required else 7)


def test_real_writer_removes_reported_velocities_for_position_only_contract(tmp_path: Path) -> None:
    dataset_type = real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    episode = synthetic_episode(width=48, height=32)
    episode.frames = [
        replace(frame, state=replace(frame.state, joint_velocities_rad_s=(0.2,) * 6))
        for frame in episode.frames
    ]
    writer = LeRobotDatasetWriter(config)
    try:
        writer.write_episode(episode)
    finally:
        writer.finalize()
    reloaded = dataset_type(config.repo_id, root=config.dataset_path)
    assert len(reloaded.hf_dataset.with_format(None)[0]["observation.state"]) == 7


def test_real_velocity_contract_refuses_missing_state_before_recording(tmp_path: Path) -> None:
    dataset_type = real_dataset_type()
    config = replace(writer_config(tmp_path / "dataset"), image_width=48, image_height=32)
    config = replace(config, contract=replace(config.contract, state_has_velocity=True))
    writer = LeRobotDatasetWriter(config)
    frame = synthetic_episode(width=48, height=32).frames[0]

    def unexpected_capture() -> None:
        raise AssertionError("no image should be sampled after failed state preflight")

    recorder = PhysicalEpisodeRecorder(
        config=config,
        state_source=SimpleNamespace(read=lambda: frame.state, close=lambda: None),
        action_source=NextStateActionSource(),
        wrist_source=SimpleNamespace(read=unexpected_capture, close=lambda: None),
        front_source=SimpleNamespace(read=unexpected_capture, close=lambda: None),
        writer=writer,
    )
    try:
        with pytest.raises(ValueError, match="requires six reported joint velocities"):
            recorder.start()
        assert recorder.state is RecorderState.IDLE
        assert recorder._pending == []
    finally:
        recorder.close()
    assert writer.next_episode_index == 0
    assert not (config.dataset_path / "physical_quality_records.jsonl").exists()
    resumed = dataset_type.resume(config.repo_id, root=config.dataset_path)
    try:
        assert resumed.meta.total_episodes == 0
    finally:
        resumed.finalize()


@pytest.mark.parametrize("state_source", ["standalone_driver", "ros2_control"])
def test_real_resumed_runs_preserve_distinct_incoming_rate_evidence(
    tmp_path: Path, state_source: str
) -> None:
    library = real_dataset_type()
    from synria_lerobot.quality_gates import (
        GateConfig,
        load_episode_records,
        load_limits,
        write_aggregate_summary,
        write_session_artifacts,
    )

    source_provenance = StateSourceProvenance(state_source, "/joint_states")
    base_config = replace(
        writer_config(tmp_path / "dataset"), image_width=48, image_height=32,
        state_source_provenance=source_provenance,
    )
    measurements = [
        StateRateMeasurement(50, 100, 2, 0, 2, 0.02),
        StateRateMeasurement(40, 80, 2, 10, 12, 0.025),
    ]
    for index, measurement in enumerate(measurements):
        config = replace(base_config, state_rate_measurement=measurement)
        episode = synthetic_episode(index, width=48, height=32)
        episode.state_rate_measurement = measurement
        episode.state_source_provenance = source_provenance
        offset = measurement.ended_monotonic_s + 1
        episode.started_monotonic_s += offset
        episode.ended_monotonic_s += offset
        episode.frames = [
            replace(
                frame,
                state=replace(frame.state, monotonic_timestamp_s=(
                    frame.state.monotonic_timestamp_s + offset
                )),
                wrist=replace(frame.wrist, monotonic_timestamp_s=(
                    frame.wrist.monotonic_timestamp_s + offset
                )),
                front=replace(frame.front, monotonic_timestamp_s=(
                    frame.front.monotonic_timestamp_s + offset
                )),
                sample_monotonic_timestamp_s=frame.sample_monotonic_timestamp_s + offset,
                action_monotonic_timestamp_s=frame.action_monotonic_timestamp_s + offset,
            )
            for frame in episode.frames
        ]
        writer = LeRobotDatasetWriter(config)
        try:
            writer.write_episode(episode)
        finally:
            writer.finalize()
    assert library(base_config.repo_id, root=base_config.dataset_path).num_episodes == 2
    for filename in ("physical_episode_metadata.jsonl", "physical_capture_provenance.jsonl"):
        records = [
            json.loads(line)
            for line in (base_config.dataset_path / filename).read_text().splitlines()
        ]
        assert [row["state_rate_measurement"] for row in records] == [
            asdict(measurement) for measurement in measurements
        ]
        assert all(row["state_source_provenance"] == source_provenance.as_dict() for row in records)
    episodes = load_episode_records(base_config.dataset_path / "physical_quality_records.jsonl")
    provenance = {
        "follower_serial": "fake-follower", "leader_serial": "fake-leader", "host": "fake",
        "git_sha": "test", "utc_date": "2026-10-08", "operator": "test",
        "power_state_start": "synthetic", "power_state_end": "synthetic", "scene": "synthetic",
        "camera_ids": {"wrist": "synthetic-wrist", "front": "synthetic-front"},
        "resolution": {"width": 48, "height": 32}, "rate_hz": 15,
        **base_config.contract.as_dict(fps=15),
        "state_source_provenance": source_provenance.as_dict(),
    }
    data_root = tmp_path / "reports"
    summary = write_session_artifacts(
        session_dir=data_root / "session", dataset_path=base_config.dataset_path,
        provenance=provenance, episodes=episodes,
        limits=load_limits(Path("config/synria_limits.yaml")),
        gate_config=GateConfig(min_episode_s=0.1, max_episode_s=1),
    )
    expected = [
        {"episode_index": index, "measurement": asdict(measurement)}
        for index, measurement in enumerate(measurements)
    ]
    assert summary["state_rate_measurements"] == expected
    assert summary["state_source_provenance"] == source_provenance.as_dict()
    saved_provenance = json.loads((data_root / "session" / "provenance.json").read_text())
    assert saved_provenance["state_rate_measurements"] == expected
    assert saved_provenance["state_source_provenance"] == source_provenance.as_dict()
    aggregate = write_aggregate_summary(
        data_root=data_root, output_path=data_root / "aggregate.json",
        timeline_path=tmp_path / "timeline.jsonl",
    )
    assert aggregate["state_rate_measurements"][0]["episodes"] == expected
    assert aggregate["state_sources"][0]["provenance"] == source_provenance.as_dict()

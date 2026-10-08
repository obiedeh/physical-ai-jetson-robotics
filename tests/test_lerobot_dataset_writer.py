"""Dataset-boundary regressions; all images and state are synthetic."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
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
)
from synria_lerobot.recorder import (
    LeRobotDatasetWriter,
    OperatorLabel,
    PhysicalRecorderConfig,
    RecordedPhysicalEpisode,
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
    return {
        f"observation.images.{name}": {"dtype": "video", "shape": (3, 224, 224)}
        for name in ("wrist", "front")
    }


def _install_fake_library(monkeypatch: pytest.MonkeyPatch, library: type) -> None:
    package = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
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
            return SimpleNamespace(meta=SimpleNamespace(total_episodes=0))

    _install_fake_library(monkeypatch, Dataset)
    writer = LeRobotDatasetWriter(config)
    assert writer.next_episode_index == 0
    assert json.loads((config.dataset_path / "physical_contract.json").read_text()) == (
        config.contract.as_dict()
    )


def test_resume_uses_dataset_metadata_and_refuses_contract_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = writer_config(tmp_path / "dataset")
    (config.dataset_path / "meta").mkdir(parents=True)
    (config.dataset_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    contract_path = config.dataset_path / "physical_contract.json"
    contract_path.write_text(json.dumps(config.contract.as_dict()), encoding="utf-8")
    calls = []

    class Dataset:
        @staticmethod
        def resume(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                meta=SimpleNamespace(total_episodes=7, features=image_features()),
                finalize=lambda: None,
            )

    _install_fake_library(monkeypatch, Dataset)
    writer = LeRobotDatasetWriter(config)
    assert writer.next_episode_index == 7
    assert calls == [{"repo_id": config.repo_id, "root": config.dataset_path}]
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
        recorder, "RosJointStateSource", lambda *args, **kwargs: SimpleNamespace(close=lambda: None)
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
        lambda recorder: SimpleNamespace(
            episode_index=0, operator_label=OperatorLabel.FAILURE, duration_s=20.0, smoke=True
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
            config.contract.as_dict()
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
    index: int = 0, *, width: int = 224, height: int = 224
) -> RecordedPhysicalEpisode:
    import numpy as np

    frames = []
    for number in range(3):
        stamp = number / 15
        frames.append(
            PhysicalFrame(
                state=PhysicalState((0.01,) * 6, 0.01, stamp, stamp + 1000),
                action=(0.01,) * 7,
                action_monotonic_timestamp_s=stamp,
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
    return RecordedPhysicalEpisode(
        episode_index=index,
        frames=frames,
        started_monotonic_s=0,
        ended_monotonic_s=0.2,
        operator_label=OperatorLabel.SUCCESS,
        action_source=ActionSource.NEXT_STATE,
        contract_version="synria_physical_v1",
        gripper_type="50mm",
    )


def test_writer_passes_task_in_frame_and_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_frames = []
    calls = {"saved": 0, "finalized": 0}

    class Dataset:
        meta = SimpleNamespace(total_episodes=0)

        @staticmethod
        def create(**kwargs: Any) -> Dataset:
            kwargs["root"].mkdir()
            return Dataset()

        def add_frame(self, frame: dict[str, object]) -> None:
            saved_frames.append(frame)

        def save_episode(self) -> None:
            calls["saved"] += 1

        def finalize(self) -> None:
            calls["finalized"] += 1

    _install_fake_library(monkeypatch, Dataset)
    config = writer_config(tmp_path / "dataset")
    writer = LeRobotDatasetWriter(config)
    writer.write_episode(synthetic_episode())
    writer.finalize()
    writer.finalize()
    assert calls == {"saved": 1, "finalized": 1}
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
        episode.final_still_path = writer.save_final_still(0, episode.frames[-1].front)
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

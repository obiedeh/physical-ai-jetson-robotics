"""Dataset-boundary regressions; all images and state are synthetic."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract
from synria_lerobot.recorder import LeRobotDatasetWriter, OperatorLabel, PhysicalRecorderConfig


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
            return SimpleNamespace(meta=SimpleNamespace(total_episodes=7))

    _install_fake_library(monkeypatch, Dataset)
    writer = LeRobotDatasetWriter(config)
    assert writer.next_episode_index == 7
    assert calls == [{"repo_id": config.repo_id, "root": config.dataset_path}]
    contract_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="contract differs"):
        LeRobotDatasetWriter(config)
    assert len(calls) == 1


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
            "--smoke",
        ],
    )
    roots = []

    def fake_writer(config: PhysicalRecorderConfig) -> object:
        roots.append(config.dataset_path)
        assert config.dataset_path.parent.is_dir()
        assert not config.dataset_path.exists()
        return object()

    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", fake_writer)
    monkeypatch.setattr(recorder, "RosJointStateSource", lambda *args, **kwargs: object())
    monkeypatch.setattr(recorder, "OpenCVFrameSource", lambda *args, **kwargs: object())
    monkeypatch.setattr(recorder, "PhysicalEpisodeRecorder", lambda **kwargs: object())
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
        writer._dataset.finalize()
    resumed = LeRobotDatasetWriter(config)
    try:
        assert resumed.next_episode_index == 0
    finally:
        resumed._dataset.finalize()


def test_real_writer_resumes_persisted_episode_index(tmp_path: Path) -> None:
    import numpy as np

    library = real_dataset_type()
    config = writer_config(tmp_path / "existing-dataset")
    # Seed directly through the upstream API to isolate creation/resumption from
    # the frame-writing adapter, which has separate compatibility tests.
    dataset = library.create(
        repo_id=config.repo_id,
        root=config.dataset_path,
        fps=int(config.fps),
        robot_type="synria_alicia_d",
        features={"observation.state": {"dtype": "float32", "shape": (7,)}},
        use_videos=False,
    )
    try:
        for _ in range(2):
            dataset.add_frame({"observation.state": np.zeros(7, dtype=np.float32), "task": "test"})
            dataset.save_episode()
    finally:
        dataset.finalize()
    (config.dataset_path / "physical_contract.json").write_text(
        json.dumps(config.contract.as_dict()), encoding="utf-8"
    )
    resumed = LeRobotDatasetWriter(config)
    try:
        assert resumed.next_episode_index == 2
        assert resumed._dataset.meta.total_episodes == 2
    finally:
        resumed._dataset.finalize()

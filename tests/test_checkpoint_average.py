"""Averaging a run's last checkpoints: the weights, the receipt and serving the result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synria_lerobot.act_training import train
from synria_lerobot.checkpoint_average import average_checkpoints, average_state_dicts


def test_average_state_dicts_means_floats_and_keeps_the_newest_integers() -> None:
    """Float tensors are averaged in double precision; integer buffers come from the last."""
    import torch

    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([1, 1])}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([7, 7])}
    averaged = average_state_dicts([a, b])
    assert torch.equal(averaged["w"], torch.tensor([2.0, 4.0]))
    assert torch.equal(averaged["n"], torch.tensor([7, 7]))
    with pytest.raises(ValueError, match="different tensors"):
        average_state_dicts([a, {"w": torch.zeros(2)}])
    with pytest.raises(ValueError, match="changes shape"):
        average_state_dicts([a, {"w": torch.zeros(3), "n": torch.zeros(2, dtype=torch.int64)}])
    with pytest.raises(ValueError, match="nothing"):
        average_state_dicts([])


def test_real_run_average_is_evaluated_receipted_and_servable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two one-step checkpoints averaged on the CPU, then loaded by the simulation-lane server."""
    from test_synria_act_training_real import training_fixture
    from test_task_registry import synthetic_registry

    from synria_lerobot.policy_server import CheckpointIdentity, LocalACTModel

    config, dataset = training_fixture(tmp_path, monkeypatch, simulation=True)
    manifest = train(dataset, dataset / "physical_contract.json", tmp_path / "raw", 5, 2, config)
    assert [entry["step"] for entry in manifest["checkpoints"]] == [1, 2]
    repository = Path(config["repository"])
    evidence = repository / config["evidence_directory"]
    with pytest.raises(ValueError, match="at least two"):
        average_checkpoints(evidence, 1, repository=repository, dataset_root=dataset, device="cpu")
    with pytest.raises(ValueError, match="fewer than"):
        average_checkpoints(evidence, 3, repository=repository, dataset_root=dataset, device="cpu")
    summary = average_checkpoints(evidence, 2, repository=repository, dataset_root=dataset,
                                  device="cpu")
    assert summary["averaged_from_steps"] == [1, 2] and summary["evaluation_status"] == "evaluated"
    assert summary["joint_mae_rad_mean"] is not None and summary["gripper_mae_m"] is not None
    target = Path(summary["checkpoint_path"])
    assert target.name == "averaged-last2-step-000000002"
    assert not (target / "training_state").exists()
    receipt = json.loads(Path(summary["checkpoint_record"]).read_text())
    assert receipt["averaged_from_steps"] == [1, 2] and receipt["evaluation_status"] == "evaluated"
    assert receipt["checkpoint_content_sha256"] == summary["checkpoint_content_sha256"]
    [record] = (evidence / "evaluations").glob("*_avg2-step-*.json")
    assert json.loads(record.read_text())["status"] == "evaluated"

    from safetensors.torch import load_file

    sources = [load_file(Path(e["path"]) / "pretrained_model" / "model.safetensors")
               for e in manifest["checkpoints"]]
    averaged = load_file(target / "pretrained_model" / "model.safetensors")
    key = next(k for k, v in averaged.items() if v.is_floating_point() and v.numel() > 1)
    expected = (sources[0][key].double() + sources[1][key].double()) / 2
    assert (averaged[key].double() - expected).abs().max().item() < 1e-6

    registry = synthetic_registry(tmp_path / "registry.json", 0.1, 1)
    identity = CheckpointIdentity.load(target, Path(summary["checkpoint_record"]), registry,
                                       "simulation")
    assert identity.manifest["averaged_from_steps"] == [1, 2]
    model = LocalACTModel(identity, "cpu")
    assert set(model.image_shapes) == {"observation.images.wrist", "observation.images.front"}
    with pytest.raises(FileExistsError):
        average_checkpoints(evidence, 2, repository=repository, dataset_root=dataset, device="cpu")

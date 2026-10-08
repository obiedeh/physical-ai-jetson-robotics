"""Optional real upstream ACT updates, using only a fake-source temporary D1 dataset."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from test_lerobot_dataset_writer import synthetic_episode, writer_config
from test_synria_act_training import settings
from test_synria_checkpoint_eval import commit, fixed_probe_trial

from synria_lerobot.act_training import train
from synria_lerobot.checkpoint_eval import register_probes, strict_content_hash
from synria_lerobot.recorder import LeRobotDatasetWriter


def training_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        pytest.skip("optional upstream LeRobot is absent")
    assert installed.split(".")[:2] == ["0", "6"], "installed upstream must be supported 0.6"
    if find_spec("torch") is None:
        pytest.skip("optional tensor runtime is absent")
    import torch

    torch.set_num_threads(1)
    dataset = tmp_path / "dataset"
    writer = LeRobotDatasetWriter(replace(writer_config(dataset), image_width=32, image_height=32))
    try:
        for index, value in enumerate((0.01, 0.03, 0.9)):
            episode = synthetic_episode(index, width=32, height=32)
            episode.frames[:] = [
                replace(
                    frame,
                    state=replace(
                        frame.state, joint_positions_rad=(value,) * 6, gripper_m=value / 100
                    ),
                    action=(*((value,) * 6), value / 100),
                )
                for frame in episode.frames
            ]
            writer.write_episode(episode)
    finally:
        writer.finalize()
    config, contract = settings(tmp_path)
    assert json.loads((dataset / "physical_contract.json").read_text()) == contract
    repository = Path(config["repository"])
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "POLICY_LEDGER.md").write_text("# Synthetic ledger\n")
    probe = repository / "probes.json"
    values = json.loads(Path("config/synria_checkpoint_probes.json").read_text())
    values.update(
        probe_set_id="synthetic-training",
        registered_by="synthetic fixture",
        registered_on="2026-10-08",
        dataset_content_sha256=strict_content_hash(dataset),
        task_id=contract["task_id"],
        physical_contract=contract,
        held_out_episodes=[2],
        held_out_frame_counts={"2": 3},
        physical_trials=[fixed_probe_trial("a")],
        capture=dict(fps=15, max_duration_s=10, shutdown_timeout_s=1),
    )
    probe.write_text(json.dumps(values))
    config["probe_sha256"] = register_probes(probe)
    commit(repository, probe)
    return config, dataset


def test_real_two_cpu_updates_save_each_checkpoint_evaluate_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = training_fixture(tmp_path, monkeypatch)
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    before = strict_content_hash(dataset)
    manifest = train(
        dataset, dataset / "physical_contract.json", tmp_path / "raw-training", 7, 2, config
    )
    assert manifest["status"] == "completed" and manifest["completed_steps"] == 2
    assert manifest["upstream_version"] == version("lerobot")
    assert manifest["task_text_conditioning"] is False
    assert manifest["declared_input_conditioning"] == [
        "observation.state",
        "observation.images.wrist",
        "observation.images.front",
    ]
    assert manifest["visual_goal_cue_declared"] is False
    assert manifest["varying_target_motion_eligible"] is False
    assert manifest["fixed_scene_motion_eligible"] is True
    assert manifest["eligible_task_ids"] == ["die_into_cup"]
    assert [entry["step"] for entry in manifest["checkpoints"]] == [1, 2]
    assert all(entry["evaluation_status"] == "evaluated" for entry in manifest["checkpoints"])
    assert strict_content_hash(dataset) == before
    stats = manifest["training_stats"]
    np.testing.assert_allclose(stats["observation.state"]["mean"][:6], [0.02] * 6, atol=1e-6)
    assert manifest["training_episode_indices"] == (0, 1)
    assert manifest["held_out_episodes"] == [2]
    repository = Path(config["repository"])
    evidence = repository / config["evidence_directory"]
    assert len(list((evidence / "evaluations").glob("*.json"))) == 2
    events = [
        json.loads(line) for line in (repository / config["timeline"]).read_text().splitlines()
    ]
    assert [event["step"] for event in events] == [1, 2]
    ledger = (repository / config["policy_ledger"]).read_text()
    assert sum(line.startswith("|") for line in ledger.splitlines()) == 1
    assert "response_timeout_s=0.3" in ledger
    for entry in manifest["checkpoints"]:
        checkpoint = Path(entry["path"])
        physical_manifest = json.loads((checkpoint / "physical_policy_manifest.json").read_text())
        assert physical_manifest["upstream_version"] == version("lerobot")
        assert physical_manifest["task_text_conditioning"] is False
        assert physical_manifest["varying_target_motion_eligible"] is False
        assert physical_manifest["fixed_scene_motion_eligible"] is True
        assert physical_manifest["eligible_task_ids"] == ["die_into_cup"]
        assert strict_content_hash(checkpoint) == entry["checkpoint_content_sha256"]
        local = checkpoint / "pretrained_model"
        saved = ACTConfig.from_pretrained(local, local_files_only=True)
        saved.device = "cpu"
        policy = ACTPolicy.from_pretrained(local, config=saved, local_files_only=True, strict=True)
        preprocessor, postprocessor = make_pre_post_processors(
            saved,
            pretrained_path=str(local),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        policy.eval()
        with torch.inference_mode():
            prediction = postprocessor(
                policy.predict_action_chunk(
                    preprocessor(
                        {
                            "observation.state": torch.zeros((1, 7)),
                            "observation.images.wrist": torch.zeros((1, 3, 32, 32)),
                            "observation.images.front": torch.zeros((1, 3, 32, 32)),
                        }
                    )
                )
            )
            zero_physical = postprocessor(torch.zeros((1, 6, 7)))
        assert tuple(prediction.shape) == (1, 6, 7) and torch.isfinite(prediction).all()
        np.testing.assert_allclose(zero_physical[0, 0].numpy(), stats["action"]["mean"], atol=1e-6)
        assert (local / "policy_preprocessor.json").is_file()
        assert (local / "policy_postprocessor.json").is_file()

    # The same finalized upstream checkpoint serves the physical HTTP protocol.
    import threading

    from test_task_registry import synthetic_registry

    from synria_lerobot.policy_client import HttpPolicyTransport
    from synria_lerobot.policy_server import (
        CheckpointIdentity,
        FixedScenePolicyService,
        LocalACTModel,
        make_http_server,
    )

    registry = synthetic_registry(tmp_path / "serving-registry.json", 0.1, 1)
    identity = CheckpointIdentity.load(
        checkpoint, evidence / "checkpoint-step-000000002.json", registry,
    )
    model = LocalACTModel(identity, "cpu")
    service = FixedScenePolicyService(identity, model)
    server = make_http_server(service, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = {
            **identity.contract.as_dict(), "embodiment": "synria_alicia_d",
            "observation.state": [0.0] * 7,
            **{key: np.zeros(shape, np.uint8) for key, shape in model.image_shapes.items()},
            "timestamps": dict(state=1.0, state_ros=10.0, wrist=1.0, front=1.0),
            "task": identity.contract.task_text, "request_id": "real-synthetic-checkpoint",
            "reset": True, "observation_timestamp_s": 1.0,
        }
        result = HttpPolicyTransport(
            f"http://127.0.0.1:{server.server_address[1]}",
        ).request(request, 10)
        assert result.error is None and result.response is not None
        assert len(result.response["action"]) == 7
        assert np.isfinite(result.response["action"]).all()
        assert result.response["request_id"] == request["request_id"]
        assert identity.metadata()["eligible_task_ids"] == ["die_into_cup"]
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
    assert not worker.is_alive()


@pytest.mark.parametrize(
    "fault", ["update", "held_out_prediction", "final_manifest", "held_out_and_checkpoint_manifest"]
)
def test_real_runtime_failures_keep_one_policy_row_and_preserve_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    config, dataset = training_fixture(tmp_path, monkeypatch)
    import lerobot.scripts.lerobot_train as upstream
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    import synria_lerobot.act_training as implementation

    if fault in {"update", "final_manifest"}:

        def fail_update(*args: object, **kwargs: object) -> object:
            raise RuntimeError("synthetic optimizer failure")

        monkeypatch.setattr(upstream, "update_policy", fail_update)
    else:
        monkeypatch.setattr(
            ACTPolicy,
            "predict_action_chunk",
            lambda self, batch: torch.full((len(batch["observation.state"]), 6, 7), float("nan")),
        )
    if fault in {"final_manifest", "held_out_and_checkpoint_manifest"}:
        write = implementation._write

        def fail_manifest(path: Path, value: object) -> None:
            if (fault == "final_manifest" and path.name == "training_run.json") or (
                fault == "held_out_and_checkpoint_manifest"
                and path.name.startswith("checkpoint-step-")
            ):
                raise OSError("synthetic manifest write failure")
            write(path, value)

        monkeypatch.setattr(implementation, "_write", fail_manifest)
    with pytest.raises(RuntimeError, match="inspect evidence") as caught:
        train(dataset, dataset / "physical_contract.json", tmp_path / "raw-training", 7, 2, config)
    repository = Path(config["repository"])
    evidence = repository / config["evidence_directory"]
    ledger = (repository / config["policy_ledger"]).read_text()
    assert sum(line.startswith("|") for line in ledger.splitlines()) == 1
    assert "failed" in ledger and "response_timeout_s=0.3" in ledger
    if fault in {"update", "final_manifest"}:
        assert str(caught.value.__cause__) == "synthetic optimizer failure"
        assert "N/A (no finalized checkpoint)" in ledger
        assert not (tmp_path / "raw-training/checkpoints").exists()
    if fault == "final_manifest":
        errors = json.loads((evidence / "finalization_errors.json").read_text())
        assert errors["primary_error"] == "synthetic optimizer failure"
        assert "manifest write failure" in errors["errors"][0]
    else:
        manifest = json.loads((evidence / "training_run.json").read_text())
        assert manifest["status"] == "failed"
        if fault.startswith("held_out"):
            assert len(manifest["checkpoints"]) == 1
            evaluation = json.loads(next((evidence / "evaluations").glob("*.json")).read_text())
            assert evaluation["status"] == "failed" and evaluation["metrics"] is None
            assert len((repository / config["timeline"]).read_text().splitlines()) == 1
            assert "checkpoint evaluation failed" in str(caught.value.__cause__)
            if fault == "held_out_and_checkpoint_manifest":
                assert "manifest write failure" in manifest["checkpoints"][0]["evidence_error"]


def test_uncommitted_probe_refuses_before_model_or_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset = training_fixture(tmp_path, monkeypatch)
    from lerobot.policies.act.modeling_act import ACTPolicy

    def never_construct(*args: object, **kwargs: object) -> None:
        pytest.fail("model constructed before frozen probe gate")

    monkeypatch.setattr(ACTPolicy, "__init__", never_construct)
    probe = Path(config["repository"]) / config["probe_set"]
    probe.write_text(probe.read_text() + " ")
    with pytest.raises(ValueError, match="committed"):
        train(dataset, dataset / "physical_contract.json", tmp_path / "raw-training", 7, 2, config)
    assert not (tmp_path / "raw-training").exists()
    assert not (Path(config["repository"]) / config["evidence_directory"]).exists()


@pytest.mark.parametrize("fault", ["physical_manifest", "upstream_save"])
def test_incomplete_checkpoint_preparation_or_save_is_recorded_without_plausible_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    config, dataset = training_fixture(tmp_path, monkeypatch)
    import lerobot.common.train_utils as upstream

    import synria_lerobot.act_training as implementation

    calls = []
    if fault == "physical_manifest":
        write = implementation._write

        def fail_manifest(path: Path, value: object) -> None:
            if path.name == "physical_policy_manifest.json":
                raise OSError("synthetic physical manifest failure")
            write(path, value)

        monkeypatch.setattr(implementation, "_write", fail_manifest)
        monkeypatch.setattr(upstream, "save_checkpoint", lambda *args, **kwargs: calls.append(args))
    else:

        def fail_save(checkpoint: Path, *args: object, **kwargs: object) -> None:
            calls.append(checkpoint)
            (checkpoint / "partial.bin").write_bytes(b"incomplete synthetic save")
            raise OSError("synthetic upstream save failure")

        monkeypatch.setattr(upstream, "save_checkpoint", fail_save)
    with pytest.raises(RuntimeError, match="inspect evidence"):
        train(dataset, dataset / "physical_contract.json", tmp_path / "raw-training", 7, 2, config)
    repository = Path(config["repository"])
    evidence = repository / config["evidence_directory"]
    manifest = json.loads((evidence / "training_run.json").read_text())
    assert manifest["completed_steps"] == 1 and not manifest["checkpoints"]
    assert len(manifest["partial_checkpoints"]) == 1
    partial = manifest["partial_checkpoints"][0]
    assert "do not load" in partial["status"] and "checkpoint_content_sha256" not in partial
    assert len(calls) == (0 if fault == "physical_manifest" else 1)
    ledger = (repository / config["policy_ledger"]).read_text()
    assert "N/A (no finalized checkpoint)" in ledger
    assert sum(line.startswith("|") for line in ledger.splitlines()) == 1
    assert not (repository / config["timeline"]).exists()

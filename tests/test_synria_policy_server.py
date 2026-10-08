"""Fixed-scene HTTP serving uses stub chunks and synthetic checkpoint receipts."""

from __future__ import annotations

import builtins
import copy
import json
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from test_synria_act_training import settings
from test_task_registry import synthetic_registry, synthetic_task

from synria_lerobot.act_training import FEATURE_KEYS, TRAINING_VERSION, validate_config
from synria_lerobot.checkpoint_eval import strict_content_hash
from synria_lerobot.policy_client import HttpPolicyTransport
from synria_lerobot.policy_codec import (
    MAX_REQUEST_BYTES,
    decode_policy_response,
    encode_policy_request,
)
from synria_lerobot.policy_server import (
    CheckpointIdentity,
    FixedScenePolicyService,
    LocalACTModel,
    make_http_server,
)


class StubModel:
    image_shapes = {
        "observation.images.wrist": (4, 4, 3),
        "observation.images.front": (4, 4, 3),
    }

    def __init__(self) -> None:
        self.reset_count = self.calls = 0
        self.chunk = np.repeat(np.arange(12, dtype=float)[:, None], 7, axis=1)

    def reset(self) -> None:
        self.reset_count += 1

    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        assert observation["task"]
        return self.chunk


def fixture(
    tmp_path: Path,
    task_id: str = "die_into_cup",
) -> tuple[CheckpointIdentity, Path, Path]:
    config, contract = settings(tmp_path)
    contract.update(synthetic_task(0.1, 1, task_id).metadata())
    config["chunk_size"] = 12
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "version": TRAINING_VERSION,
        "policy_id": config["policy_id"],
        "step": 2,
        "completed_steps": 2,
        "physical_contract": contract,
        "dataset_content_sha256": "a" * 64,
        "probe_sha256": config["probe_sha256"],
        "configuration": config,
        "cadence": validate_config(config, contract),
        "fixed_scene_motion_eligible": True,
        "eligible_task_ids": [task_id],
        "varying_target_motion_eligible": False,
        "task_text_conditioning": False,
        "declared_input_conditioning": list(FEATURE_KEYS),
        "model_feature_shapes": {
            "observation.state": [7],
            "action": [7],
            "observation.images.wrist": [3, 4, 4],
            "observation.images.front": [3, 4, 4],
        },
    }
    (checkpoint / "physical_policy_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    (checkpoint / "synthetic_weights.txt").write_text("stub; not a trained model")
    receipt = tmp_path / "checkpoint-record.json"
    receipt.write_text(
        json.dumps(
            {
                **manifest,
                "path": "/original-host/checkpoint",
                "checkpoint_content_sha256": strict_content_hash(checkpoint),
                "evaluation_status": "evaluated",
            }
        )
    )
    registry = synthetic_registry(tmp_path / "registry.json", 0.1, 1)
    return CheckpointIdentity.load(checkpoint, receipt, registry), receipt, registry


def payload(identity: CheckpointIdentity, *, reset: bool = False) -> dict[str, Any]:
    return {
        **identity.contract.as_dict(),
        "embodiment": "synria_alicia_d",
        "observation.state": [0.0] * 7,
        "observation.images.wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.front": np.full((4, 4, 3), 255, dtype=np.uint8),
        "timestamps": dict(state=1.0, state_ros=10.0, wrist=1.0, front=1.0),
        "task": identity.contract.task_text,
        "request_id": "synthetic-request",
        "reset": reset,
        "observation_timestamp_s": 1.0,
    }


def test_service_preserves_contract_and_cadence_and_reset(tmp_path: Path) -> None:
    identity, _, _ = fixture(tmp_path)
    model = StubModel()
    service = FixedScenePolicyService(identity, model)

    def request(reset: bool = False) -> dict[str, Any]:
        return decode_policy_response(
            service.request(encode_policy_request(payload(identity, reset=reset)))
        )

    first, second, third = request(True), request(), request()
    assert [result["action"][0] for result in (first, second, third)] == [0, 6, 0]
    assert model.calls == 2 and model.reset_count == 1
    assert request(True)["action"][0] == 0
    assert model.reset_count == 2
    assert first["request_id"] == "synthetic-request" and first["observation_timestamp_s"] == 1
    assert first["inference_s"] >= 0 and first["server_request_decode_s"] >= 0
    assert second["inference_s"] == 0


@pytest.mark.parametrize("task_id", ["die_into_cup", "roll_and_dump", "cup_return"])
def test_only_one_fixed_trained_roll_skill_is_served(tmp_path: Path, task_id: str) -> None:
    identity, _, _ = fixture(tmp_path, task_id)
    model = StubModel()
    service = FixedScenePolicyService(identity, model)
    assert identity.metadata()["eligible_task_ids"] == [task_id]
    assert decode_policy_response(service.request(encode_policy_request(payload(identity))))[
        "action"
    ]
    other = payload(identity, reset=True)
    other["task_id"] = "cup_return" if task_id != "cup_return" else "die_into_cup"
    with pytest.raises(ValueError, match="task identity"):
        service.request(encode_policy_request(other))
    assert model.reset_count == 0


def test_reordered_nested_json_keeps_the_same_task_and_contract(tmp_path: Path) -> None:
    identity, _, _ = fixture(tmp_path)
    model = StubModel()
    service = FixedScenePolicyService(identity, model)
    packet = json.loads(encode_policy_request(payload(identity)))
    reordered = json.dumps(packet, sort_keys=True).encode()
    assert decode_policy_response(service.request(reordered))["action"] == [0.0] * 7


@pytest.mark.parametrize(
    "change",
    [
        "task_id",
        "text",
        "goal",
        "lookahead",
        "contract",
        "window",
        "reset",
        "timestamp",
        "stamp_keys",
        "state",
        "gripper",
        "image_shape",
        "identifier",
        "contract_boolean",
    ],
)
def test_invalid_request_does_not_reset_or_consume_pending_chunk(
    tmp_path: Path, change: str
) -> None:
    identity, _, _ = fixture(tmp_path)
    model = StubModel()
    service = FixedScenePolicyService(identity, model)
    service.request(encode_policy_request(payload(identity)))
    broken = payload(identity, reset=True)
    if change == "task_id":
        broken["task_id"] = "cup_return"
    elif change == "text":
        broken["task"] = "place token at arbitrary goal"
    elif change == "goal":
        broken["goal"] = [1, 2, 3]
    elif change == "lookahead":
        broken["action_lookahead_steps"] = 2
    elif change == "contract":
        broken["contract_version"] = "unknown"
    elif change == "window":
        broken["min_episode_s"] = 0.2
    elif change == "reset":
        broken["reset"] = 1
    elif change == "timestamp":
        broken["observation_timestamp_s"] = 2
    elif change == "stamp_keys":
        broken["timestamps"].pop("front")
    elif change == "state":
        broken["observation.state"][0] = True
    elif change == "gripper":
        broken["observation.state"][6] = 0.1
    elif change == "image_shape":
        broken["observation.images.front"] = np.zeros((8, 8, 3), dtype=np.uint8)
    elif change == "contract_boolean":
        broken["state_has_velocity"] = 0
    else:
        broken["request_id"] = "a" * 129
    with pytest.raises(ValueError):
        service.request(encode_policy_request(broken))
    assert model.reset_count == 0 and model.calls == 1
    assert (
        decode_policy_response(service.request(encode_policy_request(payload(identity))))["action"][
            0
        ]
        == 6
    )


@pytest.mark.parametrize(
    "chunk",
    [
        np.zeros((11, 7)),
        np.zeros((12, 6)),
        np.full((12, 7), np.nan),
        np.full((12, 7), np.inf),
        np.full((12, 7), True),
        np.full((12, 7), "bad"),
    ],
)
def test_bad_chunk_resets_and_leaves_no_action_to_reuse(tmp_path: Path, chunk: np.ndarray) -> None:
    identity, _, _ = fixture(tmp_path)
    model = StubModel()
    model.chunk = chunk
    service = FixedScenePolicyService(identity, model)
    with pytest.raises(ValueError, match="finite seven-value"):
        service.request(encode_policy_request(payload(identity)))
    assert model.reset_count == 1
    model.chunk = np.ones((12, 7))
    assert (
        decode_policy_response(service.request(encode_policy_request(payload(identity))))["action"]
        == [1.0] * 7
    )
    assert model.calls == 2


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "partial",
        "hash",
        "receipt_contract",
        "manifest_recipe",
        "cadence",
        "eligibility",
        "goal",
        "registry",
        "changed_weights",
        "step",
    ],
)
def test_incomplete_or_conflicting_checkpoint_is_refused(tmp_path: Path, change: str) -> None:
    identity, receipt, registry = fixture(tmp_path)
    record = json.loads(receipt.read_text())
    path = identity.checkpoint / "physical_policy_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "missing":
        receipt.unlink()
    elif change == "partial":
        record["evaluation_status"] = "failed"
    elif change == "hash":
        record["checkpoint_content_sha256"] = "0" * 64
    elif change == "receipt_contract":
        record["physical_contract"]["task_id"] = "cup_return"
    elif change == "manifest_recipe":
        manifest["configuration"]["response_timeout_s"] = 1
    elif change == "changed_weights":
        (identity.checkpoint / "synthetic_weights.txt").write_text("changed")
    elif change == "registry":
        tasks = json.loads(registry.read_text())
        tasks["tasks"][0]["task_text"] = "Changed instruction"
        registry.write_text(json.dumps(tasks))
    elif change == "goal":
        tasks = json.loads(registry.read_text())
        tasks["tasks"][0]["requires_unobserved_goal"] = True
        registry.write_text(json.dumps(tasks))
    else:
        if change == "cadence":
            manifest["cadence"]["command_stride_steps"] = 1
        elif change == "step":
            manifest["step"] = True
        else:
            manifest["fixed_scene_motion_eligible"] = False
        record.update(copy.deepcopy(manifest))
        path.write_text(json.dumps(manifest))
        record["checkpoint_content_sha256"] = strict_content_hash(identity.checkpoint)
    if change != "missing":
        receipt.write_text(json.dumps(record))
    if change == "manifest_recipe":
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        CheckpointIdentity.load(identity.checkpoint, receipt, registry)


def test_http_protocol_transport_metadata_and_malformed_body(tmp_path: Path) -> None:
    identity, _, _ = fixture(tmp_path)
    service = FixedScenePolicyService(identity, StubModel())
    server = make_http_server(service, port=0)
    assert server.server_address[0] == "127.0.0.1"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = HttpPolicyTransport(endpoint).request(payload(identity, reset=True), 2)
        assert result.error is None and result.response["action"] == [0.0] * 7
        assert result.request_encode_s >= 0 and result.response_decode_s >= 0
        with urlopen(endpoint + "/metadata", timeout=2) as response:
            metadata = decode_policy_response(response.read())
        assert metadata == identity.metadata()
        assert HttpPolicyTransport(endpoint).metadata(2) == metadata
        metadata["eligible_task_ids"].append("cup_return")
        assert identity.metadata()["eligible_task_ids"] == ["die_into_cup"]
        for body in (b"not JSON",):
            with pytest.raises(HTTPError) as caught:
                urlopen(Request(endpoint, body, {"Content-Type": "application/json"}), timeout=2)
            assert caught.value.code == 400
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        try:
            connection.putrequest("POST", "/")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 400
            assert "oversized" in decode_policy_response(response.read())["error"]
        finally:
            connection.close()
        # Rejections do not consume the next valid chunk.
        assert (
            HttpPolicyTransport(endpoint).request(payload(identity), 2).response["action"]
            == [6.0] * 7
        )
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
    assert not worker.is_alive()


def test_service_hash_rechecks_before_using_stub_model(tmp_path: Path) -> None:
    identity, _, _ = fixture(tmp_path)
    (identity.checkpoint / "synthetic_weights.txt").write_text("changed during load")
    with pytest.raises(ValueError, match="changed"):
        FixedScenePolicyService(identity, StubModel())


def _processor_artifacts(checkpoint: Path) -> Path:
    local = checkpoint / "pretrained_model"
    local.mkdir()
    (local / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "pretrained_backbone_weights": None,
            }
        )
    )
    for name, names in {
        "policy_preprocessor": [
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
        ],
        "policy_postprocessor": ["unnormalizer_processor", "device_processor"],
    }.items():
        steps = []
        for registered in names:
            config = {}
            if registered == "rename_observations_processor":
                config = {"rename_map": {}}
            elif registered == "device_processor":
                config = {"device": "cpu", "float_dtype": None}
            step = {"registry_name": registered, "config": config}
            if registered in {"normalizer_processor", "unnormalizer_processor"}:
                step["state_file"] = f"{registered}.safetensors"
                (local / step["state_file"]).write_bytes(b"synthetic; never deserialized")
            steps.append(step)
        (local / f"{name}.json").write_text(json.dumps({"name": name, "steps": steps}))
    return local


@pytest.mark.parametrize(
    "change",
    [
        "class",
        "unregistered",
        "absolute_state",
        "parent_state",
        "missing_state",
        "executable_state",
        "backbone_download",
        "missing_backbone_declaration",
        "rename",
    ],
)
def test_local_loader_refuses_custom_imports_escaped_state_or_download_before_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    identity, receipt, registry = fixture(tmp_path)
    local = _processor_artifacts(identity.checkpoint)
    path = local / "policy_preprocessor.json"
    processors = json.loads(path.read_text())
    if change in {"backbone_download", "missing_backbone_declaration"}:
        config = {"type": "act"}
        if change == "backbone_download":
            config["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
        (local / "config.json").write_text(json.dumps(config))
    elif change == "class":
        processors["steps"][0] = {"class": "synthetic_module.NeverImport", "config": {}}
    elif change == "unregistered":
        processors["steps"][0]["registry_name"] = "custom_processor"
    elif change == "rename":
        processors["steps"][0]["config"]["rename_map"] = {"goal": "observation.state"}
    else:
        processors["steps"][-1]["state_file"] = {
            "absolute_state": str(tmp_path / "outside.safetensors"),
            "parent_state": "../outside.safetensors",
            "missing_state": "missing.safetensors",
            "executable_state": "optimizer.pkl",
        }[change]
    path.write_text(json.dumps(processors))
    record = json.loads(receipt.read_text())
    record["checkpoint_content_sha256"] = strict_content_hash(identity.checkpoint)
    receipt.write_text(json.dumps(record))
    identity = CheckpointIdentity.load(identity.checkpoint, receipt, registry)
    imports = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("lerobot") or name == "synthetic_module":
            imports.append(name)
            raise AssertionError("model/processor factory import must not occur")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError):
        LocalACTModel(identity, "cpu")
    assert imports == []

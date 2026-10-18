"""Serve one hash-bound fixed-scene checkpoint without robot interfaces."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .act_training import (
    FEATURE_KEYS,
    TRAINING_VERSION,
    validate_config,
    validate_temporal_ensemble,
)
from .checkpoint_eval import strict_content_hash
from .physical_contract import (
    IMAGE_KEYS,
    PHYSICAL_LANE,
    SIMULATION_LANE,
    TRAINING_LANES,
    PhysicalDatasetContract,
    training_lane,
)
from .policy_codec import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    decode_policy_request,
    decode_policy_response,
    encode_policy_response,
)
from .task_registry import load_task_registry


def _read_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return decode_policy_response(stream.read(MAX_RESPONSE_BYTES + 1))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a nonnegative finite number")
    return float(value)


@dataclass(frozen=True)
class CheckpointIdentity:
    checkpoint: Path
    content_sha256: str
    manifest: dict[str, Any]
    contract: PhysicalDatasetContract
    lane: str = PHYSICAL_LANE

    @classmethod
    def load(
        cls, checkpoint: Path, receipt: Path, registry: Path, lane: str = PHYSICAL_LANE,
    ) -> CheckpointIdentity:
        """A finalized diagnostic receipt confirms completeness, not policy quality.

        The checkpoint must belong to the serving lane: a physical server never loads a
        simulation checkpoint, and a simulation server never loads a physical one.
        """
        record = _read_record(receipt)
        manifest = _read_record(checkpoint / "physical_policy_manifest.json")
        if set(record) != set(manifest) | {
            "path",
            "checkpoint_content_sha256",
            "evaluation_status",
        } or _canonical({key: record[key] for key in manifest}) != (_canonical(manifest)):
            raise ValueError("checkpoint completion receipt differs from its manifest")
        if record.get("evaluation_status") != "evaluated":
            raise ValueError("checkpoint requires a completed diagnostic evaluation receipt")
        digest = record.get("checkpoint_content_sha256")
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("checkpoint receipt requires a content hash")
        if strict_content_hash(checkpoint) != digest:
            raise ValueError("checkpoint content hash differs from completion receipt")
        if manifest.get("version") != TRAINING_VERSION:
            raise ValueError("unsupported checkpoint training manifest")
        for key in ("dataset_content_sha256", "probe_sha256"):
            if (
                type(manifest.get(key)) is not str
                or re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is None
            ):
                raise ValueError("checkpoint requires dataset and frozen probe hashes")
        if (
            type(manifest.get("step")) is not int
            or manifest["step"] <= 0
            or type(manifest.get("completed_steps")) is not int
            or (manifest.get("completed_steps") != manifest["step"])
        ):
            raise ValueError("checkpoint receipt step is inconsistent")
        contract = PhysicalDatasetContract.from_dict(manifest["physical_contract"])
        lane = training_lane({"lane": lane})
        contract.require_lane(lane)
        if training_lane(manifest) != lane or training_lane(
            manifest.get("configuration") or {}
        ) != lane:
            raise ValueError("checkpoint lane differs from the serving lane")
        tasks = load_task_registry(registry)
        if tasks[contract.task_id] != contract.task_definition:
            raise ValueError("checkpoint task snapshot differs from the fixed-scene registry")
        if (
            manifest.get("fixed_scene_motion_eligible") is not (lane == PHYSICAL_LANE)
            or (lane == SIMULATION_LANE
                and manifest.get("simulation_motion_eligible") is not True)
            or manifest.get("eligible_task_ids") != [contract.task_id]
            or manifest.get("varying_target_motion_eligible") is not False
            or manifest.get("task_text_conditioning") is not False
            or manifest.get("declared_input_conditioning") != list(FEATURE_KEYS)
        ):
            raise ValueError("checkpoint is not eligible for its single fixed-scene task")
        cadence = validate_config(manifest["configuration"], manifest["physical_contract"])
        if type(manifest.get("cadence")) is not dict or _canonical(
            manifest["cadence"]
        ) != _canonical(cadence):
            raise ValueError("checkpoint cadence differs from its training configuration")
        if manifest.get("policy_id") != manifest["configuration"]["policy_id"] or (
            manifest["probe_sha256"] != manifest["configuration"]["probe_sha256"]
        ):
            raise ValueError("checkpoint policy/probe identity differs from its configuration")
        shapes = manifest.get("model_feature_shapes")
        state_size = 13 if contract.state_has_velocity else 7
        if (
            type(shapes) is not dict
            or set(shapes) != {*FEATURE_KEYS, "action"}
            or (shapes["observation.state"] != [state_size] or shapes["action"] != [7])
        ):
            raise ValueError("checkpoint requires physical model feature shapes")
        if any(
            type(value) is not int
            for key in ("observation.state", "action")
            for value in shapes[key]
        ):
            raise ValueError("checkpoint state/action dimensions must be integers")
        for key in IMAGE_KEYS:
            shape = shapes[key]
            if (
                type(shape) is not list
                or len(shape) != 3
                or shape[0] != 3
                or any(type(side) is not int or not 0 < side <= 512 for side in shape)
            ):
                raise ValueError("checkpoint image feature shapes must be bounded CHW RGB")
        # Original absolute receipt paths may differ on the serving host; identity is content.
        return cls(checkpoint.resolve(), digest, copy.deepcopy(manifest), contract, lane)

    def verify_content(self) -> None:
        if strict_content_hash(self.checkpoint) != self.content_sha256:
            raise ValueError("checkpoint changed while loading the local policy")

    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "policy_id": self.manifest["policy_id"],
                "step": self.manifest["step"],
                "checkpoint_content_sha256": self.content_sha256,
                "dataset_content_sha256": self.manifest["dataset_content_sha256"],
                "probe_sha256": self.manifest["probe_sha256"],
                "task_id": self.contract.task_id,
                "physical_contract": self.manifest["physical_contract"],
                "cadence": self.manifest["cadence"],
                "response_timeout_s": self.manifest["configuration"]["response_timeout_s"],
                "temporal_ensemble_coeff": self.manifest["configuration"].get(
                    "temporal_ensemble_coeff"
                ),
                "image_transforms": self.manifest["configuration"].get("image_transforms"),
                "model_feature_shapes": self.manifest["model_feature_shapes"],
                "lane": self.lane,
                "physical_eligible": self.lane == PHYSICAL_LANE,
                "fixed_scene_motion_eligible": self.lane == PHYSICAL_LANE,
                "simulation_motion_eligible": self.lane == SIMULATION_LANE,
                "eligible_task_ids": [self.contract.task_id],
                "varying_target_motion_eligible": False,
                "task_text_conditioning": False,
                "status": "implemented, unmeasured",
                "stage_status": "planned",
            }
        )


class ChunkModel(Protocol):
    image_shapes: dict[str, tuple[int, int, int]]

    def reset(self) -> None: ...
    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray: ...


def _validate_local_artifacts(local: Path) -> None:
    """Permit only wrapper-produced registered ACT steps and confined tensor state.

    Upstream processor pipelines support custom class imports and external state
    paths; this serving path deliberately accepts neither extension.
    """
    config = _read_record(local / "config.json")
    if (
        config.get("type") != "act"
        or "pretrained_backbone_weights" not in config
        or (config["pretrained_backbone_weights"] is not None)
    ):
        raise ValueError("local ACT configuration must disable pretrained backbone downloads")
    sequences = {
        "policy_preprocessor": (
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
        ),
        "policy_postprocessor": ("unnormalizer_processor", "device_processor"),
    }
    for name, sequence in sequences.items():
        pipeline = _read_record(local / f"{name}.json")
        if (
            set(pipeline) != {"name", "steps"}
            or pipeline["name"] != name
            or (type(pipeline["steps"]) is not list or len(pipeline["steps"]) != len(sequence))
        ):
            raise ValueError("only saved default ACT processor pipelines are supported")
        for step, registered in zip(pipeline["steps"], sequence, strict=True):
            stateful = registered in {"normalizer_processor", "unnormalizer_processor"}
            keys = (
                {"registry_name", "config", "state_file"}
                if stateful
                else {
                    "registry_name",
                    "config",
                }
            )
            if (
                type(step) is not dict
                or set(step) != keys
                or (step["registry_name"] != registered or type(step["config"]) is not dict)
            ):
                raise ValueError("custom processor classes/imports are not supported")
            if stateful:
                state = step["state_file"]
                if (
                    type(state) is not str
                    or re.fullmatch(
                        r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors",
                        state,
                    )
                    is None
                ):
                    raise ValueError("processor state must be a confined safetensors filename")
                path = local / state
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.resolve().parent != local.resolve()
                ):
                    raise ValueError("processor state must stay inside the hashed local checkpoint")
            elif registered == "rename_observations_processor" and step["config"] != {
                "rename_map": {},
            }:
                raise ValueError("saved ACT observation names cannot be remapped")
            elif registered == "to_batch_processor" and step["config"]:
                raise ValueError("saved ACT batch processor must use default settings")
            elif registered == "device_processor" and (
                set(step["config"]) != {"device", "float_dtype"}
                or step["config"]["device"] not in {"cpu", "cuda"}
                or step["config"]["float_dtype"] is not None
            ):
                raise ValueError("saved ACT device processor must use default settings")


class LocalACTModel:
    """Load upstream local safetensors and saved processors, never training state.

    Public APIs: https://huggingface.co/docs/lerobot/act and
    https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/factory.py
    """

    def __init__(self, identity: CheckpointIdentity, device: str) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("explicit cpu or cuda serving device required")
        identity.verify_content()
        local = identity.checkpoint / "pretrained_model"
        _validate_local_artifacts(local)
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = ACTConfig.from_pretrained(local, local_files_only=True)
        if config.pretrained_backbone_weights is not None:
            raise ValueError("local ACT configuration must disable pretrained backbone downloads")
        config.device = device
        shapes = {
            key: list(feature.shape)
            for key, feature in {**config.input_features, **config.output_features}.items()
        }
        if shapes != identity.manifest["model_feature_shapes"]:
            raise ValueError("saved model shapes differ from checkpoint evidence")
        state_size = 13 if identity.contract.state_has_velocity else 7
        if (
            set(config.input_features) != set(FEATURE_KEYS)
            or set(config.output_features)
            != {
                "action",
            }
            or tuple(config.input_features["observation.state"].shape) != (state_size,)
            or (tuple(config.output_features["action"].shape) != (7,))
            or config.chunk_size != identity.manifest["configuration"]["chunk_size"]
        ):
            raise ValueError("saved model features differ from its physical manifest")
        self.image_shapes = {}
        for key in IMAGE_KEYS:
            chw = tuple(config.input_features[key].shape)
            if len(chw) != 3 or chw[0] != 3:
                raise ValueError("saved model requires CHW RGB images")
            self.image_shapes[key] = (chw[1], chw[2], chw[0])
        self.policy = ACTPolicy.from_pretrained(
            local,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(local),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.policy.eval()
        self.torch = torch
        identity.verify_content()

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        torch = self.torch
        batch = {
            "observation.state": torch.tensor(
                [observation["observation.state"]],
                dtype=torch.float32,
            ),
        }
        for key in IMAGE_KEYS:
            # Codec arrays are read-only; own the tensor buffer before normalization.
            image = np.array(observation[key], dtype=np.float32, copy=True) / 255.0
            batch[key] = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            prediction = self.postprocessor(
                self.policy.predict_action_chunk(self.preprocessor(batch)),
            )
        return cast(np.ndarray, np.asarray(prediction.detach().cpu().numpy())[0])


class TemporalEnsemble:
    """Temporal ensembling over overlapping chunks at the serving cadence.

    Every request predicts a fresh chunk; the action returned for the current step is
    the weighted mean of what every still-overlapping chunk predicted for that step,
    with weights exp(-coeff * age) where the oldest chunk has age 0, as in the ACT
    paper (coefficient 0 weighs chunks equally; a positive coefficient favours older
    chunks). Time advances by the command stride between requests, so a chunk of
    ``chunk_size`` dataset steps stays in the ensemble for ``chunk_size // stride``
    requests.
    """

    def __init__(self, chunk_size: int, stride: int, coeff: float) -> None:
        if type(chunk_size) is not int or chunk_size < 1 or type(stride) is not int or stride < 1:
            raise ValueError("ensemble chunk size and stride must be positive integers")
        if stride > chunk_size:
            raise ValueError("command stride must not exceed the chunk size")
        self.chunk_size, self.stride, self.coeff = chunk_size, stride, float(coeff)
        self.reset()

    def reset(self) -> None:
        """Forget every chunk; the next request starts a new ensemble."""
        self._chunks: list[tuple[int, np.ndarray]] = []
        self._now = 0

    def step(self, chunk: np.ndarray) -> list[float]:
        """Add a chunk predicted at the current step and return the ensembled action for it."""
        chunk = np.asarray(chunk, dtype=float)
        if chunk.shape != (self.chunk_size, 7) or not np.isfinite(chunk).all():
            raise ValueError("policy chunk must contain finite seven-value absolute actions")
        self._chunks = [(origin, c) for origin, c in self._chunks
                        if origin + self.chunk_size > self._now]
        self._chunks.append((self._now, chunk))
        rows = [c[self._now - origin] for origin, c in self._chunks]
        weights = np.exp(-self.coeff * np.arange(len(rows), dtype=float))
        action = (np.asarray(rows) * weights[:, None]).sum(axis=0) / weights.sum()
        self._now += self.stride
        return [float(value) for value in action]


class FixedScenePolicyService:
    """One model/queue; callers must reset at each independent skill trial.

    With a ``temporal_ensemble_coeff`` in the checkpoint's training configuration the
    service predicts on every request and ensembles overlapping chunks; otherwise it
    predicts once per chunk and hands out the chunk's actions at the command stride.
    ``serve_ensemble=False`` serves such a checkpoint chunk by chunk instead, as an
    ablation of the ensembling alone; the override is named in the metadata every
    trial report copies, so a campaign run that way can never pass as the manifest's
    own serving mode, and a physical session, which binds only metadata equal to the
    receipt, refuses such a server. The physical lane also refuses the override at
    the command line.
    """

    def __init__(
        self,
        identity: CheckpointIdentity,
        model: ChunkModel,
        *,
        clock: Callable[[], float] = time.perf_counter,
        serve_ensemble: bool = True,
    ) -> None:
        identity.verify_content()
        if set(model.image_shapes) != set(IMAGE_KEYS):
            raise ValueError("both saved model image shapes are required")
        for key in IMAGE_KEYS:
            shape = identity.manifest["model_feature_shapes"][key]
            if model.image_shapes[key] != (shape[1], shape[2], shape[0]):
                raise ValueError("loaded model image shapes differ from checkpoint evidence")
        self.identity, self.model, self.clock = identity, model, clock
        self._queue: deque[list[float]] = deque()
        self._lock = threading.Lock()
        coeff = validate_temporal_ensemble(
            identity.manifest["configuration"].get("temporal_ensemble_coeff")
        )
        self.serving_overrides: dict[str, str] = {}
        if coeff is not None and not serve_ensemble:
            self.serving_overrides["temporal_ensemble"] = "off"
            coeff = None
        self.ensemble = None if coeff is None else TemporalEnsemble(
            identity.manifest["configuration"]["chunk_size"],
            identity.manifest["cadence"]["command_stride_steps"], coeff,
        )

    def metadata(self) -> dict[str, Any]:
        """The checkpoint's metadata, plus the serving overrides only when one is active.

        A physical session binds an endpoint only when its metadata equals the frozen
        receipt's byte for byte, so the key is absent when nothing is overridden and a
        server with an override can never be bound to the physical arm.
        """
        metadata = self.identity.metadata()
        if self.serving_overrides:
            metadata["serving_overrides"] = dict(self.serving_overrides)
        return metadata

    def _validate(self, payload: dict[str, Any]) -> None:
        contract = self.identity.contract.as_dict()
        required = set(contract) | {
            "embodiment",
            "observation.state",
            *IMAGE_KEYS,
            "timestamps",
            "task",
            "request_id",
            "reset",
            "observation_timestamp_s",
        }
        if set(payload) != required or _canonical(
            {key: payload[key] for key in contract}
        ) != _canonical(contract):
            raise ValueError("request contract/task identity differs from the served checkpoint")
        if payload["embodiment"] != "synria_alicia_d" or payload["task"] != (
            self.identity.contract.task_text
        ):
            raise ValueError("request task must be the fixed trained instruction")
        if (
            type(payload["request_id"]) is not str
            or re.fullmatch(
                r"[A-Za-z0-9_-]{1,128}",
                payload["request_id"],
            )
            is None
            or type(payload["reset"]) is not bool
        ):
            raise ValueError("request requires a bounded identifier and boolean reset")
        state = payload["observation.state"]
        if (
            type(state) not in (list, tuple)
            or len(state) != (13 if self.identity.contract.state_has_velocity else 7)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in state)
        ):
            raise ValueError("invalid physical observation state")
        if not 0 <= state[6] <= self.identity.contract.gripper_stroke_m:
            raise ValueError("state gripper differs from the contract range")
        stamps = payload["timestamps"]
        if type(stamps) is not dict or set(stamps) != {"state", "state_ros", "wrist", "front"}:
            raise ValueError("all physical observation timestamps are required")
        for stamp in stamps.values():
            _finite(stamp, "observation timestamp")
        if _finite(payload["observation_timestamp_s"], "observation timestamp") != stamps["state"]:
            raise ValueError("observation timestamp differs from state source")
        for key in IMAGE_KEYS:
            image = payload[key]
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or tuple(image.shape) != self.model.image_shapes[key]
            ):
                raise ValueError("request image shape/dtype differs from the saved model")

    def request(self, body: bytes) -> bytes:
        started = self.clock()
        payload = decode_policy_request(body)
        decode_s = _finite(self.clock() - started, "decode latency")
        self._validate(payload)  # Invalid requests cannot reset a model or consume a chunk.
        with self._lock:
            inference_s = 0.0
            try:
                if payload["reset"]:
                    self._queue.clear()
                    self.model.reset()
                    if self.ensemble is not None:
                        self.ensemble.reset()
                if self.ensemble is not None or not self._queue:
                    started = self.clock()
                    chunk = np.asarray(self.model.predict_chunk(payload))
                    if chunk.shape != (
                        self.identity.manifest["configuration"]["chunk_size"],
                        7,
                    ) or (chunk.dtype.kind not in "fiu" or not np.isfinite(chunk).all()):
                        raise ValueError(
                            "policy chunk must contain finite seven-value absolute actions"
                        )
                    if self.ensemble is not None:
                        self._queue.append(self.ensemble.step(chunk))
                    else:
                        stride = self.identity.manifest["cadence"]["command_stride_steps"]
                        self._queue.extend(chunk[::stride].astype(float).tolist())
                    inference_s = _finite(self.clock() - started, "inference latency")
                action = self._queue.popleft()
                return encode_policy_response(
                    {
                        "request_id": payload["request_id"],
                        "observation_timestamp_s": payload["observation_timestamp_s"],
                        "action": action,
                        "inference_s": inference_s,
                        "server_request_decode_s": decode_s,
                    }
                )
            except BaseException:
                self._queue.clear()
                if self.ensemble is not None:
                    self.ensemble.reset()
                try:
                    self.model.reset()
                except Exception:
                    pass
                raise


def make_http_server(
    service: FixedScenePolicyService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> HTTPServer:
    """Serial HTTP requests preserve model chunk order; body reads have a time bound."""

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(2.0)

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/metadata":
                self._send(404, encode_policy_response({"error": "unknown policy route"}))
                return
            self._send(200, encode_policy_response(service.metadata()))

        def do_POST(self) -> None:
            try:
                if self.path != "/" or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("policy requests require JSON on the root route")
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None
                    or (self.headers.get("Transfer-Encoding") is not None)
                ):
                    raise ValueError("one bounded Content-Length is required")
                length = int(lengths[0])
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("oversized or empty policy body")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("truncated policy body")
                response = service.request(body)
            except (ValueError, KeyError, TypeError, TimeoutError) as error:
                self._send(400, encode_policy_response({"error": str(error)[:512]}))
            except Exception:
                self._send(500, encode_policy_response({"error": "policy inference failed"}))
            else:
                self._send(200, response)

    return HTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one fixed-scene physical ACT checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-record", required=True, type=Path)
    parser.add_argument("--task-registry", type=Path, default=Path("config/synria_tasks.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lane", choices=TRAINING_LANES, default=PHYSICAL_LANE,
                        help="Serve a physical checkpoint (default) or a simulation-lane one.")
    parser.add_argument("--temporal-ensemble", choices=("manifest", "off"), default="manifest",
                        help="Serve a checkpoint trained for temporal ensembling without it "
                             "(simulation lane only; recorded in the metadata).")
    args = parser.parse_args(argv)
    if args.temporal_ensemble == "off" and args.lane != SIMULATION_LANE:
        parser.error("--temporal-ensemble off is a simulation-lane ablation only")
    identity = CheckpointIdentity.load(
        args.checkpoint, args.checkpoint_record, args.task_registry, args.lane,
    )
    service = FixedScenePolicyService(identity, LocalACTModel(identity, args.device),
                                      serve_ensemble=args.temporal_ensemble == "manifest")
    with make_http_server(service, args.host, args.port) as server:
        server.serve_forever()

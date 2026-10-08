"""Policy-wire checks use byte buffers, synthetic images and mocked HTTP only."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_task_registry import synthetic_task

from synria_lerobot import policy_codec as codec
from synria_lerobot.embodiment import SynriaEmbodiment, SynriaObservation
from synria_lerobot.physical_contract import (
    IMAGE_KEYS,
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.policy_client import HttpPolicyTransport, PolicyClient, PolicySafetyConfig
from synria_lerobot.quality_gates import load_limits


def payload(side: int = 4) -> dict[str, Any]:
    pixels = np.arange(side * side * 3, dtype=np.uint8).reshape(side, side, 3)
    return {
        IMAGE_KEYS[0]: pixels,
        IMAGE_KEYS[1]: 255 - pixels,
        "observation.state": [0.0] * 6 + [0.01],
        "request_id": "test-request", "observation_timestamp_s": 10.0, "task": "token A to B",
    }


def test_two_224_rgb_images_round_trip_as_uint8_bytes_without_pixel_lists() -> None:
    source = payload(224)
    encoded = codec.encode_policy_request(source)
    packet = json.loads(encoded)
    assert packet["codec"] == codec.CODEC_VERSION
    assert len(encoded) < codec.MAX_REQUEST_BYTES
    decoded = codec.decode_policy_request(encoded)
    for name in IMAGE_KEYS:
        image = packet["payload"][name]
        assert image["shape"] == [224, 224, 3]
        assert image["dtype"] == "uint8" and image["color_space"] == "RGB"
        assert image["encoding"] == "base64" and isinstance(image["data"], str)
        assert decoded[name].shape == (224, 224, 3) and decoded[name].dtype == np.uint8
        np.testing.assert_array_equal(source[name], decoded[name])
    assert decoded["observation.state"] == source["observation.state"]
    assert isinstance(source[IMAGE_KEYS[0]], np.ndarray)


def test_noncontiguous_rgb_array_retains_pixel_order() -> None:
    source = payload()
    source[IMAGE_KEYS[0]] = source[IMAGE_KEYS[0]][::-1, ::-1]
    decoded = codec.decode_policy_request(codec.encode_policy_request(source))
    np.testing.assert_array_equal(decoded[IMAGE_KEYS[0]], source[IMAGE_KEYS[0]])


@pytest.mark.parametrize("image", [
    None, [[[0, 1, 2]]], np.ones((4, 4, 3), dtype=np.float32),
    np.full((4, 4, 3), np.nan), np.zeros((4, 4, 4), np.uint8),
    np.zeros((0, 4, 3), np.uint8), np.zeros((513, 1, 3), np.uint8),
])
def test_encoder_rejects_bad_image_types_shapes_and_oversized_dimensions(image: Any) -> None:
    source = payload()
    source[IMAGE_KEYS[1]] = image
    with pytest.raises(ValueError):
        codec.encode_policy_request(source)


@pytest.mark.parametrize("shape", [
    [True, 4, 3], [4.0, 4, 3], [0, 4, 3], [-1, 4, 3], [4, 4], [4, 4, 1],
    [4, 4, 3, 1], [10**9, 10**9, 3], [513, 1, 3],
])
def test_decoder_checks_both_shapes_before_any_image_allocation(
    monkeypatch: pytest.MonkeyPatch, shape: list[Any]
) -> None:
    packet = json.loads(codec.encode_policy_request(payload()))
    packet["payload"][IMAGE_KEYS[1]]["shape"] = shape

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("image decoding must not start before both shape checks")

    monkeypatch.setattr(codec.base64, "b64decode", forbidden)
    with pytest.raises(ValueError):
        codec.decode_policy_request(json.dumps(packet).encode())


@pytest.mark.parametrize("field,value", [
    ("dtype", "float32"), ("dtype", "object"), ("encoding", "compressed"),
    ("color_space", "BGR"), ("data", None), ("data", ""), ("data", "!" * 64),
    ("data", "é" * 64), ("data", "A" * 62 + "=="),
])
def test_decoder_rejects_bad_image_metadata_base64_or_byte_length(field: str, value: Any) -> None:
    packet = json.loads(codec.encode_policy_request(payload()))
    packet["payload"][IMAGE_KEYS[0]][field] = value
    with pytest.raises(ValueError):
        codec.decode_policy_request(json.dumps(packet).encode())


@pytest.mark.parametrize("fault", ["missing", "extra", "pixel-list", "version", "extra-packet"])
def test_decoder_requires_exact_image_packets(fault: str) -> None:
    packet = json.loads(codec.encode_policy_request(payload()))
    if fault == "missing":
        del packet["payload"][IMAGE_KEYS[0]]
    elif fault == "extra":
        packet["payload"]["observation.images.third"] = packet["payload"][IMAGE_KEYS[0]]
    elif fault == "pixel-list":
        packet["payload"][IMAGE_KEYS[0]] = payload()[IMAGE_KEYS[0]].tolist()
    elif fault == "version":
        packet["codec"] = "unknown"
    else:
        packet["payload"][IMAGE_KEYS[0]]["other"] = 1
    with pytest.raises(ValueError):
        codec.decode_policy_request(json.dumps(packet).encode())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_request_values_are_rejected_on_both_sides(value: float) -> None:
    source = payload()
    source["observation.state"][0] = value
    with pytest.raises(ValueError, match="finite"):
        codec.encode_policy_request(source)
    packet = json.loads(codec.encode_policy_request(payload()))
    packet["payload"]["observation.state"][0] = value
    with pytest.raises(ValueError, match="finite"):
        codec.decode_policy_request(json.dumps(packet).encode())


def test_request_and_combined_image_size_caps_precede_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = codec.encode_policy_request(payload())
    monkeypatch.setattr(codec, "MAX_TOTAL_IMAGE_BYTES", 64)
    with pytest.raises(ValueError, match="combined"):
        codec.encode_policy_request(payload())

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("combined allocation must be checked first")

    monkeypatch.setattr(codec.base64, "b64decode", forbidden)
    with pytest.raises(ValueError, match="combined"):
        codec.decode_policy_request(encoded)
    monkeypatch.setattr(codec.json, "loads", forbidden)
    with pytest.raises(ValueError, match="oversized"):
        codec.decode_policy_request(b" " * (codec.MAX_REQUEST_BYTES + 1))


def test_encoder_caps_metadata_and_total_serialized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    source = payload()
    monkeypatch.setattr(codec, "MAX_REQUEST_BYTES", 1024)
    source["task"] = "x" * 1025
    with pytest.raises(ValueError, match="oversized"):
        codec.encode_policy_request(source)
    source["task"] = "x" * 800
    with pytest.raises(ValueError, match="oversized encoded"):
        codec.encode_policy_request(source)


@pytest.mark.parametrize("body", [
    b"[]", b"null", b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}',
    b'{"a":1,"a":2}', b'{"bad":', b'\xff',
])
def test_response_decoder_refuses_nonobjects_bad_json_and_nonfinite_values(body: bytes) -> None:
    with pytest.raises(ValueError):
        codec.decode_policy_response(body)


def test_response_cap_precedes_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("oversized response must not reach the parser")

    monkeypatch.setattr(codec.json, "loads", forbidden)
    with pytest.raises(ValueError, match="oversized"):
        codec.decode_policy_response(b" " * (codec.MAX_RESPONSE_BYTES + 1))


class Reply:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_limits: list[int] = []

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self.body[:limit]


def test_http_round_trip_records_local_encode_and_decode_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    replies = []

    def fake_open(request: Any, *, timeout: float) -> Reply:
        decoded = codec.decode_policy_request(request.data)
        seen.append(decoded)
        assert timeout == 0.5
        reply = Reply(json.dumps({
            "request_id": decoded["request_id"],
            "observation_timestamp_s": decoded["observation_timestamp_s"],
            "action": [0] * 7, "inference_s": 0.003,
            "request_encode_s": 999, "response_decode_s": 999,
        }).encode())
        replies.append(reply)
        return reply

    monkeypatch.setattr("synria_lerobot.policy_client.urlopen", fake_open)
    ticks = iter([1.0, 1.02, 1.04, 1.05])
    transport = HttpPolicyTransport("http://unused.test", clock=lambda: next(ticks))
    embodiment = SynriaEmbodiment(PhysicalDatasetContract("50mm", ActionSource.NEXT_STATE, False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    ))
    source = payload(224)
    observation = SynriaObservation(
        PhysicalState((0,) * 6, 0.01, 10, 100),
        ImageFrame(source[IMAGE_KEYS[0]], 10), ImageFrame(source[IMAGE_KEYS[1]], 10),
        "token A to B",
    )
    client = PolicyClient(
        embodiment, transport, load_limits(Path("config/synria_limits.yaml")),
        PolicySafetyConfig.load(
            Path("config/synria_policy.json"), command_period_s=0.4, response_timeout_s=0.5
        ), clock=lambda: 10.01,
    )
    decision = client.step(observation)
    assert not decision.hold and decision.inference_s == 0.003
    assert decision.request_encode_s == pytest.approx(0.02)
    assert decision.response_decode_s == pytest.approx(0.01)
    assert replies[0].read_limits == [codec.MAX_RESPONSE_BYTES + 1]
    assert all(
        seen[0][key] == value
        for key, value in embodiment.contract.task_definition.metadata().items()
    )
    for name in IMAGE_KEYS:
        np.testing.assert_array_equal(seen[0][name], source[name])
    client.write_latencies(tmp_path / "latencies.jsonl")
    row = json.loads((tmp_path / "latencies.jsonl").read_text())
    assert row["request_encode_s"] == pytest.approx(0.02)
    assert row["response_decode_s"] == pytest.approx(0.01)
    assert row["inference_s"] == 0.003 and "end_to_end_s" in row


@pytest.mark.parametrize(
    "bad_response", [b'[]', b'{"action":[NaN]}', b"x" * (1024 * 1024 + 1)],
    ids=["non-object", "nonfinite", "oversized"],
)
def test_transport_faults_return_request_local_timings(
    monkeypatch: pytest.MonkeyPatch, bad_response: bytes,
) -> None:
    monkeypatch.setattr("synria_lerobot.policy_client.urlopen", lambda *a, **k: Reply(bad_response))
    ticks = iter([0.0, 0.01, 0.02, 0.03])
    transport = HttpPolicyTransport("http://unused.test", clock=lambda: next(ticks))
    result = transport.request(payload(), 1)
    assert result.error and result.response is None
    assert result.request_encode_s == 0.01 and result.response_decode_s == pytest.approx(0.01)


def test_invalid_request_never_opens_a_network_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("synria_lerobot.policy_client.urlopen", lambda *a, **k: calls.append(a))
    source = payload()
    source[IMAGE_KEYS[0]] = np.ones((4, 4, 3), np.float32)
    result = HttpPolicyTransport("http://unused.test").request(source, 1)
    assert result.error and result.response is None and not calls
    assert result.request_encode_s is not None and result.response_decode_s is None


def test_shared_transport_keeps_concurrent_request_timings_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = threading.local()
    barrier = threading.Barrier(2)

    def clock() -> float:
        return next(local.ticks)

    def fake_open(request: Any, *, timeout: float) -> Reply:
        request_id = codec.decode_policy_request(request.data)["request_id"]
        barrier.wait(timeout=2)
        return Reply(json.dumps({"request_id": request_id}).encode())

    monkeypatch.setattr("synria_lerobot.policy_client.urlopen", fake_open)
    transport = HttpPolicyTransport("http://unused.test", clock=clock)

    def call(index: int) -> Any:
        local.ticks = iter([0, index, 10, 10 + index / 10])
        source = payload()
        source["request_id"] = str(index)
        return transport.request(source, 1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(call, [1, 2]))
    for index, result in enumerate(results, 1):
        assert result.response == {"request_id": str(index)}
        assert result.request_encode_s == index
        assert result.response_decode_s == pytest.approx(index / 10)
        assert result.error is None


def test_image_byte_cap_and_json_structure_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = codec.encode_policy_request(payload())
    monkeypatch.setattr(codec, "MAX_IMAGE_BYTES", 32)
    with pytest.raises(ValueError, match="image exceeds"):
        codec.encode_policy_request(payload())
    with pytest.raises(ValueError, match="image exceeds"):
        codec.decode_policy_request(encoded)
    monkeypatch.setattr(codec, "MAX_IMAGE_BYTES", 512 * 512 * 3)
    nested: dict[str, Any] = {}
    root = nested
    for _ in range(codec.MAX_JSON_DEPTH + 1):
        child: dict[str, Any] = {}
        nested["nested"] = child
        nested = child
    source = payload()
    source["metadata"] = root
    with pytest.raises(ValueError, match="structural"):
        codec.encode_policy_request(source)
    packet = json.loads(encoded)
    packet["payload"]["metadata"] = root
    with pytest.raises(ValueError, match="structural"):
        codec.decode_policy_request(json.dumps(packet).encode())
    monkeypatch.setattr(codec, "MAX_JSON_NODES", 5)
    with pytest.raises(ValueError, match="structural"):
        codec.decode_policy_request(encoded)

"""Bounded RGB byte transport shared by policy clients and servers."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any

import numpy as np

from .physical_contract import IMAGE_KEYS

CODEC_VERSION = "synria_rgb_uint8_v1"
MAX_IMAGE_SIDE = 512
MAX_IMAGE_BYTES = MAX_IMAGE_SIDE * MAX_IMAGE_SIDE * 3
MAX_TOTAL_IMAGE_BYTES = 2 * MAX_IMAGE_BYTES
MAX_REQUEST_BYTES = 3 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_JSON_NODES = 100_000
MAX_JSON_DEPTH = 32


def _validate_json(value: Any, byte_limit: int) -> None:
    remaining = MAX_JSON_NODES

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            raise ValueError("policy JSON exceeds structural limits")
        if item is None or type(item) is bool:
            return
        if type(item) is str:
            if len(item) > byte_limit:
                raise ValueError("oversized policy JSON string")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("oversized policy JSON integer")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("policy JSON must contain only finite numbers")
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
        elif type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("policy JSON keys must be strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            raise ValueError("unsupported policy JSON value")

    visit(value, 0)


def _encode_json(value: dict[str, Any], limit: int) -> bytes:
    _validate_json(value, limit)
    output = bytearray()
    for piece in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
        if len(piece) > limit - len(output):
            raise ValueError("oversized encoded policy request")
        encoded = piece.encode("utf-8")
        if len(encoded) > limit - len(output):
            raise ValueError("oversized encoded policy request")
        output.extend(encoded)
    return bytes(output)


def _decode_json(data: bytes, limit: int) -> dict[str, Any]:
    if type(data) is not bytes or len(data) > limit:
        raise ValueError("oversized or invalid policy body")

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite policy JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate policy JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("malformed policy JSON") from error
    if type(value) is not dict:
        raise ValueError("policy body must be a JSON object")
    _validate_json(value, limit)
    return value


def _image_size(shape: Any) -> int:
    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(
        type(side) is not int or side <= 0 for side in shape
    ) or shape[2] != 3:
        raise ValueError("images require positive integer HWC RGB shape")
    if shape[0] > MAX_IMAGE_SIDE or shape[1] > MAX_IMAGE_SIDE:
        raise ValueError("image dimensions exceed allocation limits")
    size = shape[0] * shape[1] * shape[2]
    if size > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds allocation limit")
    return int(size)


def _require_images(payload: Any) -> None:
    if type(payload) is not dict or any(type(key) is not str for key in payload) or {
        key for key in payload if key.startswith("observation.images.")
    } != set(IMAGE_KEYS):
        raise ValueError("exactly wrist and front policy images are required")


def encode_policy_request(payload: dict[str, Any]) -> bytes:
    """Serialize two uint8 HWC RGB arrays without pixel lists or image compression."""
    _require_images(payload)
    sizes = []
    for name in IMAGE_KEYS:
        image = payload[name]
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("policy images must be uint8 arrays")
        sizes.append(_image_size(image.shape))
    if sum(sizes) > MAX_TOTAL_IMAGE_BYTES:
        raise ValueError("combined images exceed allocation limit")
    encoded = dict(payload)
    for name in IMAGE_KEYS:
        image = payload[name]
        encoded[name] = {
            "encoding": "base64", "dtype": "uint8", "color_space": "RGB",
            "shape": list(image.shape),
            "data": base64.b64encode(image.tobytes(order="C")).decode("ascii"),
        }
    return _encode_json({"codec": CODEC_VERSION, "payload": encoded}, MAX_REQUEST_BYTES)


def decode_policy_request(data: bytes) -> dict[str, Any]:
    """Validate both image allocations before decoding either base64 byte buffer."""
    packet = _decode_json(data, MAX_REQUEST_BYTES)
    if set(packet) != {"codec", "payload"} or packet["codec"] != CODEC_VERSION:
        raise ValueError("unsupported policy request codec")
    payload = packet["payload"]
    _require_images(payload)
    sizes = []
    for name in IMAGE_KEYS:
        image = payload[name]
        if type(image) is not dict or set(image) != {
            "encoding", "dtype", "color_space", "shape", "data"
        }:
            raise ValueError("invalid policy image packet")
        if (image["encoding"], image["dtype"], image["color_space"]) != (
            "base64", "uint8", "RGB"
        ):
            raise ValueError("policy images require base64 RGB uint8 bytes")
        size = _image_size(image["shape"])
        if type(image["data"]) is not str or len(image["data"]) != 4 * ((size + 2) // 3):
            raise ValueError("image byte length differs from declared shape")
        sizes.append(size)
    if sum(sizes) > MAX_TOTAL_IMAGE_BYTES:
        raise ValueError("combined images exceed allocation limit")
    decoded = dict(payload)
    for name, size in zip(IMAGE_KEYS, sizes, strict=True):
        image = payload[name]
        try:
            raw = base64.b64decode(image["data"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid base64 policy image") from error
        if len(raw) != size:
            raise ValueError("decoded image byte length differs from declared shape")
        decoded[name] = np.frombuffer(raw, dtype=np.uint8).reshape(image["shape"])
    return decoded


def decode_policy_response(data: bytes) -> dict[str, Any]:
    """Decode a bounded finite JSON response, never executable objects."""
    return _decode_json(data, MAX_RESPONSE_BYTES)


def encode_policy_response(payload: dict[str, Any]) -> bytes:
    """Encode a bounded finite response using the same limits as the client."""
    return _encode_json(payload, MAX_RESPONSE_BYTES)

"""Tests for edge_ai.camera_inference — MockFrameSource and CameraInferenceLoop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_ai.camera_inference import (
    CameraInferenceLoop,
    InferenceLoopStats,
    InferenceResult,
    MockFrameSource,
)

# ---------------------------------------------------------------------------
# MockFrameSource
# ---------------------------------------------------------------------------


def test_mock_frame_source_default_shape() -> None:
    src = MockFrameSource()
    ok, frame = src.read()
    assert ok is True
    assert frame.shape == (480, 640, 3)


def test_mock_frame_source_custom_shape() -> None:
    src = MockFrameSource(height=240, width=320, channels=1)
    ok, frame = src.read()
    assert ok is True
    # channels axis is always preserved — single-channel frame is (h, w, 1)
    assert frame.shape == (240, 320, 1)


def test_mock_frame_source_dtype() -> None:
    src = MockFrameSource()
    _, frame = src.read()
    assert frame.dtype == np.uint8


def test_mock_frame_source_exhaustion() -> None:
    """After max_frames reads, source returns (False, empty array)."""
    src = MockFrameSource(max_frames=3)
    results = [src.read() for _ in range(5)]
    # First 3 should succeed
    assert all(r[0] for r in results[:3])
    # 4th and 5th should fail
    assert all(not r[0] for r in results[3:])


def test_mock_frame_source_release_idempotent() -> None:
    src = MockFrameSource()
    src.release()
    src.release()  # second call must not raise


# ---------------------------------------------------------------------------
# CameraInferenceLoop
# ---------------------------------------------------------------------------


def test_camera_loop_runs_expected_frames() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(height=64, width=64),
        inference_fn=lambda f: f,
        max_frames=10,
    )
    results = loop.run()
    assert len(results) == 10


def test_camera_loop_result_type() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=lambda f: f,
        max_frames=5,
    )
    results = loop.run()
    for r in results:
        assert isinstance(r, InferenceResult)


def test_camera_loop_result_latency_positive() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=lambda f: f,
        max_frames=5,
    )
    loop.run()
    for r in loop._results:  # type: ignore[attr-defined]
        assert r.inference_time_ms >= 0.0


def test_camera_loop_stats_none_before_run() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=lambda f: f,
        max_frames=5,
    )
    assert loop.stats() is None


def test_camera_loop_stats_after_run() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=lambda f: f,
        max_frames=20,
    )
    loop.run()
    stats = loop.stats()
    assert stats is not None
    assert isinstance(stats, InferenceLoopStats)
    assert stats.n_frames == 20
    assert stats.mean_fps > 0.0
    assert stats.mean_inference_ms >= 0.0


def test_camera_loop_stats_to_dict() -> None:
    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=lambda f: f,
        max_frames=5,
    )
    loop.run()
    stats = loop.stats()
    assert stats is not None
    d = stats.to_dict()
    assert "n_frames" in d
    assert "mean_fps" in d
    assert "mean_inference_ms" in d


def test_camera_loop_inference_fn_receives_ndarray() -> None:
    """Inference function must receive a numpy array."""
    received: list[object] = []

    def capturing_fn(frame: object) -> object:
        received.append(frame)
        return frame

    loop = CameraInferenceLoop(
        source=MockFrameSource(),
        inference_fn=capturing_fn,
        max_frames=3,
    )
    loop.run()
    assert len(received) == 3
    for frame in received:
        assert isinstance(frame, np.ndarray)


def test_camera_loop_source_exhaustion_stops_loop() -> None:
    """Loop terminates when source signals exhaustion (no more frames)."""
    # Source only produces 4 frames; loop limit is higher — should stop at 4.
    loop = CameraInferenceLoop(
        source=MockFrameSource(max_frames=4),
        inference_fn=lambda f: f,
        max_frames=100,  # loop limit is intentionally larger
    )
    results = loop.run()
    assert len(results) == 4

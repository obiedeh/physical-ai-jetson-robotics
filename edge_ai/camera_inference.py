"""Camera stream inference loop scaffold for Jetson edge deployment.

Provides a hardware-agnostic ``CameraInferenceLoop`` that wires together:
  - a frame source (OpenCV VideoCapture or a mock frame generator)
  - an inference callable (OnnxRunner, TensorRT session, or any callable)
  - a result sink (callback, queue, or log)

The loop is designed for Jetson deployment but works on any machine. When
OpenCV is absent, the loop falls back to a ``MockFrameSource`` that generates
random numpy arrays so downstream code can be tested without a camera.

Typical usage on Jetson::

    from edge_ai.camera_inference import CameraInferenceLoop, InferenceResult

    def my_model(frame: np.ndarray) -> np.ndarray:
        # TensorRT / OnnxRunner call here
        return frame[:, :, 0:1]  # trivial example

    loop = CameraInferenceLoop(
        source=0,                   # /dev/video0
        inference_fn=my_model,
        max_frames=200,
    )
    results = loop.run()
    print(f"Processed {len(results)} frames at {loop.mean_fps():.1f} FPS")
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """One frame's inference output and timing metadata."""

    frame_index: int
    capture_time_s: float        # wall-clock time when the frame was captured
    inference_time_ms: float     # time spent in inference_fn (milliseconds)
    output: np.ndarray           # raw model output tensor


@dataclass
class InferenceLoopStats:
    """Aggregate statistics after a camera inference loop run."""

    n_frames: int
    total_time_s: float
    mean_inference_ms: float
    p95_inference_ms: float
    mean_fps: float
    source_type: str  # "camera", "mock", or "file"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "total_time_s": round(self.total_time_s, 3),
            "mean_inference_ms": round(self.mean_inference_ms, 3),
            "p95_inference_ms": round(self.p95_inference_ms, 3),
            "mean_fps": round(self.mean_fps, 2),
            "source_type": self.source_type,
        }


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


class MockFrameSource:
    """Generates random uint8 frames for testing without a camera.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        channels: Number of colour channels (3 = BGR, 1 = grayscale).
            When ``channels == 1``, the returned frame has shape ``(h, w, 1)``
            (the channel axis is preserved for consistency).
        seed: Random seed for reproducibility.
        max_frames: Stop after this many reads (0 = unlimited). Once exhausted,
            ``read()`` returns ``(False, empty_array)``.
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        channels: int = 3,
        seed: int = 42,
        max_frames: int = 0,
    ) -> None:
        self._shape: tuple[int, ...] = (height, width, channels)
        self._rng = np.random.default_rng(seed)
        self._open = True
        self._max_frames = max_frames
        self._frame_count = 0

    def is_open(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray]:
        """Return (success, frame) — same interface as cv2.VideoCapture.read().

        Returns ``(False, np.empty((0,)))`` when the source is exhausted.
        """
        if self._max_frames > 0 and self._frame_count >= self._max_frames:
            return False, np.empty((0,), dtype=np.uint8)
        frame = self._rng.integers(0, 256, size=self._shape, dtype=np.uint8)
        self._frame_count += 1
        return True, frame

    def release(self) -> None:
        self._open = False


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


class CameraInferenceLoop:
    """Drive an inference callable over a camera stream or mock source.

    Args:
        source: Camera device index (int), video file path (str/Path), or a
            ``MockFrameSource`` instance. When ``source`` is an int or str and
            OpenCV is not installed, automatically falls back to a mock source.
        inference_fn: Callable that takes a ``np.ndarray`` frame and returns
            a ``np.ndarray`` result.
        max_frames: Stop after this many frames (0 = run until source ends).
        frame_height: Target height for frame resize (0 = no resize).
        frame_width: Target width for frame resize (0 = no resize).
    """

    def __init__(
        self,
        source: int | str | MockFrameSource = 0,
        inference_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        max_frames: int = 0,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> None:
        self._inference_fn = inference_fn or (lambda x: x)
        self._max_frames = max_frames
        self._resize = (frame_height > 0 and frame_width > 0)
        self._target_h = frame_height
        self._target_w = frame_width
        self._results: list[InferenceResult] = []
        self._stats: InferenceLoopStats | None = None

        if isinstance(source, MockFrameSource):
            self._cap = source
            self._source_type = "mock"
        elif not _CV2_AVAILABLE:
            self._cap = MockFrameSource()
            self._source_type = "mock"
        else:
            self._cap = cv2.VideoCapture(source)
            self._source_type = "camera" if isinstance(source, int) else "file"

    @property
    def source_type(self) -> str:
        return self._source_type

    def run(self) -> list[InferenceResult]:
        """Run the inference loop and return per-frame results.

        Captures frames until ``max_frames`` is reached or the source ends.
        After completion, call ``stats()`` to retrieve aggregate metrics.
        """
        self._results = []
        inference_times: list[float] = []
        t_start = time.perf_counter()
        frame_idx = 0

        while True:
            if self._max_frames > 0 and frame_idx >= self._max_frames:
                break

            t_capture = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                break

            if self._resize and _CV2_AVAILABLE and cv2 is not None:
                frame = cv2.resize(frame, (self._target_w, self._target_h))

            t0 = time.perf_counter()
            output = self._inference_fn(frame)
            t1 = time.perf_counter()
            inf_ms = (t1 - t0) * 1000.0

            self._results.append(
                InferenceResult(
                    frame_index=frame_idx,
                    capture_time_s=round(t_capture - t_start, 5),
                    inference_time_ms=round(inf_ms, 3),
                    output=output,
                )
            )
            inference_times.append(inf_ms)
            frame_idx += 1

        self._cap.release()

        total_s = time.perf_counter() - t_start
        mean_inf = sum(inference_times) / len(inference_times) if inference_times else 0.0
        sorted_inf = sorted(inference_times)
        p95_idx = max(0, int(len(sorted_inf) * 0.95) - 1)
        p95_inf = sorted_inf[p95_idx] if sorted_inf else 0.0
        mean_fps = len(self._results) / total_s if total_s > 0 else 0.0

        self._stats = InferenceLoopStats(
            n_frames=len(self._results),
            total_time_s=round(total_s, 3),
            mean_inference_ms=round(mean_inf, 3),
            p95_inference_ms=round(p95_inf, 3),
            mean_fps=round(mean_fps, 2),
            source_type=self._source_type,
        )
        return list(self._results)

    def stats(self) -> InferenceLoopStats | None:
        """Return aggregate statistics from the last ``run()`` call."""
        return self._stats

    def mean_fps(self) -> float:
        """Convenience accessor for mean throughput FPS after ``run()``."""
        return self._stats.mean_fps if self._stats else 0.0

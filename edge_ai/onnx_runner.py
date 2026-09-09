"""ONNX inference runner with a consistent public interface.

On machines with ``onnxruntime`` installed (Linux RTX workstation, Jetson),
this wraps a real ``InferenceSession``. On machines without it (Windows dev,
CI), the runner falls back to a **mock mode** that returns zero-filled numpy
arrays so tests and downstream code are always exercisable.

Usage::

    runner = OnnxRunner("path/to/model.onnx")
    if runner.is_mock:
        print("Running in mock mode — no onnxruntime installed.")

    info = runner.info()
    outputs = runner.run({"input_0": np.zeros((1, 3, 224, 224), dtype=np.float32)})
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort  # type: ignore[import-not-found]

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    ort = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OnnxModelInfo:
    """Metadata about a loaded ONNX model."""

    model_path: str
    input_names: list[str]
    output_names: list[str]
    input_shapes: list[list[int]]
    backend: str  # "onnxruntime" or "mock"

    def is_mock(self) -> bool:
        return self.backend == "mock"


class OnnxRunner:
    """Thin wrapper around an onnxruntime InferenceSession.

    Falls back to mock mode when ``onnxruntime`` is not installed.
    Mock mode returns zero-filled float32 arrays so the public interface
    is always fully exercisable without hardware or a real model file.

    Args:
        model_path: Path to an ``.onnx`` file. In mock mode this path is
            stored but never opened.
        providers: ORT execution providers in priority order. Defaults to
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        force_mock: bool = False,
    ) -> None:
        self._model_path = str(model_path)
        self._mock = force_mock or not _ORT_AVAILABLE

        if self._mock and not force_mock:
            # onnxruntime absent: the old behaviour fell back to a mock
            # SILENTLY, which the 2026-08-19 portfolio audit flagged as a
            # silent-CPU-fallback-class defect. Mock now requires explicit
            # opt-in (force_mock=True or PHYSICAL_AI_ALLOW_MOCK=1).
            import os
            if os.getenv("PHYSICAL_AI_ALLOW_MOCK") != "1":
                raise RuntimeError(
                    "onnxruntime is not installed and mock mode was not "
                    "explicitly requested. Install onnxruntime, or pass "
                    "force_mock=True / set PHYSICAL_AI_ALLOW_MOCK=1 to "
                    "run with the mock (never for benchmarks).")

        if self._mock:
            self._session = None
            self._input_names: list[str] = ["input_0"]
            self._output_names: list[str] = ["output_0"]
            self._input_shapes: list[list[int]] = [[-1, -1]]
            return

        _providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self._model_path, providers=_providers)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._input_shapes = [
            [d if isinstance(d, int) else -1 for d in inp.shape]
            for inp in self._session.get_inputs()
        ]

    @property
    def is_mock(self) -> bool:
        """True when onnxruntime is not installed and mock mode is active."""
        return self._mock

    def info(self) -> OnnxModelInfo:
        """Return metadata about the loaded model."""
        return OnnxModelInfo(
            model_path=self._model_path,
            input_names=list(self._input_names),
            output_names=list(self._output_names),
            input_shapes=[list(s) for s in self._input_shapes],
            backend="mock" if self._mock else "onnxruntime",
        )

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on a dict of named input arrays.

        In mock mode, returns zero-filled float32 arrays with shape
        ``(batch_size, 1)`` for each output name. This ensures the public API
        is always callable without a real model or GPU.

        Args:
            inputs: Mapping of input name → numpy array.

        Returns:
            Mapping of output name → numpy array.
        """
        if self._mock:
            first = next(iter(inputs.values()), np.zeros((1,), dtype=np.float32))
            batch = first.shape[0] if first.ndim >= 1 else 1
            return {
                name: np.zeros((batch, 1), dtype=np.float32)
                for name in self._output_names
            }

        assert self._session is not None
        outputs = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, outputs, strict=False))

"""Inference latency and throughput benchmark runner.

Provides a hardware-agnostic harness that times any callable inference
function and produces structured ``InferenceBenchmarkResult`` records suitable
for evidence reports under ``reports/edge_ai/``.

Hardware-specific metrics (CUDA memory, TensorRT layer times, Jetson power draw)
are added by the Jetson deployment path (``scripts/jetson/``) once the device
is live. This module runs anywhere — no GPU required.

Example::

    import numpy as np
    from edge_ai.benchmark import run_latency_benchmark, write_benchmark_report

    def my_model(x: np.ndarray) -> np.ndarray:
        return x * 2.0

    result = run_latency_benchmark(
        inference_fn=my_model,
        sample_input=np.zeros((1, 3, 224, 224), dtype=np.float32),
        n_warmup=10,
        n_runs=200,
        workload_name="resnet18-mock",
        backend="mock",
        device="cpu",
        input_shape=[1, 3, 224, 224],
    )
    print(result.to_json())
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceLatencyStats:
    """Per-run latency statistics in milliseconds."""

    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_runs: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class InferenceBenchmarkResult:
    """Benchmark result for one model / workload combination.

    Attributes:
        workload_name: Human-readable name for the benchmark (e.g. "yolov8n-fp16").
        backend: Inference backend — "onnx", "tensorrt", "torch", or "mock".
        device: Target device — "cpu", "cuda", "jetson-orin", "jetson-thor".
        input_shape: Shape of the model's primary input tensor.
        batch_size: Batch size used during benchmarking.
        latency: Per-run latency statistics.
        throughput_fps: Frames / inferences per second at the measured mean latency.
        model_size_mb: On-disk model size in megabytes (None if not measured).
        notes: Free-text notes on environment, optimisation flags, etc.
    """

    workload_name: str
    backend: str
    device: str
    input_shape: list[int]
    batch_size: int
    latency: InferenceLatencyStats
    throughput_fps: float
    model_size_mb: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "workload_name": self.workload_name,
            "backend": self.backend,
            "device": self.device,
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "latency": self.latency.to_dict(),
            "throughput_fps": self.throughput_fps,
            "model_size_mb": self.model_size_mb,
            "notes": self.notes,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the pct-th percentile of a pre-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = max(0, min(int(n * pct / 100.0), n - 1))
    return sorted_values[idx]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_latency_benchmark(
    inference_fn: Callable[[Any], Any],
    sample_input: Any,
    n_warmup: int = 5,
    n_runs: int = 100,
    workload_name: str = "unnamed",
    backend: str = "mock",
    device: str = "cpu",
    input_shape: list[int] | None = None,
    batch_size: int = 1,
    model_size_mb: float | None = None,
    notes: str = "",
) -> InferenceBenchmarkResult:
    """Time ``inference_fn(sample_input)`` and return structured benchmark stats.

    Args:
        inference_fn: The callable to benchmark — called with ``sample_input``.
        sample_input: Input passed to ``inference_fn`` on every call.
        n_warmup: Number of warmup calls before timing begins.
        n_runs: Number of timed calls.
        workload_name: Label for the benchmark.
        backend: Inference backend identifier.
        device: Device identifier.
        input_shape: Primary input tensor shape (for metadata only).
        batch_size: Batch size (for metadata only).
        model_size_mb: On-disk model size in MB (for metadata only).
        notes: Free-text environment notes.

    Returns:
        ``InferenceBenchmarkResult`` with full latency statistics.
    """
    for _ in range(n_warmup):
        inference_fn(sample_input)

    latencies_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inference_fn(sample_input)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    throughput_fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    stats = InferenceLatencyStats(
        min_ms=round(latencies_ms[0], 4) if latencies_ms else 0.0,
        max_ms=round(latencies_ms[-1], 4) if latencies_ms else 0.0,
        mean_ms=round(mean_ms, 4),
        p50_ms=round(_percentile(latencies_ms, 50), 4),
        p95_ms=round(_percentile(latencies_ms, 95), 4),
        p99_ms=round(_percentile(latencies_ms, 99), 4),
        n_runs=n_runs,
    )

    return InferenceBenchmarkResult(
        workload_name=workload_name,
        backend=backend,
        device=device,
        input_shape=input_shape or [],
        batch_size=batch_size,
        latency=stats,
        throughput_fps=round(throughput_fps, 2),
        model_size_mb=model_size_mb,
        notes=notes,
    )


def write_benchmark_report(
    result: InferenceBenchmarkResult,
    output: Path,
) -> Path:
    """Serialize a benchmark result to a JSON evidence file.

    Creates parent directories as needed. Returns the written path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.to_json() + "\n", encoding="utf-8")
    return output

"""Edge AI inference and benchmarking scaffolding for Jetson deployment.

Public surface:
    benchmark        — latency/throughput benchmark runner
    onnx_runner      — ONNX inference wrapper (hardware-gated, falls back to mock)
    camera_inference — camera frame loop with inference timing stats
"""

from edge_ai.benchmark import (
    InferenceBenchmarkResult,
    InferenceLatencyStats,
    run_latency_benchmark,
    write_benchmark_report,
)
from edge_ai.camera_inference import (
    CameraInferenceLoop,
    InferenceLoopStats,
    InferenceResult,
    MockFrameSource,
)
from edge_ai.onnx_runner import OnnxModelInfo, OnnxRunner

__all__ = [
    "InferenceBenchmarkResult",
    "InferenceLatencyStats",
    "run_latency_benchmark",
    "write_benchmark_report",
    "OnnxModelInfo",
    "OnnxRunner",
    "CameraInferenceLoop",
    "InferenceLoopStats",
    "InferenceResult",
    "MockFrameSource",
]

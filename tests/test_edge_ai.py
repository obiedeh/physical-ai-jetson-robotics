"""Tests for the edge_ai benchmark and ONNX runner modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_ai.benchmark import (
    InferenceBenchmarkResult,
    run_latency_benchmark,
    write_benchmark_report,
)
from edge_ai.onnx_runner import OnnxModelInfo, OnnxRunner

# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def test_benchmark_returns_result_with_correct_shape() -> None:
    sample = np.zeros((1, 3, 224, 224), dtype=np.float32)
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=sample,
        n_warmup=2,
        n_runs=20,
        workload_name="identity-224",
        backend="mock",
        device="cpu",
        input_shape=[1, 3, 224, 224],
        batch_size=1,
    )
    assert isinstance(result, InferenceBenchmarkResult)
    assert result.workload_name == "identity-224"
    assert result.backend == "mock"
    assert result.latency.n_runs == 20


def test_benchmark_latency_stats_are_ordered() -> None:
    sample = np.zeros((1,), dtype=np.float32)
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=sample,
        n_warmup=2,
        n_runs=50,
    )
    s = result.latency
    assert s.min_ms <= s.p50_ms <= s.p95_ms <= s.p99_ms <= s.max_ms


def test_benchmark_throughput_is_positive() -> None:
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=0,
        n_warmup=1,
        n_runs=10,
    )
    assert result.throughput_fps > 0.0


def test_benchmark_model_size_optional() -> None:
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=0,
        n_warmup=1,
        n_runs=5,
        model_size_mb=42.5,
    )
    assert result.model_size_mb == pytest.approx(42.5)


def test_benchmark_to_json_round_trips() -> None:
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=0,
        n_warmup=1,
        n_runs=5,
        workload_name="round-trip-test",
    )
    parsed = json.loads(result.to_json())
    assert parsed["workload_name"] == "round-trip-test"
    assert "mean_ms" in parsed["latency"]
    assert "n_runs" in parsed["latency"]


def test_write_benchmark_report_creates_file(tmp_path: Path) -> None:
    result = run_latency_benchmark(
        inference_fn=_identity,
        sample_input=np.zeros((1,), dtype=np.float32),
        n_warmup=1,
        n_runs=5,
        workload_name="file-write-test",
    )
    output = tmp_path / "edge_ai" / "report.json"
    written = write_benchmark_report(result, output)

    assert written == output
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["workload_name"] == "file-write-test"


# ---------------------------------------------------------------------------
# ONNX runner tests — use force_mock=True so these run on any machine
# regardless of whether onnxruntime is installed or a real model file exists.
# ---------------------------------------------------------------------------


def test_onnx_runner_mock_is_always_importable() -> None:
    runner = OnnxRunner("non_existent_model.onnx", force_mock=True)
    assert isinstance(runner, OnnxRunner)
    assert runner.is_mock is True


def test_onnx_runner_mock_info_returns_model_info() -> None:
    runner = OnnxRunner("non_existent_model.onnx", force_mock=True)
    info = runner.info()
    assert isinstance(info, OnnxModelInfo)
    assert info.model_path == "non_existent_model.onnx"
    assert info.input_names
    assert info.output_names


def test_onnx_runner_mock_run_returns_dict_of_arrays() -> None:
    runner = OnnxRunner("dummy.onnx", force_mock=True)
    outputs = runner.run({"input_0": np.zeros((2, 4), dtype=np.float32)})
    assert isinstance(outputs, dict)
    assert len(outputs) >= 1
    for v in outputs.values():
        assert isinstance(v, np.ndarray)
        assert v.shape[0] == 2  # batch dim preserved


def test_onnx_runner_mock_info_backend_is_mock() -> None:
    runner = OnnxRunner("dummy.onnx", force_mock=True)
    info = runner.info()
    assert info.backend == "mock"


def test_onnx_runner_is_mock_true_when_forced() -> None:
    runner = OnnxRunner("dummy.onnx", force_mock=True)
    assert runner.is_mock is True

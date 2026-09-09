# Edge AI Evidence Reports

This directory holds inference benchmark and camera loop evidence captured from
Jetson hardware and the RTX 5090 workstation.

## Report Types

| File pattern | Contents |
|---|---|
| `mock_benchmark.json` | CLI mock run — baseline sanity check, no hardware required |
| `jetson_orin_*.json` | Jetson Orin NX 8GB inference benchmarks (latency, FPS, memory) |
| `jetson_thor_*.json` | Jetson AGX Thor inference benchmarks |
| `rtx_5090_*.json` | RTX 5090 workstation inference benchmarks |

## Schema

Each report is an `InferenceBenchmarkResult` JSON:

```json
{
  "workload_name": "yolov8n-fp16",
  "backend": "tensorrt",
  "device": "jetson-orin",
  "input_shape": [1, 3, 640, 640],
  "batch_size": 1,
  "latency": {
    "min_ms": 4.2,
    "mean_ms": 5.1,
    "p95_ms": 6.3,
    "p99_ms": 7.1,
    "max_ms": 9.8,
    "n_runs": 500
  },
  "throughput_fps": 196.0,
  "model_size_mb": 6.3,
  "notes": "TensorRT FP16, Jetson Orin NX 8GB, JetPack 6.x"
}
```

## Run CLI Demo

```bash
physical-ai-lab edge-ai-benchmark --n-runs 200 --frames 100
```

## Hardware Evidence Gate

Reports in this directory are **planned evidence** until Jetson hardware is live
and matching logs, screenshots, and thermal readings are committed alongside them.

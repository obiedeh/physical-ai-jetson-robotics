# Production Roadmap

## MVP

- Pydantic schemas for telemetry, features, detection results, and alerts.
- CSV replay for Edge-IIoT style datasets.
- Sliding feature windows.
- Threshold baseline detector.
- Pipeline metrics.
- CLI commands for validation, replay, and demo runs.
- Tests for schema, replay, windows, and pipeline smoke behavior.

## Near-Term

- Add JSONL replay for normalized telemetry.
- Implement Zeek `conn.log` and Suricata EVE JSON ingestion.
- Add alert sinks for JSONL, MQTT, and SIEM forwarding.
- Add benchmark reports for latency, throughput, and memory use.
- Add model artifact loading with explicit feature schema checks.

## Advanced

- Optional ONNX or TensorRT model runner.
- Drift detection for source mix and packet-size distributions.
- Configurable time-based windows alongside count-based windows.
- Backpressure controls for live streams.
- Signed model/config bundles for edge deployment.

## Jetson Deployment

- Build a minimal container image.
- Validate on Jetson Orin Nano and Orin NX.
- Add GPU-optional inference paths while keeping CPU fallback.
- Track power, CPU, memory, and latency metrics.
- Support systemd or containerized service deployment.

## Private 5G / AI-RAN Extension

- Add telemetry adapters for private 5G lab logs and brokered RAN telemetry.
- Track cell, slice, device, and application metadata in `TelemetryEvent.metadata`.
- Build per-slice feature windows.
- Correlate network telemetry with edge application health.
- Keep attack traffic handling limited to defensive replay, simulation, and IDS-log ingestion.


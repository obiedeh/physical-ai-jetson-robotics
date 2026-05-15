# Jetson Edge AI Security

Jetson Edge AI Security is a production-oriented defensive telemetry runtime for edge IDS research and deployment. It converts replayed or observed security telemetry into normalized events, rolling features, baseline detections, alerts, and benchmark-ready metrics.

The project is edge-native because the runtime is built around streaming sources, small memory windows, conservative baseline detection, and simple dependencies that can run on Jetson-class devices before heavier model runners are introduced.

## Current MVP

- Pluggable `TrafficSource` API with context-manager lifecycle.
- `TelemetryEvent` and `Alert` schemas using Pydantic.
- CSV replay for Edge-IIoT style datasets and similar IDS exports.
- Sliding window feature extraction over an iterable event stream.
- Rule-based baseline detector with optional `sklearn` IsolationForest.
- Pipeline runner that tracks events, windows, detections, and emitted alerts.
- Typer CLI for config validation, CSV replay, and a built-in demo.

## Telemetry Source Strategy

The runtime normalizes every source into `TelemetryEvent` before feature extraction. That keeps ML, detection, alerting, and reporting independent from the source adapter.

Supported now:

- CSV replay from defensive datasets and lab captures.

Planned adapters:

- PCAP replay for packet captures.
- Live capture for defensive interface monitoring.
- MQTT telemetry ingestion.
- Zeek log ingestion.
- Suricata EVE JSON ingestion.

Attack traffic support is intentionally limited to controlled defensive lab telemetry, replay, simulation, and IDS-log ingestion.

## What Is Intentionally Not Included

- Offensive malware generation or exploitation tooling.
- Autonomous attack execution.
- Notebook-only runtime paths.
- Deep learning as the default detector.
- Hardcoded `/data` paths or hidden global state.

Existing notebooks and reports should remain reference material. Reusable academic ideas are preserved in runtime form through CSV column mapping, label normalization, timestamp parsing, temporal aggregation, leakage-aware feature windows, and attack-count forecasting scaffolding.

## Install

```bash
cd projects/jetson-edge-ai-security
python -m pip install -e ".[dev]"
```

Optional ML detector support:

```bash
python -m pip install -e ".[ml]"
```

## Run Tests

```bash
cd projects/jetson-edge-ai-security
python -m pytest
```

## Validate Config

```bash
edge-security validate-config --config configs/default.yaml
```

## Run CSV Replay

```bash
edge-security replay-csv --path data/sample.csv --limit 1000
```

For malformed-row enforcement during data quality checks:

```bash
edge-security replay-csv --path data/sample.csv --strict
```

## Public Dataset Mode

You can list known public defensive datasets:

```bash
edge-security list-datasets
```

For datasets with an allowlisted direct archive URL, the runtime can download, extract, find a CSV, and replay it:

```bash
edge-security fetch-dataset wustl-iiot-2021
edge-security replay-dataset wustl-iiot-2021 --limit 1000
```

The first auto-download target is `wustl-iiot-2021`, because its official project page exposes a direct ZIP archive. Other noteworthy datasets, such as CICIoT2023, ToN_IoT, BoT-IoT, and Edge-IIoTset, are listed for awareness but may require manual download through forms or SharePoint. For those, download the dataset yourself and use `replay-csv --path <local.csv>`.

## Run Demo

```bash
edge-security run-demo
```

## Future Jetson Deployment Path

The runtime is designed so Jetson deployment can add hardware-specific packaging without changing the detection pipeline:

- Package the project into a small Python environment or container.
- Mount read-only configs and source-specific credentials.
- Run CSV replay for lab validation.
- Add live capture or broker-based telemetry adapters.
- Export alerts to a local file, MQTT topic, SIEM forwarder, or edge dashboard.
- Benchmark CPU, memory, latency, and alert throughput on Jetson Orin-class hardware.

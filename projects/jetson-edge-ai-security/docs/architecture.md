# Architecture

```text
Traffic Source -> Event Normalization -> Feature Windows -> Detection -> Alerts
```

## Traffic Source

`TrafficSource` adapters isolate source-specific details. A source owns connection, file, or replay state and yields normalized `TelemetryEvent` records.

Current source:

- `CsvReplaySource`

Future sources:

- `PcapReplaySource`
- `LiveCaptureSource`
- `MqttTelemetrySource`
- `ZeekLogSource`
- `SuricataEveSource`

## Event Normalization

All sources emit the same Pydantic `TelemetryEvent` shape:

- timestamp
- source and destination IPs
- source and destination ports
- protocol
- packet size
- TCP flags
- ICMP type
- flow ID
- source type
- optional attack label and attack type
- metadata

Labels are optional because production traffic normally does not arrive with ground-truth attack labels.

## Feature Windows

`SlidingWindowExtractor` aggregates a stream of events into count-based windows. Features include packet counts, packet size statistics, protocol counts, TCP flag counts, attack counts when labels exist, event rate, and unique endpoint counts.

## Detection

The default detector is conservative and production-safe:

- threshold rules for clear operational signals
- optional IsolationForest if `scikit-learn` is installed and explicitly enabled
- graceful fallback when optional ML dependencies are missing

Deep learning is not the default runtime path.

## Alerts

`AlertBuilder` converts anomalous detection results into structured `Alert` records with severity, title, description, feature context, recommended action, and metadata.

## Runtime

`PipelineRunner` ties the stages together:

```text
TrafficSource.events()
  -> SlidingWindowExtractor.windows()
  -> detector.detect()
  -> AlertBuilder.from_detection()
```

The runner also tracks runtime metrics for events, windows, detections, alerts, and duration.


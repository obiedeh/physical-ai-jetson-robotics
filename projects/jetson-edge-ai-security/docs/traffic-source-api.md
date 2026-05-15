# Traffic Source API

`TrafficSource` is the stable ingestion contract for the runtime.

```python
class TrafficSource(ABC):
    name: str

    def open(self) -> None: ...
    def events(self) -> Iterator[TelemetryEvent]: ...
    def close(self) -> None: ...
```

Sources are also context managers:

```python
with CsvReplaySource("data/sample.csv", limit=1000) as source:
    for event in source.events():
        ...
```

## Source Responsibilities

A source should:

- Own file handles, network clients, and capture sessions.
- Normalize records into `TelemetryEvent`.
- Put source-specific raw fields in `metadata`.
- Tolerate optional missing fields when safe.
- Use clear errors for unrecoverable source failures.

A source should not:

- Run detection logic.
- Generate alerts directly.
- Depend on notebook state.
- Require labels for live production traffic.
- Include offensive malware behavior or attack execution.

## CSV Replay

`CsvReplaySource` maps Edge-IIoT style columns and common IDS aliases into normalized events. It supports:

- `limit`
- `replay_delay_seconds`
- strict or forgiving malformed-row behavior
- unmapped column preservation in event metadata

## Adding a Source

1. Create a module in `jetson_edge_ai_security.sources`.
2. Subclass `TrafficSource`.
3. Implement `open`, `events`, and `close`.
4. Normalize every output into `TelemetryEvent`.
5. Add tests for lifecycle, mapping, malformed data, and stream behavior.
6. Export the source from `sources/__init__.py`.


"""Future Suricata EVE JSON source adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource


class SuricataEveSource(TrafficSource):
    """Read Suricata EVE JSON alerts/flows and normalize them into TelemetryEvent."""

    name = "suricata-eve"

    def __init__(self, path: str | Path, *, follow: bool = False, event_types: set[str] | None = None) -> None:
        self.path = Path(path)
        self.follow = follow
        self.event_types = event_types

    def open(self) -> None:
        raise NotImplementedError("Suricata EVE ingestion is planned for alert and flow records.")

    def events(self) -> Iterator[TelemetryEvent]:
        raise NotImplementedError("Suricata EVE normalization is not implemented yet.")

    def close(self) -> None:
        return None


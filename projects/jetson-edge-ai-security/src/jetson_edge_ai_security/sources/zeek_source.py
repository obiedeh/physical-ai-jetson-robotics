"""Future Zeek log source adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource


class ZeekLogSource(TrafficSource):
    """Read Zeek conn.log style files and normalize records into TelemetryEvent."""

    name = "zeek-log"

    def __init__(self, path: str | Path, *, follow: bool = False) -> None:
        self.path = Path(path)
        self.follow = follow

    def open(self) -> None:
        raise NotImplementedError("Zeek log ingestion is planned for conn.log and JSON logs.")

    def events(self) -> Iterator[TelemetryEvent]:
        raise NotImplementedError("Zeek log normalization is not implemented yet.")

    def close(self) -> None:
        return None


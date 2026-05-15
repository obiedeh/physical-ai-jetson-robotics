"""Future live defensive packet capture adapter."""

from __future__ import annotations

from collections.abc import Iterator

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource


class LiveCaptureSource(TrafficSource):
    """Capture live traffic from an interface for defensive monitoring."""

    name = "live-capture"

    def __init__(self, interface: str, *, bpf_filter: str | None = None, snaplen: int = 1518) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.snaplen = snaplen

    def open(self) -> None:
        raise NotImplementedError("Live capture is planned for defensive interface monitoring only.")

    def events(self) -> Iterator[TelemetryEvent]:
        raise NotImplementedError("Live capture event normalization is not implemented yet.")

    def close(self) -> None:
        return None


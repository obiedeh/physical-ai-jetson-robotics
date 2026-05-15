"""Future PCAP replay adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource


class PcapReplaySource(TrafficSource):
    """Replay packet captures into normalized events using a future parser backend."""

    name = "pcap-replay"

    def __init__(self, path: str | Path, *, bpf_filter: str | None = None) -> None:
        self.path = Path(path)
        self.bpf_filter = bpf_filter

    def open(self) -> None:
        raise NotImplementedError("PCAP replay requires a packet parser such as scapy or pyshark.")

    def events(self) -> Iterator[TelemetryEvent]:
        raise NotImplementedError("PCAP replay event normalization is not implemented yet.")

    def close(self) -> None:
        return None


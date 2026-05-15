"""Rolling and sliding feature windows for telemetry streams."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Iterator
from datetime import datetime

from jetson_edge_ai_security.schemas import FeatureWindow, TelemetryEvent


class SlidingWindowExtractor:
    """Create count-based sliding feature windows from an event stream."""

    def __init__(self, *, window_size: int = 50, step: int = 10) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if step < 1:
            raise ValueError("step must be at least 1")
        self.window_size = window_size
        self.step = step

    def windows(self, events: Iterable[TelemetryEvent]) -> Iterator[FeatureWindow]:
        buffer: deque[TelemetryEvent] = deque(maxlen=self.window_size)
        since_last_emit = 0
        for event in events:
            buffer.append(event)
            since_last_emit += 1
            if len(buffer) < self.window_size:
                continue
            if since_last_emit >= self.step:
                since_last_emit = 0
                yield build_feature_window(list(buffer))


def build_feature_window(events: list[TelemetryEvent]) -> FeatureWindow:
    """Build one feature window from normalized events."""

    if not events:
        raise ValueError("Cannot build a feature window from no events")

    timestamps = [event.timestamp for event in events]
    start = min(timestamps)
    end = max(timestamps)
    packet_sizes = [event.packet_size or 0 for event in events]
    duration = max((end - start).total_seconds(), 1e-6)

    protocol_counts = Counter(event.protocol for event in events)
    tcp_flag_counts = Counter(event.tcp_flags for event in events if event.tcp_flags)
    attack_count = sum(1 for event in events if event.attack_label is True)

    return FeatureWindow(
        window_start=start,
        window_end=end,
        packet_count=len(events),
        mean_packet_size=sum(packet_sizes) / len(packet_sizes),
        max_packet_size=max(packet_sizes),
        protocol_counts=dict(protocol_counts),
        tcp_flag_counts=dict(tcp_flag_counts),
        attack_count=attack_count,
        event_rate=len(events) / duration,
        unique_source_ip_count=len({event.source_ip for event in events if event.source_ip}),
        unique_dest_ip_count=len({event.dest_ip for event in events if event.dest_ip}),
        metadata={"window_size": len(events)},
    )


def window_stream(
    events: Iterable[TelemetryEvent],
    *,
    window_size: int = 50,
    step: int = 10,
) -> Iterator[FeatureWindow]:
    """Convenience wrapper for streaming window extraction."""

    yield from SlidingWindowExtractor(window_size=window_size, step=step).windows(events)


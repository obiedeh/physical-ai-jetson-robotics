from datetime import UTC, datetime, timedelta

from jetson_edge_ai_security.features import SlidingWindowExtractor
from jetson_edge_ai_security.schemas import TelemetryEvent


def test_sliding_window_feature_creation() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        TelemetryEvent(
            timestamp=start + timedelta(seconds=index),
            source_ip=f"10.0.0.{index}",
            dest_ip="10.0.1.1",
            protocol="TCP" if index % 2 else "UDP",
            packet_size=100 + index,
            tcp_flags="S" if index % 2 else None,
            attack_label=index == 2,
        )
        for index in range(5)
    ]

    windows = list(SlidingWindowExtractor(window_size=3, step=1).windows(events))

    assert len(windows) == 3
    assert windows[0].packet_count == 3
    assert windows[0].max_packet_size == 102
    assert windows[0].protocol_counts["UDP"] == 2
    assert windows[0].attack_count == 1
    assert windows[0].unique_source_ip_count == 3


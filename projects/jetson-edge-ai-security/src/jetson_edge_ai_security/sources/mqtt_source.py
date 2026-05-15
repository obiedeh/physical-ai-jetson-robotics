"""Future MQTT telemetry source adapter."""

from __future__ import annotations

from collections.abc import Iterator

from jetson_edge_ai_security.schemas import TelemetryEvent
from jetson_edge_ai_security.sources.base import TrafficSource


class MqttTelemetrySource(TrafficSource):
    """Read normalized or near-normalized telemetry from MQTT topics."""

    name = "mqtt-telemetry"

    def __init__(self, broker_url: str, topic: str, *, client_id: str = "edge-security") -> None:
        self.broker_url = broker_url
        self.topic = topic
        self.client_id = client_id

    def open(self) -> None:
        raise NotImplementedError("MQTT ingestion requires an MQTT client backend such as paho-mqtt.")

    def events(self) -> Iterator[TelemetryEvent]:
        raise NotImplementedError("MQTT payload normalization is not implemented yet.")

    def close(self) -> None:
        return None


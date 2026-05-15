"""Streaming telemetry pipeline."""

from __future__ import annotations

from collections.abc import Iterator

from jetson_edge_ai_security.alerts import AlertBuilder
from jetson_edge_ai_security.detection import BaselineDetector, Detector
from jetson_edge_ai_security.features import SlidingWindowExtractor
from jetson_edge_ai_security.runtime.metrics import RuntimeMetrics
from jetson_edge_ai_security.schemas import Alert, TelemetryEvent
from jetson_edge_ai_security.sources import TrafficSource


class PipelineRunner:
    """Run source ingestion, feature extraction, detection, and alerting."""

    def __init__(
        self,
        source: TrafficSource,
        *,
        window_size: int = 50,
        step: int = 10,
        detector: Detector | None = None,
        alert_builder: AlertBuilder | None = None,
    ) -> None:
        self.source = source
        self.window_extractor = SlidingWindowExtractor(window_size=window_size, step=step)
        self.detector = detector or BaselineDetector()
        self.alert_builder = alert_builder or AlertBuilder()
        self.metrics = RuntimeMetrics()

    def run(self, *, max_alerts: int | None = None) -> list[Alert]:
        alerts = list(self.stream_alerts(max_alerts=max_alerts))
        self.metrics.finish()
        return alerts

    def stream_alerts(self, *, max_alerts: int | None = None) -> Iterator[Alert]:
        event_stream = self._count_events(self.source.events())
        for window in self.window_extractor.windows(event_stream):
            self.metrics.windows_seen += 1
            result = self.detector.detect(window)
            if result.is_anomaly:
                self.metrics.detections_seen += 1
            alert = self.alert_builder.from_detection(result)
            if alert is None:
                continue
            self.metrics.alerts_emitted += 1
            yield alert
            if max_alerts is not None and self.metrics.alerts_emitted >= max_alerts:
                break

    def _count_events(self, events: Iterator[TelemetryEvent]) -> Iterator[TelemetryEvent]:
        for event in events:
            self.metrics.events_seen += 1
            yield event


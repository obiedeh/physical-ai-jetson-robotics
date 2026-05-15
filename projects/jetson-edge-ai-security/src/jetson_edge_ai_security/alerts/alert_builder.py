"""Alert rendering from detector results."""

from __future__ import annotations

from jetson_edge_ai_security.schemas import Alert, DetectionResult

DEFAULT_ACTION = "Review correlated telemetry and isolate affected defensive lab systems if confirmed."


class AlertBuilder:
    """Build operator-facing alerts from detector results."""

    def __init__(self, *, source: str = "edge-security-runtime", recommended_action: str = DEFAULT_ACTION) -> None:
        self.source = source
        self.recommended_action = recommended_action

    def from_detection(self, result: DetectionResult) -> Alert | None:
        if not result.is_anomaly:
            return None
        reasons = "; ".join(result.reasons) if result.reasons else "baseline detector flagged anomaly"
        return Alert(
            timestamp=result.timestamp,
            severity=result.severity,
            title=f"{result.severity.title()} telemetry anomaly detected",
            description=f"Feature window triggered detection rules: {reasons}.",
            source=self.source,
            features=result.features.model_dump(mode="json"),
            recommended_action=self.recommended_action,
            metadata={"score": result.score, **result.metadata},
        )


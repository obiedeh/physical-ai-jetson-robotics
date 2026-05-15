"""Production-safe baseline detectors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from jetson_edge_ai_security.schemas import DetectionResult, FeatureWindow


class BaselineThresholds(BaseModel):
    """Thresholds for the simple rule-based detector."""

    packet_count_threshold: int = 500
    event_rate_threshold: float = 200.0
    unique_source_ip_threshold: int = 100
    attack_count_threshold: int = 1


class BaselineDetector:
    """Rule-based detector with optional IsolationForest scoring when installed."""

    def __init__(
        self,
        thresholds: BaselineThresholds | None = None,
        *,
        use_isolation_forest: bool = False,
        isolation_forest_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.thresholds = thresholds or BaselineThresholds()
        self.use_isolation_forest = use_isolation_forest
        self.isolation_forest_available = False
        self._model: Any | None = None
        self._isolation_forest_kwargs = isolation_forest_kwargs or {"random_state": 42, "contamination": "auto"}
        if use_isolation_forest:
            self._init_isolation_forest()

    def fit(self, windows: Iterable[FeatureWindow]) -> None:
        if self._model is None:
            return
        matrix = [_window_vector(window) for window in windows]
        if matrix:
            self._model.fit(matrix)

    def detect(self, window: FeatureWindow) -> DetectionResult:
        reasons: list[str] = []
        if window.packet_count >= self.thresholds.packet_count_threshold:
            reasons.append(f"packet_count >= {self.thresholds.packet_count_threshold}")
        if window.event_rate >= self.thresholds.event_rate_threshold:
            reasons.append(f"event_rate >= {self.thresholds.event_rate_threshold}")
        if window.unique_source_ip_count >= self.thresholds.unique_source_ip_threshold:
            reasons.append(f"unique_source_ip_count >= {self.thresholds.unique_source_ip_threshold}")
        if window.attack_count >= self.thresholds.attack_count_threshold:
            reasons.append("labeled attack events present in replay window")

        score = float(len(reasons))
        metadata: dict[str, Any] = {"detector": "baseline-threshold"}
        if self._model is not None:
            prediction = int(self._model.predict([_window_vector(window)])[0])
            anomaly_score = float(-self._model.score_samples([_window_vector(window)])[0])
            metadata["isolation_forest_score"] = anomaly_score
            if prediction == -1:
                reasons.append("IsolationForest marked window as anomalous")
                score += anomaly_score

        severity = _severity(score, window.attack_count)
        return DetectionResult(
            is_anomaly=bool(reasons),
            score=score,
            reasons=reasons,
            severity=severity,
            features=window,
            metadata=metadata,
        )

    def _init_isolation_forest(self) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            self.isolation_forest_available = False
            return
        self._model = IsolationForest(**self._isolation_forest_kwargs)
        self.isolation_forest_available = True


def _window_vector(window: FeatureWindow) -> list[float]:
    return [
        float(window.packet_count),
        float(window.mean_packet_size),
        float(window.max_packet_size),
        float(window.attack_count),
        float(window.event_rate),
        float(window.unique_source_ip_count),
        float(window.unique_dest_ip_count),
    ]


def _severity(score: float, attack_count: int) -> str:
    if score >= 4 or attack_count >= 10:
        return "critical"
    if score >= 3 or attack_count >= 3:
        return "high"
    if score >= 2 or attack_count >= 1:
        return "medium"
    return "low"


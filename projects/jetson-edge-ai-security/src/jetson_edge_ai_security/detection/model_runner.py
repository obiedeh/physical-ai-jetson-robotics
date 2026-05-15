"""Extensible detector runner interfaces."""

from __future__ import annotations

from typing import Protocol

from jetson_edge_ai_security.schemas import DetectionResult, FeatureWindow


class Detector(Protocol):
    """Protocol implemented by runtime detectors."""

    def detect(self, window: FeatureWindow) -> DetectionResult:
        """Score a feature window."""


class ModelRunner:
    """Thin wrapper around a detector implementation."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def detect(self, window: FeatureWindow) -> DetectionResult:
        return self.detector.detect(window)

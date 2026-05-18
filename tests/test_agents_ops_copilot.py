"""Smoke tests for the agentic ops copilot prototype."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agents.ops_copilot.copilot import (
    DeterministicClassifier,
    Enrichment,
    OpsCopilotReport,
    enrich,
)
from physical_ai_lab.ops_copilot import TriageFinding, TriageReport, triage_telemetry
from physical_ai_lab.telemetry import RobotTelemetrySample, generate_demo_telemetry


def test_enrich_with_deterministic_classifier_returns_ops_copilot_report() -> None:
    telemetry = generate_demo_telemetry(robot_id="robo-car-01", samples=12, seed=7)
    triage = triage_telemetry(telemetry)

    report = enrich(triage, DeterministicClassifier())

    assert isinstance(report, OpsCopilotReport)
    assert report.robot_id == "robo-car-01"
    assert report.status in {"nominal", "watch", "critical"}
    assert report.classifier_name == "deterministic"
    assert report.confidence in {"high", "medium", "low"}
    assert isinstance(report.escalate_to_human, bool)
    assert report.root_cause_hypothesis
    assert report.operator_recommendation


def test_to_dict_round_trips_through_json() -> None:
    telemetry = generate_demo_telemetry(samples=12, seed=7)
    triage = triage_telemetry(telemetry)
    report = enrich(triage, DeterministicClassifier())

    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    parsed = json.loads(rendered)

    assert parsed["robot_id"] == report.robot_id
    assert parsed["classifier_name"] == "deterministic"
    assert isinstance(parsed["operator_recommendation"], list)
    assert isinstance(parsed["triage"]["findings"], list)


def test_critical_finding_triggers_human_escalation() -> None:
    triage = TriageReport(
        robot_id="robo-car-01",
        status="critical",
        findings=[
            TriageFinding(
                severity="critical",
                signal="battery",
                summary="Battery is 5.0%.",
                recommended_action="Return to dock immediately.",
            ),
        ],
    )

    report = enrich(triage, DeterministicClassifier())

    assert report.escalate_to_human is True
    assert report.confidence == "high"
    assert "battery" in report.root_cause_hypothesis.lower()
    assert report.operator_recommendation == ["Return to dock immediately."]


def test_nominal_telemetry_does_not_escalate() -> None:
    sample = RobotTelemetrySample(
        timestamp=datetime.now(tz=timezone.utc),
        robot_id="robo-car-01",
        battery_percent=92.0,
        motor_temp_c=48.0,
        edge_latency_ms=18.0,
        network_latency_ms=14.0,
        localization_quality=0.95,
        task_success_probability=0.97,
    )

    triage = triage_telemetry([sample])
    report = enrich(triage, DeterministicClassifier())

    assert report.escalate_to_human is False
    assert report.confidence == "high"


def test_classifier_protocol_is_pluggable() -> None:
    class StubClassifier:
        name = "test-stub"

        def enrich(self, triage: TriageReport) -> Enrichment:
            return Enrichment(
                root_cause_hypothesis="stub hypothesis",
                operator_recommendation=["stub action"],
                confidence="low",
                escalate_to_human=False,
            )

    telemetry = generate_demo_telemetry(samples=12, seed=7)
    triage = triage_telemetry(telemetry)
    report = enrich(triage, StubClassifier())

    assert report.classifier_name == "test-stub"
    assert report.root_cause_hypothesis == "stub hypothesis"
    assert report.operator_recommendation == ["stub action"]
    assert report.confidence == "low"

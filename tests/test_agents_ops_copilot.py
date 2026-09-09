"""Smoke tests for the agentic ops copilot prototype."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agents.ops_copilot.copilot import (
    AnthropicClassifier,
    DeterministicClassifier,
    Enrichment,
    OpsCopilotReport,
    _build_triage_prompt,
    _parse_enrichment_json,
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


# ---------------------------------------------------------------------------
# AnthropicClassifier tests
# ---------------------------------------------------------------------------


def _make_triage(status: str = "watch") -> TriageReport:
    """Return a minimal TriageReport for classifier tests."""
    return TriageReport(
        robot_id="test-bot",
        status=status,
        findings=[
            TriageFinding(
                severity=status,
                signal="thermal",
                summary="Motor temperature is 72.0 C.",
                recommended_action="Reduce speed limits and allow cool-down.",
            )
        ],
    )


def test_anthropic_classifier_name_embeds_model() -> None:
    """AnthropicClassifier.name is 'anthropic/<model>' for report observability."""
    mock_client = MagicMock()
    classifier = AnthropicClassifier(model="claude-haiku-4-5-20251001", _client=mock_client)
    assert classifier.name == "anthropic/claude-haiku-4-5-20251001"


def test_anthropic_classifier_enrich_happy_path() -> None:
    """A well-formed JSON response from the API is parsed into an Enrichment."""
    payload = json.dumps({
        "root_cause_hypothesis": "Motor thermal stress from sustained load.",
        "operator_recommendation": ["Reduce speed.", "Inspect cooling."],
        "confidence": "medium",
        "escalate_to_human": False,
    })
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=payload)]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    classifier = AnthropicClassifier(_client=mock_client)
    triage = _make_triage(status="watch")
    report = enrich(triage, classifier)

    assert report.classifier_name.startswith("anthropic/")
    assert report.root_cause_hypothesis == "Motor thermal stress from sustained load."
    assert report.operator_recommendation == ["Reduce speed.", "Inspect cooling."]
    assert report.confidence == "medium"
    assert report.escalate_to_human is False

    # Verify the API was called once with a non-empty prompt.
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"].startswith("claude-")
    prompt_text: str = call_kwargs["messages"][0]["content"]
    assert "test-bot" in prompt_text
    assert "thermal" in prompt_text


def test_anthropic_classifier_falls_back_on_malformed_json() -> None:
    """When the API returns unparseable JSON, the deterministic fallback is used."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Sorry, I cannot help with that.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    classifier = AnthropicClassifier(_client=mock_client)
    triage = _make_triage(status="watch")
    report = enrich(triage, classifier)

    # Fell back to deterministic → hypothesis contains the watch-path wording.
    assert "narrowing" in report.root_cause_hypothesis
    assert report.escalate_to_human is False


def test_anthropic_classifier_falls_back_on_api_exception() -> None:
    """When the API call raises, the deterministic fallback is used."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("connection refused")

    classifier = AnthropicClassifier(_client=mock_client)
    triage = _make_triage(status="critical")
    report = enrich(triage, classifier)

    # Fell back to deterministic → critical path escalates.
    assert report.escalate_to_human is True
    assert report.confidence == "high"


def test_anthropic_classifier_raises_import_error_without_sdk() -> None:
    """Constructing AnthropicClassifier without the 'llm' extras raises ImportError."""
    with patch.dict(sys.modules, {"anthropic": None}):
        import pytest  # noqa: PLC0415 — imported here to avoid hoisting into module scope
        with pytest.raises(ImportError, match="llm"):
            AnthropicClassifier()


def test_build_triage_prompt_includes_robot_context() -> None:
    """_build_triage_prompt serializes robot_id, status, and finding signals."""
    triage = _make_triage(status="watch")
    prompt = _build_triage_prompt(triage)
    assert "test-bot" in prompt
    assert "watch" in prompt
    assert "thermal" in prompt
    assert "JSON" in prompt


def test_parse_enrichment_json_strips_markdown_fences() -> None:
    """_parse_enrichment_json handles ```json ... ``` wrapping from some models."""
    wrapped = (
        "```json\n"
        '{"root_cause_hypothesis": "test", "operator_recommendation": ["x"], '
        '"confidence": "high", "escalate_to_human": false}\n'
        "```"
    )
    enrichment = _parse_enrichment_json(wrapped)
    assert enrichment.root_cause_hypothesis == "test"
    assert enrichment.operator_recommendation == ["x"]
    assert enrichment.confidence == "high"
    assert enrichment.escalate_to_human is False

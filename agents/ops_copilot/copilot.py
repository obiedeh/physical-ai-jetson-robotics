"""Agentic ops copilot — enrichment layer over the deterministic safety-bounded triage.

This module consumes a deterministic ``TriageReport`` produced by
``physical_ai_lab.ops_copilot.triage_telemetry`` and asks a pluggable
``Classifier`` to add operator-facing context: a root-cause hypothesis, a list
of recommended actions, a confidence label, and an explicit
``escalate_to_human`` flag.

The deterministic triage is the safety-bounded baseline. The enrichment is
decision-support for qualified human operators, not autonomous control output.

Run the prototype end-to-end::

    # Deterministic (no API key required — hermetic CI default):
    python -m agents.ops_copilot.copilot --robot-id robo-car-01 --samples 12 \\
        --output reports/agents/ops_copilot/latest.json

    # LLM-backed (requires ANTHROPIC_API_KEY + pip install ...[llm]):
    python -m agents.ops_copilot.copilot --robot-id robo-car-01 --samples 12 \\
        --llm --output reports/agents/ops_copilot/latest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from physical_ai_lab.ops_copilot import TriageReport, triage_telemetry
from physical_ai_lab.telemetry import generate_demo_telemetry


@dataclass(frozen=True)
class Enrichment:
    """Classifier output, combined with a TriageReport to form an OpsCopilotReport."""

    root_cause_hypothesis: str
    operator_recommendation: list[str]
    confidence: str
    escalate_to_human: bool


class Classifier(Protocol):
    """Produces an enrichment over an existing deterministic TriageReport.

    Implementations may be deterministic rule-based fallbacks (for hermetic CI
    and machines without an LLM API configured) or LLM-backed adapters. The
    plug point is the protocol — no caller-side changes when implementations
    are swapped.
    """

    name: str

    def enrich(self, triage: TriageReport) -> Enrichment:
        ...


@dataclass(frozen=True)
class OpsCopilotReport:
    """Enriched ops report — deterministic triage plus classifier-added context."""

    robot_id: str
    status: str
    triage: TriageReport
    classifier_name: str
    root_cause_hypothesis: str
    operator_recommendation: list[str]
    confidence: str
    escalate_to_human: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "robot_id": self.robot_id,
            "status": self.status,
            "classifier_name": self.classifier_name,
            "confidence": self.confidence,
            "escalate_to_human": self.escalate_to_human,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "operator_recommendation": list(self.operator_recommendation),
            "triage": self.triage.to_dict(),
        }


class DeterministicClassifier:
    """Rule-based stub that produces structured enrichment without an LLM.

    This is the safety-bounded default that runs in hermetic CI and on machines
    without an LLM API configured. Replace with an LLM-backed Classifier when
    an LLM dependency is explicitly approved.
    """

    name = "deterministic"

    def enrich(self, triage: TriageReport) -> Enrichment:
        critical_findings = [f for f in triage.findings if f.severity == "critical"]
        watch_findings = [f for f in triage.findings if f.severity == "watch"]

        if critical_findings:
            signals = ", ".join(sorted({f.signal for f in critical_findings}))
            return Enrichment(
                root_cause_hypothesis=(
                    f"Critical baseline thresholds exceeded on {signals}. "
                    "Investigate before resuming autonomous operation."
                ),
                operator_recommendation=[f.recommended_action for f in critical_findings],
                confidence="high",
                escalate_to_human=True,
            )

        if watch_findings:
            signals = ", ".join(sorted({f.signal for f in watch_findings}))
            return Enrichment(
                root_cause_hypothesis=(
                    f"Operational margins narrowing on {signals}. "
                    "Continue with caution and capture telemetry evidence."
                ),
                operator_recommendation=[f.recommended_action for f in watch_findings],
                confidence="medium",
                escalate_to_human=False,
            )

        return Enrichment(
            root_cause_hypothesis="All telemetry signals within nominal bounds.",
            operator_recommendation=["Continue the planned run; keep logging evidence."],
            confidence="high",
            escalate_to_human=False,
        )


# ---------------------------------------------------------------------------
# Helpers for AnthropicClassifier
# ---------------------------------------------------------------------------


def _build_triage_prompt(triage: TriageReport) -> str:
    """Serialize a TriageReport into a structured prompt for the LLM."""
    findings_lines = "\n".join(
        f"  [{f.severity.upper()}] signal={f.signal}: {f.summary} "
        f"Suggested action: {f.recommended_action}"
        for f in triage.findings
    )
    return (
        "You are a robotics operations expert. "
        "Analyze the following telemetry triage for a physical robot and produce a "
        "structured enrichment in JSON.\n\n"
        f"Robot ID: {triage.robot_id}\n"
        f"Overall status: {triage.status}\n\n"
        f"Telemetry findings:\n{findings_lines}\n\n"
        "Respond with ONLY a JSON object — no markdown fences, no prose, no extra keys.\n"
        "Required schema:\n"
        '{\n'
        '  "root_cause_hypothesis": "<1-2 sentence explanation of the most likely root cause>",\n'
        '  "operator_recommendation": ["<action 1>", "<action 2>"],\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "escalate_to_human": true | false\n'
        '}\n\n'
        "Rules:\n"
        "- escalate_to_human MUST be true when status is \"critical\".\n"
        "- escalate_to_human MUST be false when status is \"nominal\".\n"
        "- confidence is \"high\" for critical or nominal, \"medium\" for watch.\n"
        "- operator_recommendation must contain at least one item.\n"
    )


def _parse_enrichment_json(text: str) -> Enrichment:
    """Parse an LLM text response into an Enrichment.

    Strips optional markdown code fences, then parses the inner JSON.
    Raises ``json.JSONDecodeError`` or ``KeyError`` on malformed input so
    ``AnthropicClassifier.enrich`` can catch and fall back to the deterministic path.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    data: Any = json.loads(text)
    return Enrichment(
        root_cause_hypothesis=str(data["root_cause_hypothesis"]),
        operator_recommendation=[str(r) for r in data["operator_recommendation"]],
        confidence=str(data["confidence"]),
        escalate_to_human=bool(data["escalate_to_human"]),
    )


# ---------------------------------------------------------------------------
# LLM-backed classifier
# ---------------------------------------------------------------------------


class AnthropicClassifier:
    """LLM-backed classifier using the Anthropic Messages API.

    Requires the ``llm`` optional dependency group::

        pip install "physical-ai-jetson-robotics-lab[llm]"

    and an ``ANTHROPIC_API_KEY`` environment variable (or an explicit
    ``api_key`` argument).

    Falls back to ``DeterministicClassifier`` output if the API call fails or
    returns unparseable JSON, so the ops copilot always emits a valid report
    regardless of transient network or API errors.

    The ``_client`` parameter is an injection point for unit tests; pass a mock
    object to exercise ``enrich`` without a real API call::

        classifier = AnthropicClassifier(_client=mock_client)
    """

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 512,
        _client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._fallback = DeterministicClassifier()
        # Allow injecting a pre-built client for testing without SDK import.
        if _client is not None:
            self._client: Any = _client
        else:
            try:
                import anthropic as _anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "AnthropicClassifier requires the 'llm' optional dependency. "
                    "Install it with: pip install 'physical-ai-jetson-robotics-lab[llm]'"
                ) from exc
            self._client = _anthropic.Anthropic(api_key=api_key)
        # Include model name in classifier_name for observability in reports.
        self.name = f"anthropic/{model}"

    def enrich(self, triage: TriageReport) -> Enrichment:
        """Call Claude to produce enrichment; fall back to deterministic on any error."""
        prompt = _build_triage_prompt(triage)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text: str = response.content[0].text
            return _parse_enrichment_json(text)
        except Exception:  # noqa: BLE001
            # Deterministic fallback keeps the copilot functional under API failures.
            return self._fallback.enrich(triage)


# ---------------------------------------------------------------------------
# Core enrichment function
# ---------------------------------------------------------------------------


def enrich(triage: TriageReport, classifier: Classifier) -> OpsCopilotReport:
    """Combine a deterministic TriageReport with a Classifier enrichment."""
    enrichment = classifier.enrich(triage)
    return OpsCopilotReport(
        robot_id=triage.robot_id,
        status=triage.status,
        triage=triage,
        classifier_name=classifier.name,
        root_cause_hypothesis=enrichment.root_cause_hypothesis,
        operator_recommendation=list(enrichment.operator_recommendation),
        confidence=enrichment.confidence,
        escalate_to_human=enrichment.escalate_to_human,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agents.ops_copilot.copilot",
        description="Generate an enriched agentic ops copilot report.",
    )
    parser.add_argument("--robot-id", default="robo-car-01", help="Robot identifier.")
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of synthetic telemetry samples to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/agents/ops_copilot/latest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help=(
            "Use the Anthropic LLM classifier instead of the deterministic rule-based fallback. "
            "Requires ANTHROPIC_API_KEY env var and 'pip install ....[llm]'. "
            "Falls back to the deterministic classifier if the API key is absent."
        ),
    )
    args = parser.parse_args(argv)

    classifier: Classifier
    if args.llm:
        if os.environ.get("ANTHROPIC_API_KEY"):
            classifier = AnthropicClassifier()
        else:
            print(
                "Warning: --llm requested but ANTHROPIC_API_KEY is not set. "
                "Falling back to the deterministic classifier."
            )
            classifier = DeterministicClassifier()
    else:
        classifier = DeterministicClassifier()

    telemetry = generate_demo_telemetry(robot_id=args.robot_id, samples=args.samples)
    triage = triage_telemetry(telemetry)
    report = enrich(triage, classifier)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    print(f"robot_id: {report.robot_id}")
    print(f"status: {report.status}")
    print(f"classifier: {report.classifier_name}  confidence: {report.confidence}")
    print(f"escalate_to_human: {report.escalate_to_human}")
    print()
    print("root_cause_hypothesis:")
    print(f"  {report.root_cause_hypothesis}")
    print()
    print("operator_recommendation:")
    for rec in report.operator_recommendation:
        print(f"  - {rec}")
    print()
    print(f"report written to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

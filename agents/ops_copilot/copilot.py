"""Agentic ops copilot — enrichment layer over the deterministic safety-bounded triage.

This module consumes a deterministic ``TriageReport`` produced by
``physical_ai_lab.ops_copilot.triage_telemetry`` and asks a pluggable
``Classifier`` to add operator-facing context: a root-cause hypothesis, a list
of recommended actions, a confidence label, and an explicit
``escalate_to_human`` flag.

The deterministic triage is the safety-bounded baseline. The enrichment is
decision-support for qualified human operators, not autonomous control output.

Run the prototype end-to-end::

    python -m agents.ops_copilot.copilot --robot-id robo-car-01 --samples 12 \\
        --output reports/agents/ops_copilot/latest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    args = parser.parse_args(argv)

    telemetry = generate_demo_telemetry(robot_id=args.robot_id, samples=args.samples)
    triage = triage_telemetry(telemetry)
    report = enrich(triage, DeterministicClassifier())

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

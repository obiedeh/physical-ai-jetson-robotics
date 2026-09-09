"""Rule-based robot operations triage for early copilot workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from physical_ai_lab.config import TriageThresholds
from physical_ai_lab.telemetry import RobotTelemetrySample, summarize_operational_risk

_DEFAULT_THRESHOLDS = TriageThresholds()


@dataclass(frozen=True)
class TriageFinding:
    """One operator-facing finding from robot telemetry."""

    severity: str
    signal: str
    summary: str
    recommended_action: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TriageReport:
    """A compact report suitable for CLI output or future RAG grounding."""

    robot_id: str
    status: str
    findings: list[TriageFinding]

    def to_dict(self) -> dict[str, object]:
        return {
            "robot_id": self.robot_id,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def triage_telemetry(
    samples: list[RobotTelemetrySample],
    thresholds: TriageThresholds = _DEFAULT_THRESHOLDS,
) -> TriageReport:
    """Create a deterministic operations triage report from robot telemetry."""
    summary = summarize_operational_risk(samples)
    latest = samples[-1]
    findings: list[TriageFinding] = []

    total_latency_ms = latest.edge_latency_ms + latest.network_latency_ms
    if latest.battery_percent < thresholds.battery_critical_pct:
        findings.append(
            TriageFinding(
                severity="critical",
                signal="battery",
                summary=f"Battery is {latest.battery_percent:.1f}%.",
                recommended_action="Return to dock before starting a new manipulation task.",
            )
        )
    elif latest.battery_percent < thresholds.battery_watch_pct:
        findings.append(
            TriageFinding(
                severity="watch",
                signal="battery",
                summary=f"Battery is {latest.battery_percent:.1f}%.",
                recommended_action="Schedule charging after the current run.",
            )
        )

    if latest.motor_temp_c >= thresholds.motor_temp_watch_c:
        findings.append(
            TriageFinding(
                severity="watch",
                signal="thermal",
                summary=f"Motor temperature is {latest.motor_temp_c:.1f} C.",
                recommended_action=(
                    "Reduce speed limits and inspect load/friction before repeat runs."
                ),
            )
        )

    if total_latency_ms >= thresholds.latency_watch_ms:
        findings.append(
            TriageFinding(
                severity="watch",
                signal="latency",
                summary=f"Combined edge/network latency is {total_latency_ms:.1f} ms.",
                recommended_action="Prefer local inference and defer cloud-dependent decisions.",
            )
        )

    if latest.localization_quality < thresholds.localization_critical:
        findings.append(
            TriageFinding(
                severity="critical",
                signal="localization",
                summary=f"Localization quality is {latest.localization_quality:.2f}.",
                recommended_action=(
                    "Pause autonomous navigation and recapture map or fiducial evidence."
                ),
            )
        )

    if not findings:
        findings.append(
            TriageFinding(
                severity="nominal",
                signal="operations",
                summary="No immediate telemetry limits are exceeded.",
                recommended_action="Continue the planned run and keep logging evidence.",
            )
        )

    return TriageReport(
        robot_id=str(summary["robot_id"]),
        status=str(summary["status"]),
        findings=findings,
    )

from datetime import datetime, timezone

from physical_ai_lab.ops_copilot import triage_telemetry
from physical_ai_lab.telemetry import RobotTelemetrySample, generate_demo_telemetry


def test_triage_telemetry_returns_nominal_finding_for_demo_data() -> None:
    report = triage_telemetry(generate_demo_telemetry(samples=3))

    assert report.robot_id == "robo-car-01"
    assert report.findings
    assert {finding.signal for finding in report.findings}


def test_triage_telemetry_flags_low_battery_and_localization() -> None:
    sample = RobotTelemetrySample(
        timestamp=datetime.now(tz=timezone.utc),
        robot_id="yahboom-01",
        battery_percent=18.0,
        motor_temp_c=64.0,
        edge_latency_ms=30.0,
        network_latency_ms=24.0,
        localization_quality=0.58,
        task_success_probability=0.81,
    )

    report = triage_telemetry([sample])

    signals = {finding.signal for finding in report.findings}
    severities = {finding.severity for finding in report.findings}
    assert {"battery", "localization"} <= signals
    assert "critical" in severities

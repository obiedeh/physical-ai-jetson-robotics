from physical_ai_lab.telemetry import generate_demo_telemetry, summarize_operational_risk


def test_generate_demo_telemetry_is_deterministic_for_same_seed() -> None:
    first = generate_demo_telemetry(samples=3, seed=123)
    second = generate_demo_telemetry(samples=3, seed=123)

    assert [sample.battery_percent for sample in first] == [
        sample.battery_percent for sample in second
    ]
    assert all(0.0 <= sample.risk_score <= 1.0 for sample in first)


def test_summarize_operational_risk_returns_operator_fields() -> None:
    samples = generate_demo_telemetry(robot_id="arm-01", samples=6)

    summary = summarize_operational_risk(samples)

    assert summary["robot_id"] == "arm-01"
    assert summary["status"] in {"nominal", "watch", "critical"}
    assert summary["peak_risk"] >= summary["average_risk"]

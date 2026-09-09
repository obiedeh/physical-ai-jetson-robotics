"""Synthetic robot telemetry used for demos, tests, and early agent workflows."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sin
from random import Random


@dataclass(frozen=True)
class RobotTelemetrySample:
    """One telemetry sample from a robot or simulated robot."""

    timestamp: datetime
    robot_id: str
    battery_percent: float
    motor_temp_c: float
    edge_latency_ms: float
    network_latency_ms: float
    localization_quality: float
    task_success_probability: float

    @property
    def risk_score(self) -> float:
        """Estimate operational risk on a 0-1 scale."""
        battery_risk = max(0.0, (35.0 - self.battery_percent) / 35.0)
        thermal_risk = max(0.0, (self.motor_temp_c - 65.0) / 25.0)
        latency_risk = max(0.0, (self.edge_latency_ms + self.network_latency_ms - 100.0) / 200.0)
        localization_risk = max(0.0, 1.0 - self.localization_quality)
        task_risk = max(0.0, 1.0 - self.task_success_probability)
        return min(
            1.0,
            0.20 * battery_risk
            + 0.20 * thermal_risk
            + 0.20 * latency_risk
            + 0.20 * localization_risk
            + 0.20 * task_risk,
        )


def generate_demo_telemetry(
    robot_id: str = "robo-car-01",
    samples: int = 24,
    seed: int = 7,
) -> list[RobotTelemetrySample]:
    """Generate deterministic robot telemetry for demos and cross-platform tests."""
    rng = Random(seed)
    start = datetime.now(tz=timezone.utc).replace(microsecond=0)
    rows: list[RobotTelemetrySample] = []

    for idx in range(samples):
        load_wave = 0.5 + 0.5 * sin(idx / 3.0)
        battery = max(5.0, 96.0 - idx * 1.8 + rng.uniform(-1.0, 1.0))
        motor_temp = 42.0 + load_wave * 28.0 + rng.uniform(-2.0, 2.0)
        edge_latency = 18.0 + load_wave * 35.0 + rng.uniform(-3.0, 3.0)
        network_latency = 12.0 + load_wave * 45.0 + rng.uniform(-4.0, 4.0)
        localization_quality = min(
            1.0,
            max(0.0, 0.96 - load_wave * 0.18 + rng.uniform(-0.03, 0.03)),
        )
        task_success = min(1.0, max(0.0, 0.98 - load_wave * 0.22 + rng.uniform(-0.04, 0.04)))

        rows.append(
            RobotTelemetrySample(
                timestamp=start + timedelta(minutes=5 * idx),
                robot_id=robot_id,
                battery_percent=round(battery, 2),
                motor_temp_c=round(motor_temp, 2),
                edge_latency_ms=round(edge_latency, 2),
                network_latency_ms=round(network_latency, 2),
                localization_quality=round(localization_quality, 3),
                task_success_probability=round(task_success, 3),
            )
        )

    return rows


def summarize_operational_risk(samples: list[RobotTelemetrySample]) -> dict[str, float | str]:
    """Summarize telemetry into simple operator-facing risk signals."""
    if not samples:
        raise ValueError("Cannot summarize an empty telemetry sequence.")

    latest = samples[-1]
    avg_risk = sum(sample.risk_score for sample in samples) / len(samples)
    peak_risk = max(sample.risk_score for sample in samples)

    if peak_risk >= 0.7:
        status = "critical"
    elif peak_risk >= 0.4:
        status = "watch"
    else:
        status = "nominal"

    return {
        "robot_id": latest.robot_id,
        "status": status,
        "average_risk": round(avg_risk, 3),
        "peak_risk": round(peak_risk, 3),
        "latest_battery_percent": latest.battery_percent,
        "latest_total_latency_ms": round(latest.edge_latency_ms + latest.network_latency_ms, 2),
        "latest_localization_quality": latest.localization_quality,
    }

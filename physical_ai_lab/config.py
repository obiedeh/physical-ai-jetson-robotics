"""Configuration models for robot lab hardware and runtime profiles."""

from enum import Enum

from pydantic import BaseModel, Field


class ComputeTarget(str, Enum):
    """Supported compute targets for planning and benchmark metadata."""

    workstation = "workstation"
    jetson_orin = "jetson-orin"
    jetson_thor = "jetson-thor"


class RobotPlatform(str, Enum):
    """Supported robot platforms in this lab."""

    robo_car = "robo-car"
    robotic_arm = "robotic-arm"
    digital_twin = "digital-twin"


class LabProfile(BaseModel):
    """A deployment profile that describes where a workload will run."""

    name: str = Field(default="local-dev")
    compute_target: ComputeTarget = Field(default=ComputeTarget.workstation)
    robot_platforms: list[RobotPlatform] = Field(
        default_factory=lambda: [RobotPlatform.digital_twin]
    )
    ros_domain_id: int = Field(default=42, ge=0, le=232)
    enable_safety_gate: bool = Field(default=True)

    def summary(self) -> str:
        platforms = ", ".join(platform.value for platform in self.robot_platforms)
        safety = "enabled" if self.enable_safety_gate else "disabled"
        return (
            f"{self.name}: compute={self.compute_target.value}, "
            f"platforms=[{platforms}], ros_domain_id={self.ros_domain_id}, "
            f"safety_gate={safety}"
        )


class TriageThresholds(BaseModel):
    """Configurable telemetry thresholds for the operations triage copilot.

    Defaults match the Yahboom ROSMASTER M3 Pro + Synria arm operating envelope.
    Override per-robot or per-deployment as hardware evidence is collected.
    """

    battery_critical_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    battery_watch_pct: float = Field(default=40.0, ge=0.0, le=100.0)
    motor_temp_watch_c: float = Field(default=72.0, ge=0.0)
    latency_watch_ms: float = Field(default=95.0, ge=0.0)
    localization_critical: float = Field(default=0.72, ge=0.0, le=1.0)

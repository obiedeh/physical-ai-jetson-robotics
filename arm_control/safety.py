"""Safety gate for Synria 6DOF arm commands.

Validates joint positions and velocities against the limits encoded in
``ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro``.
This is the first line of defence before any command reaches hardware or sim.

Usage::

    gate = ArmSafetyGate()
    results = gate.validate(
        positions={"joint_1": 0.5, "joint_2": -0.3, ...},
        velocities={"joint_1": 0.2, ...},
    )
    if not all(r.passed for r in results):
        for r in results:
            if not r.passed:
                print(r.message)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Joint specifications from synria_6dof_arm.urdf.xacro
#   (lower_rad, upper_rad, effort_nm, max_vel_rad_s)
# ---------------------------------------------------------------------------

_SYNRIA_JOINT_SPECS: dict[str, tuple[float, float, float, float]] = {
    "joint_1": (-math.pi,   math.pi,   30.0, 1.5),
    "joint_2": (-1.5708,    1.5708,    30.0, 1.5),
    "joint_3": (-2.3562,    2.3562,    25.0, 1.5),
    "joint_4": (-math.pi,   math.pi,   15.0, 2.0),
    "joint_5": (-1.5708,    1.5708,    15.0, 2.0),
    "joint_6": (-math.pi,   math.pi,   10.0, 2.5),
}

SYNRIA_MAX_PAYLOAD_KG: float = 1.0
SYNRIA_SAFE_SPEED_FRACTION: float = 0.5  # recommended for initial bring-up


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class SafetyViolation(str, Enum):
    none = "none"
    joint_limit = "joint_limit"
    velocity_limit = "velocity_limit"
    payload_limit = "payload_limit"
    emergency_stop = "emergency_stop"


@dataclass(frozen=True)
class SafetyCheckResult:
    """Result of one safety gate evaluation."""

    violation: SafetyViolation
    joint_name: str | None
    commanded_value: float | None
    limit_value: float | None
    message: str

    @property
    def passed(self) -> bool:
        return self.violation == SafetyViolation.none


class ArmSafetyConfig(BaseModel):
    """Per-deployment safety configuration.

    Override ``speed_fraction`` to a lower value (e.g. 0.25) during initial
    bring-up, and restore it to 1.0 once the arm and calibration are verified.
    """

    speed_fraction: float = Field(default=SYNRIA_SAFE_SPEED_FRACTION, gt=0.0, le=1.0)
    max_payload_kg: float = Field(default=SYNRIA_MAX_PAYLOAD_KG, gt=0.0)
    emergency_stop_active: bool = False


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------


class ArmSafetyGate:
    """Validates Synria arm commands before they reach hardware or simulation.

    All threshold values come from the URDF and vendor documentation.
    Override via ``ArmSafetyConfig`` for reduced-speed bring-up or testing.
    """

    def __init__(self, config: ArmSafetyConfig | None = None) -> None:
        self._cfg = config or ArmSafetyConfig()

    @property
    def config(self) -> ArmSafetyConfig:
        return self._cfg

    def check_joint_positions(
        self, positions: dict[str, float]
    ) -> list[SafetyCheckResult]:
        """Validate joint positions against URDF soft limits.

        Returns a list with one ``SafetyCheckResult`` per violation, or a
        single ``none`` result when all positions are within limits.
        """
        if self._cfg.emergency_stop_active:
            return [
                SafetyCheckResult(
                    violation=SafetyViolation.emergency_stop,
                    joint_name=None,
                    commanded_value=None,
                    limit_value=None,
                    message="Emergency stop is active — clear e-stop before commanding.",
                )
            ]

        violations: list[SafetyCheckResult] = []
        for joint, value in positions.items():
            spec = _SYNRIA_JOINT_SPECS.get(joint)
            if spec is None:
                continue  # unknown joint: not our constraint to enforce
            lower, upper, _, _ = spec
            if value < lower:
                violations.append(
                    SafetyCheckResult(
                        violation=SafetyViolation.joint_limit,
                        joint_name=joint,
                        commanded_value=round(value, 5),
                        limit_value=round(lower, 5),
                        message=(
                            f"{joint}: commanded {value:.4f} rad is below "
                            f"lower limit {lower:.4f} rad."
                        ),
                    )
                )
            elif value > upper:
                violations.append(
                    SafetyCheckResult(
                        violation=SafetyViolation.joint_limit,
                        joint_name=joint,
                        commanded_value=round(value, 5),
                        limit_value=round(upper, 5),
                        message=(
                            f"{joint}: commanded {value:.4f} rad exceeds "
                            f"upper limit {upper:.4f} rad."
                        ),
                    )
                )

        return violations or [
            SafetyCheckResult(
                violation=SafetyViolation.none,
                joint_name=None,
                commanded_value=None,
                limit_value=None,
                message="All joint positions within limits.",
            )
        ]

    def check_joint_velocities(
        self, velocities: dict[str, float]
    ) -> list[SafetyCheckResult]:
        """Validate joint velocities against speed-fraction-scaled URDF limits."""
        violations: list[SafetyCheckResult] = []

        for joint, vel in velocities.items():
            spec = _SYNRIA_JOINT_SPECS.get(joint)
            if spec is None:
                continue
            _, _, _, max_vel = spec
            scaled_max = max_vel * self._cfg.speed_fraction
            if abs(vel) > scaled_max:
                violations.append(
                    SafetyCheckResult(
                        violation=SafetyViolation.velocity_limit,
                        joint_name=joint,
                        commanded_value=round(vel, 5),
                        limit_value=round(scaled_max, 5),
                        message=(
                            f"{joint}: |velocity| {abs(vel):.4f} rad/s exceeds "
                            f"speed-fraction limit {scaled_max:.4f} rad/s "
                            f"(fraction={self._cfg.speed_fraction})."
                        ),
                    )
                )

        return violations or [
            SafetyCheckResult(
                violation=SafetyViolation.none,
                joint_name=None,
                commanded_value=None,
                limit_value=None,
                message="All joint velocities within limits.",
            )
        ]

    def validate(
        self,
        positions: dict[str, float],
        velocities: dict[str, float] | None = None,
    ) -> list[SafetyCheckResult]:
        """Run full safety validation: positions then (optionally) velocities.

        Returns the first e-stop result immediately. Otherwise collects all
        position violations first, then velocity violations. A clean run
        returns a single ``none`` result.
        """
        pos_results = self.check_joint_positions(positions)
        if any(r.violation == SafetyViolation.emergency_stop for r in pos_results):
            return pos_results

        all_violations = [r for r in pos_results if not r.passed]
        if velocities:
            all_violations += [
                r for r in self.check_joint_velocities(velocities) if not r.passed
            ]

        return all_violations or [
            SafetyCheckResult(
                violation=SafetyViolation.none,
                joint_name=None,
                commanded_value=None,
                limit_value=None,
                message="All safety checks passed.",
            )
        ]

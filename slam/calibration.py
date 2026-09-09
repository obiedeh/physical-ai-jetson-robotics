"""Odometry calibration evidence helpers for the Yahboom ROSMASTER M3 Pro.

Tracks commanded vs measured motion across a set of calibration trials and
produces a ``CalibrationReport`` suitable for committing under
``reports/yahboom/`` as hardware evidence.

This module extends the mecanum calibration data model in
``physical_ai_lab/slam.py`` with richer per-axis evidence and pass/fail
criteria that match the Milestone B requirements in
``docs/YAHBOOM_ROBOT_CAR_PLAN.md``.

Usage::

    from slam.calibration import CalibrationTrial, run_calibration_session

    trials = [
        CalibrationTrial("forward-1m",  axis="x", commanded_m=1.0, measured_m=0.97, drift_m=0.03),
        CalibrationTrial("strafe-1m",   axis="y", commanded_m=1.0, measured_m=0.91, drift_m=0.07),
        CalibrationTrial(
            "rotate-90deg", axis="yaw", commanded_rad=1.5708, measured_rad=1.53, drift_rad=0.04
        ),
    ]
    report = run_calibration_session("yahboom-orin-01", trials)
    print(report.status)           # "pass" or "needs-tuning"
    print(report.to_json())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Per-trial pass criteria (from Milestone B in YAHBOOM_ROBOT_CAR_PLAN.md)
# ---------------------------------------------------------------------------

MAX_LINEAR_ERROR_PCT: float = 8.0    # ≤ 8% distance error for SLAM bring-up
MAX_ANGULAR_ERROR_RAD: float = 0.08  # ≤ ~4.6° error per 90° command
MAX_LINEAR_DRIFT_M: float = 0.06     # ≤ 6 cm lateral drift over 1 m
MAX_ANGULAR_DRIFT_RAD: float = 0.05  # ≤ ~2.9° angular drift per translation trial


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationTrial:
    """One commanded-vs-measured mecanum calibration trial.

    Attributes:
        name: Short trial identifier (e.g. "forward-1m", "strafe-right-1m").
        axis: Motion axis — "x" (forward), "y" (strafe), or "yaw" (rotation).
        commanded_m: Commanded linear displacement in metres (linear trials).
        measured_m: Measured linear displacement in metres (linear trials).
        drift_m: Lateral / perpendicular drift in metres (linear trials).
        commanded_rad: Commanded rotation in radians (yaw trials).
        measured_rad: Measured rotation in radians (yaw trials).
        drift_rad: Angular drift in radians (yaw trials).
        notes: Free-text trial context (surface, speed, load, etc.).
    """

    name: str
    axis: str
    commanded_m: float = 0.0
    measured_m: float = 0.0
    drift_m: float = 0.0
    commanded_rad: float = 0.0
    measured_rad: float = 0.0
    drift_rad: float = 0.0
    notes: str = ""

    @property
    def is_yaw(self) -> bool:
        return self.axis == "yaw"

    @property
    def linear_error_pct(self) -> float:
        """Percentage distance error for linear trials."""
        if self.is_yaw or self.commanded_m == 0.0:
            return 0.0
        return round(abs(self.measured_m - self.commanded_m) / self.commanded_m * 100.0, 3)

    @property
    def angular_error_rad(self) -> float:
        """Absolute angular error in radians for yaw trials."""
        if not self.is_yaw:
            return 0.0
        return round(abs(self.measured_rad - self.commanded_rad), 5)

    def passes(
        self,
        max_linear_error_pct: float = MAX_LINEAR_ERROR_PCT,
        max_angular_error_rad: float = MAX_ANGULAR_ERROR_RAD,
        max_linear_drift_m: float = MAX_LINEAR_DRIFT_M,
    ) -> bool:
        """Return True when this trial meets the calibration acceptance criteria."""
        if self.is_yaw:
            return self.angular_error_rad <= max_angular_error_rad
        return (
            self.linear_error_pct <= max_linear_error_pct
            and abs(self.drift_m) <= max_linear_drift_m
        )

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = asdict(self)
        d["linear_error_pct"] = self.linear_error_pct
        d["angular_error_rad"] = self.angular_error_rad
        d["passes"] = self.passes()
        return d


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregate calibration evidence for one session.

    Attributes:
        robot_id: Robot that performed the calibration.
        session_id: Human-readable session label.
        captured_at: ISO-8601 timestamp.
        trials: List of individual calibration trials.
        status: "pass" when all trials pass, "needs-tuning" otherwise.
        max_linear_error_pct: Worst linear distance error across trials.
        max_angular_error_rad: Worst angular error across yaw trials.
        max_drift_m: Worst lateral drift across linear trials.
        notes: Free-text session notes.
    """

    robot_id: str
    session_id: str
    captured_at: str
    trials: list[CalibrationTrial]
    status: str
    max_linear_error_pct: float
    max_angular_error_rad: float
    max_drift_m: float
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "robot_id": self.robot_id,
            "session_id": self.session_id,
            "captured_at": self.captured_at,
            "status": self.status,
            "max_linear_error_pct": self.max_linear_error_pct,
            "max_angular_error_rad": self.max_angular_error_rad,
            "max_drift_m": self.max_drift_m,
            "notes": self.notes,
            "trials": [t.to_dict() for t in self.trials],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def failing_trials(self) -> list[CalibrationTrial]:
        return [t for t in self.trials if not t.passes()]


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


def run_calibration_session(
    robot_id: str,
    trials: list[CalibrationTrial],
    session_id: str = "",
    notes: str = "",
    max_linear_error_pct: float = MAX_LINEAR_ERROR_PCT,
    max_angular_error_rad: float = MAX_ANGULAR_ERROR_RAD,
    max_linear_drift_m: float = MAX_LINEAR_DRIFT_M,
) -> CalibrationReport:
    """Aggregate calibration trials into a pass/fail report.

    Args:
        robot_id: Robot identifier.
        trials: List of ``CalibrationTrial`` objects.
        session_id: Human-readable session label (defaults to ISO timestamp).
        notes: Free-text session context.
        max_linear_error_pct: Linear distance error threshold.
        max_angular_error_rad: Angular error threshold for yaw trials.
        max_linear_drift_m: Lateral drift threshold for linear trials.

    Raises:
        ValueError: When ``trials`` is empty.
    """
    if not trials:
        raise ValueError("At least one calibration trial is required.")

    now = datetime.now(tz=timezone.utc).isoformat()
    sid = session_id or now

    linear_errors = [t.linear_error_pct for t in trials if not t.is_yaw]
    angular_errors = [t.angular_error_rad for t in trials if t.is_yaw]
    drifts = [abs(t.drift_m) for t in trials if not t.is_yaw]

    max_lin = max(linear_errors, default=0.0)
    max_ang = max(angular_errors, default=0.0)
    max_dri = max(drifts, default=0.0)

    all_pass = all(
        t.passes(
            max_linear_error_pct=max_linear_error_pct,
            max_angular_error_rad=max_angular_error_rad,
            max_linear_drift_m=max_linear_drift_m,
        )
        for t in trials
    )

    return CalibrationReport(
        robot_id=robot_id,
        session_id=sid,
        captured_at=now,
        trials=list(trials),
        status="pass" if all_pass else "needs-tuning",
        max_linear_error_pct=round(max_lin, 3),
        max_angular_error_rad=round(max_ang, 5),
        max_drift_m=round(max_dri, 4),
        notes=notes,
    )


def demo_calibration_trials() -> list[CalibrationTrial]:
    """Return a deterministic set of sample calibration trials for CLI demos."""
    return [
        CalibrationTrial(
            name="forward-1m",
            axis="x",
            commanded_m=1.0,
            measured_m=0.96,
            drift_m=0.028,
            notes="Indoor linoleum, speed 0.3 m/s",
        ),
        CalibrationTrial(
            name="strafe-right-1m",
            axis="y",
            commanded_m=1.0,
            measured_m=0.93,
            drift_m=0.055,
            notes="Same surface, speed 0.2 m/s",
        ),
        CalibrationTrial(
            name="diagonal-forward-right-1m",
            axis="x",
            commanded_m=1.0,
            measured_m=0.94,
            drift_m=0.043,
            notes="Diagonal mecanum mode",
        ),
        CalibrationTrial(
            name="rotate-cw-90deg",
            axis="yaw",
            commanded_rad=1.5708,
            measured_rad=1.53,
            drift_rad=0.04,
            notes="In-place rotation",
        ),
    ]

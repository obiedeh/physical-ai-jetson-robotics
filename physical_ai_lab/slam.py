"""SLAM and mecanum-base calibration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MecanumCalibrationTrial:
    """One commanded-vs-measured base-motion trial."""

    command: str
    commanded_m: float
    measured_m: float
    drift_m: float

    @property
    def distance_error_percent(self) -> float:
        if self.commanded_m == 0:
            return 0.0
        return round(abs(self.measured_m - self.commanded_m) / self.commanded_m * 100.0, 2)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["distance_error_percent"] = self.distance_error_percent
        return data


@dataclass(frozen=True)
class MecanumCalibrationSummary:
    """Aggregate calibration evidence for the Yahboom mecanum base."""

    trials: list[MecanumCalibrationTrial]
    max_distance_error_percent: float
    max_drift_m: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "max_distance_error_percent": self.max_distance_error_percent,
            "max_drift_m": self.max_drift_m,
            "trials": [trial.to_dict() for trial in self.trials],
        }


def summarize_mecanum_calibration(
    trials: list[MecanumCalibrationTrial],
    max_error_percent: float = 10.0,
    max_drift_m: float = 0.08,
) -> MecanumCalibrationSummary:
    """Summarize whether mecanum base motion is ready for SLAM bring-up."""
    if not trials:
        raise ValueError("At least one mecanum calibration trial is required.")

    worst_error = max(trial.distance_error_percent for trial in trials)
    worst_drift = max(abs(trial.drift_m) for trial in trials)
    status = (
        "pass"
        if worst_error <= max_error_percent and worst_drift <= max_drift_m
        else "needs-tuning"
    )

    return MecanumCalibrationSummary(
        trials=trials,
        max_distance_error_percent=round(worst_error, 2),
        max_drift_m=round(worst_drift, 3),
        status=status,
    )


def demo_mecanum_trials() -> list[MecanumCalibrationTrial]:
    """Return deterministic sample trials for CLI and documentation demos."""
    return [
        MecanumCalibrationTrial(
            command="forward-1m",
            commanded_m=1.0,
            measured_m=0.96,
            drift_m=0.035,
        ),
        MecanumCalibrationTrial(
            command="strafe-right-1m",
            commanded_m=1.0,
            measured_m=0.92,
            drift_m=0.061,
        ),
        MecanumCalibrationTrial(
            command="diagonal-forward-right-1m",
            commanded_m=1.0,
            measured_m=0.94,
            drift_m=0.052,
        ),
    ]

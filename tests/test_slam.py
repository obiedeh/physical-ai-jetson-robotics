import pytest

from physical_ai_lab.slam import (
    MecanumCalibrationTrial,
    demo_mecanum_trials,
    summarize_mecanum_calibration,
)


def test_demo_mecanum_trials_pass_initial_thresholds() -> None:
    summary = summarize_mecanum_calibration(demo_mecanum_trials())

    assert summary.status == "pass"
    assert summary.max_distance_error_percent <= 10.0
    assert summary.max_drift_m <= 0.08


def test_mecanum_summary_flags_large_strafe_error() -> None:
    summary = summarize_mecanum_calibration(
        [
            MecanumCalibrationTrial(
                command="strafe-right-1m",
                commanded_m=1.0,
                measured_m=0.82,
                drift_m=0.12,
            )
        ]
    )

    assert summary.status == "needs-tuning"
    assert summary.max_distance_error_percent == 18.0


def test_mecanum_summary_requires_trials() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_mecanum_calibration([])

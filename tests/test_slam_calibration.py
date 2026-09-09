"""Tests for slam.calibration — CalibrationTrial, CalibrationReport, session runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam.calibration import (
    MAX_ANGULAR_ERROR_RAD,
    MAX_LINEAR_DRIFT_M,
    MAX_LINEAR_ERROR_PCT,
    CalibrationTrial,
    demo_calibration_trials,
    run_calibration_session,
)

# ---------------------------------------------------------------------------
# CalibrationTrial — linear trial
# ---------------------------------------------------------------------------


def _linear(
    name: str = "forward-1m",
    commanded_m: float = 1.0,
    measured_m: float = 0.95,
    drift_m: float = 0.02,
) -> CalibrationTrial:
    return CalibrationTrial(
        name=name,
        axis="x",
        commanded_m=commanded_m,
        measured_m=measured_m,
        drift_m=drift_m,
    )


def _yaw(
    name: str = "rotate-90",
    commanded_rad: float = 1.5708,
    measured_rad: float = 1.54,
    drift_rad: float = 0.02,
) -> CalibrationTrial:
    return CalibrationTrial(
        name=name,
        axis="yaw",
        commanded_rad=commanded_rad,
        measured_rad=measured_rad,
        drift_rad=drift_rad,
    )


def test_linear_trial_error_pct() -> None:
    trial = _linear(commanded_m=1.0, measured_m=0.92)
    assert trial.linear_error_pct == pytest.approx(8.0)


def test_linear_trial_error_pct_exact() -> None:
    trial = _linear(commanded_m=2.0, measured_m=1.9)
    assert trial.linear_error_pct == pytest.approx(5.0)


def test_yaw_trial_linear_error_is_zero() -> None:
    trial = _yaw()
    assert trial.linear_error_pct == 0.0


def test_yaw_trial_angular_error() -> None:
    trial = _yaw(commanded_rad=1.5708, measured_rad=1.53)
    assert trial.angular_error_rad == pytest.approx(abs(1.53 - 1.5708), rel=1e-4)


def test_linear_trial_angular_error_is_zero() -> None:
    trial = _linear()
    assert trial.angular_error_rad == 0.0


def test_is_yaw_property() -> None:
    assert _yaw().is_yaw is True
    assert _linear().is_yaw is False


def test_linear_trial_passes_within_limits() -> None:
    # error = 5%, drift = 0.02 m — both well within limits
    trial = _linear(commanded_m=1.0, measured_m=0.95, drift_m=0.02)
    assert trial.passes() is True


def test_linear_trial_fails_on_excessive_error() -> None:
    # 15% error — exceeds MAX_LINEAR_ERROR_PCT
    trial = _linear(commanded_m=1.0, measured_m=0.85, drift_m=0.01)
    assert trial.passes() is False


def test_linear_trial_fails_on_excessive_drift() -> None:
    # drift = 0.10 m — exceeds MAX_LINEAR_DRIFT_M (0.06 m)
    trial = _linear(commanded_m=1.0, measured_m=0.97, drift_m=0.10)
    assert trial.passes() is False


def test_yaw_trial_passes_within_limits() -> None:
    trial = _yaw(commanded_rad=1.5708, measured_rad=1.54)
    assert trial.passes() is True


def test_yaw_trial_fails_on_large_error() -> None:
    # angular error = 0.15 rad — exceeds MAX_ANGULAR_ERROR_RAD (0.08)
    trial = _yaw(commanded_rad=1.5708, measured_rad=1.42)
    assert trial.passes() is False


def test_trial_to_dict_includes_computed_fields() -> None:
    trial = _linear()
    d = trial.to_dict()
    assert "linear_error_pct" in d
    assert "angular_error_rad" in d
    assert "passes" in d
    assert isinstance(d["passes"], bool)


# ---------------------------------------------------------------------------
# run_calibration_session
# ---------------------------------------------------------------------------


def test_session_pass_all_trials() -> None:
    trials = [
        _linear(name="fwd", commanded_m=1.0, measured_m=0.96, drift_m=0.02),
        _yaw(name="rot", commanded_rad=1.5708, measured_rad=1.54),
    ]
    report = run_calibration_session("test-bot", trials)
    assert report.status == "pass"


def test_session_needs_tuning_when_trial_fails() -> None:
    trials = [
        _linear(name="bad", commanded_m=1.0, measured_m=0.70, drift_m=0.01),  # 30% error
    ]
    report = run_calibration_session("test-bot", trials)
    assert report.status == "needs-tuning"


def test_session_empty_trials_raises() -> None:
    with pytest.raises(ValueError, match="At least one"):
        run_calibration_session("test-bot", [])


def test_session_max_metrics_computed() -> None:
    trials = [
        _linear(name="a", commanded_m=1.0, measured_m=0.94, drift_m=0.03),  # 6%
        _linear(name="b", commanded_m=1.0, measured_m=0.96, drift_m=0.05),  # 4%
    ]
    report = run_calibration_session("r", trials)
    assert report.max_linear_error_pct == pytest.approx(6.0)
    assert report.max_drift_m == pytest.approx(0.05)


def test_session_robot_id_stored() -> None:
    report = run_calibration_session("yahboom-42", [_linear()])
    assert report.robot_id == "yahboom-42"


def test_session_custom_session_id() -> None:
    report = run_calibration_session("r", [_linear()], session_id="my-session")
    assert report.session_id == "my-session"


# ---------------------------------------------------------------------------
# CalibrationReport serialisation
# ---------------------------------------------------------------------------


def test_report_to_json_is_valid_json() -> None:
    report = run_calibration_session("r", [_linear()])
    data = json.loads(report.to_json())
    assert "robot_id" in data
    assert "status" in data
    assert "trials" in data


def test_report_to_dict_round_trips() -> None:
    report = run_calibration_session("r", demo_calibration_trials())
    d = report.to_dict()
    assert d["robot_id"] == "r"
    assert isinstance(d["trials"], list)
    assert len(d["trials"]) == len(demo_calibration_trials())


def test_report_failing_trials_list() -> None:
    good = _linear(name="ok", commanded_m=1.0, measured_m=0.97, drift_m=0.01)
    bad = _linear(name="fail", commanded_m=1.0, measured_m=0.50, drift_m=0.01)
    report = run_calibration_session("r", [good, bad])
    failing = report.failing_trials()
    assert len(failing) == 1
    assert failing[0].name == "fail"


# ---------------------------------------------------------------------------
# demo_calibration_trials
# ---------------------------------------------------------------------------


def test_demo_trials_not_empty() -> None:
    trials = demo_calibration_trials()
    assert len(trials) >= 3


def test_demo_trials_all_pass() -> None:
    trials = demo_calibration_trials()
    for t in trials:
        assert t.passes(), f"Demo trial '{t.name}' should pass"


def test_demo_session_pass() -> None:
    report = run_calibration_session("demo-bot", demo_calibration_trials())
    assert report.status == "pass"


# ---------------------------------------------------------------------------
# Threshold constants sanity
# ---------------------------------------------------------------------------


def test_threshold_constants_sane() -> None:
    assert 0 < MAX_LINEAR_ERROR_PCT <= 20
    assert 0 < MAX_ANGULAR_ERROR_RAD < 0.5
    assert 0 < MAX_LINEAR_DRIFT_M < 0.2

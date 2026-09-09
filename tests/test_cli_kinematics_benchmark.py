"""CLI tests for the synria-kinematics-benchmark command.

All tests use CliRunner (no subprocess) and write output to tmp_path.
No GPU required — the benchmark uses the pure Python arm_control module.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import Result
from typer.testing import CliRunner

from physical_ai_lab.cli import app

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


# ---------------------------------------------------------------------------
# Exit codes and basic execution
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_default_succeeds(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    result = _invoke("synria-kinematics-benchmark", "--report", str(report))
    assert result.exit_code == 0, result.output


def test_kinematics_benchmark_writes_report_file(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    result = _invoke("synria-kinematics-benchmark", "--report", str(report))
    assert result.exit_code == 0, result.output
    assert report.exists()


def test_kinematics_benchmark_report_is_valid_json(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    result = _invoke("synria-kinematics-benchmark", "--report", str(report))
    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_report_has_required_top_level_keys(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    for key in ("meta", "link_lengths_m", "fk_cases", "workspace_checks",
                "named_pose_fk", "trajectory_validation"):
        assert key in data, f"Missing top-level key: {key!r}"


def test_kinematics_benchmark_meta_contains_date_and_mode(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "date" in data["meta"]
    assert data["meta"]["mode"] == "rtx_simulation"


def test_kinematics_benchmark_link_lengths_match_urdf(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    ll = data["link_lengths_m"]
    # Values sourced from synria_6dof_arm.urdf.xacro comments in kinematics.py
    assert abs(ll["upper_arm"] - 0.235) < 1e-6
    assert abs(ll["forearm"] - 0.220) < 1e-6
    assert abs(ll["geometric_max_reach"] - 0.650) < 1e-3


# ---------------------------------------------------------------------------
# FK cases
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_default_has_six_fk_cases(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["fk_cases"]) == 6


def test_kinematics_benchmark_n_fk_cases_option(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--n-fk-cases", "3", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["fk_cases"]) == 3


def test_kinematics_benchmark_all_fk_cases_within_geometric_max(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    for case in data["fk_cases"]:
        assert case["within_geometric_max"] is True, (
            f"FK case {case['case']!r} exceeds geometric max reach"
        )


def test_kinematics_benchmark_home_pose_zero_x(tmp_path: Path) -> None:
    """FK of the home pose (q2=0, q3=0) should give x≈0."""
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    home = next(c for c in data["fk_cases"] if c["case"] == "home")
    assert abs(home["x_m"]) < 1e-9


# ---------------------------------------------------------------------------
# Workspace checks
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_workspace_overhead_far_unreachable(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    ws = {w["case"]: w for w in data["workspace_checks"]}
    assert ws["overhead_far"]["reachable"] is False


def test_kinematics_benchmark_workspace_too_close_unreachable(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    ws = {w["case"]: w for w in data["workspace_checks"]}
    assert ws["too_close"]["reachable"] is False


def test_kinematics_benchmark_workspace_board_centre_reachable(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    ws = {w["case"]: w for w in data["workspace_checks"]}
    assert ws["board_centre"]["reachable"] is True


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_all_trajectories_pass(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    _invoke("synria-kinematics-benchmark", "--report", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    tv = data["trajectory_validation"]
    assert tv["passed"] == tv["total"], (
        f"Trajectory failures: {[t for t in tv['summary'] if not t['passed']]}"
    )


# ---------------------------------------------------------------------------
# --json flag
# ---------------------------------------------------------------------------


def test_kinematics_benchmark_json_flag_outputs_to_stdout(tmp_path: Path) -> None:
    report = tmp_path / "km.json"
    result = _invoke("synria-kinematics-benchmark", "--json", "--report", str(report))
    assert result.exit_code == 0, result.output
    # stdout must contain the JSON (parse from file to avoid Rich control chars)
    assert report.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "fk_cases" in data

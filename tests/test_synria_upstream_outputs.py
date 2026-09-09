"""Tests that all SYNRIA_UPSTREAM_TODO.md RTX-Now output artifacts exist.

Every item in the 'RTX-Now Activities' section of
``docs/SYNRIA_UPSTREAM_TODO.md`` lists an explicit output path. This module
asserts those paths exist so the checklist can be verified in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# RTX-Now output artifacts
# ---------------------------------------------------------------------------


def test_synria_vendor_description_parity_report_exists() -> None:
    path = REPO_ROOT / "docs" / "reports" / "synria_vendor_description_parity.md"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    text = path.read_text(encoding="utf-8")
    assert "Joint1" in text
    assert "tool0" in text
    assert "parity" in text.lower() or "audit" in text.lower()


def test_synria_ros2_jazzy_port_report_exists() -> None:
    path = REPO_ROOT / "docs" / "reports" / "synria_ros2_jazzy_port.md"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    text = path.read_text(encoding="utf-8")
    assert "Jazzy" in text
    assert "Humble" in text


def test_synria_c10_camera_contract_exists() -> None:
    path = REPO_ROOT / "docs" / "reports" / "synria_c10_camera_contract.md"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    text = path.read_text(encoding="utf-8")
    assert "c10_camera" in text or "C10" in text
    assert "/c10_camera/image_raw" in text


def test_synria_hand_eye_calibration_sim_report_exists() -> None:
    path = REPO_ROOT / "reports" / "synria" / "hand_eye_calibration_sim.md"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    text = path.read_text(encoding="utf-8")
    assert "calibration" in text.lower()
    assert "tool0" in text or "T_ee_to_camera" in text or "hand-eye" in text.lower()


def test_synria_kinematics_benchmark_json_exists() -> None:
    path = REPO_ROOT / "reports" / "training" / "synria_kinematics_benchmark.json"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "fk_cases" in data
    assert "workspace_checks" in data
    assert "trajectory_validation" in data
    assert data["trajectory_validation"]["passed"] == data["trajectory_validation"]["total"]


def test_synria_kinematics_benchmark_fk_cases_within_reach() -> None:
    path = REPO_ROOT / "reports" / "training" / "synria_kinematics_benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for case in data["fk_cases"]:
        assert case["within_geometric_max"] is True, (
            f"FK case {case['case']!r} exceeds geometric max reach"
        )


def test_synria_kinematics_benchmark_workspace_unreachable_cases() -> None:
    """The benchmark must confirm that out-of-range targets are flagged correctly."""
    path = REPO_ROOT / "reports" / "training" / "synria_kinematics_benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ws_map = {w["case"]: w for w in data["workspace_checks"]}
    assert ws_map["overhead_far"]["reachable"] is False
    assert ws_map["too_close"]["reachable"] is False
    assert ws_map["board_centre"]["reachable"] is True


# ---------------------------------------------------------------------------
# Pre-existing RTX-Now outputs (already existed before this batch)
# ---------------------------------------------------------------------------


def test_synria_mock_ros2_controllers_yaml_exists() -> None:
    """Item 3: Mock ros2_control hardware — already present."""
    path = (
        REPO_ROOT
        / "ros2_ws"
        / "src"
        / "synria_arm_gazebo"
        / "config"
        / "ros2_controllers.yaml"
    )
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"


def test_synria_lerobot_recording_schema_contract_exists() -> None:
    """Item 7: LeRobot recording schema contract — already present."""
    path = REPO_ROOT / "lerobot" / "configs" / "synria_aloha_act_notes.yaml"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"


# ---------------------------------------------------------------------------
# CLI-generated demo artifacts (ops-copilot-health + lerobot-cube-sort-demo)
# ---------------------------------------------------------------------------


def test_cube_sort_demo_report_exists() -> None:
    """lerobot-cube-sort-demo default output — written by CLI at default path."""
    path = REPO_ROOT / "reports" / "demo" / "synria_cube_sort_sim.md"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    text = path.read_text(encoding="utf-8")
    assert "Cube Sort" in text or "cube sort" in text.lower()
    assert "pass_rate" in text.lower() or "Pass rate" in text or "passed" in text.lower()


def test_cube_sort_demo_report_has_no_failures() -> None:
    """All cubes in the default-seed run should pass — no ❌ rows."""
    path = REPO_ROOT / "reports" / "demo" / "synria_cube_sort_sim.md"
    text = path.read_text(encoding="utf-8")
    assert "❌" not in text


def test_ops_copilot_health_json_exists() -> None:
    """ops-copilot-health default output — health gate evidence artifact."""
    path = REPO_ROOT / "reports" / "agents" / "ops_copilot" / "health.json"
    assert path.exists(), f"Missing: {path.relative_to(REPO_ROOT)}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "healthy"


def test_ops_copilot_health_json_all_checks_pass() -> None:
    path = REPO_ROOT / "reports" / "agents" / "ops_copilot" / "health.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = data.get("checks", {})
    assert all(v is True for v in checks.values()), f"Failed checks: {checks}"


def test_ops_copilot_health_json_has_expected_check_keys() -> None:
    path = REPO_ROOT / "reports" / "agents" / "ops_copilot" / "health.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = data.get("checks", {})
    for key in ("telemetry_generation", "triage", "enrichment", "serialisation"):
        assert key in checks, f"Missing check: {key}"

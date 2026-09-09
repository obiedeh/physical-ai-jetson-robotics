"""CLI tests for core commands: profile, demo-telemetry, collect-inventory,
ops-triage, mecanum-calibration, arm-safety-demo, edge-ai-benchmark,
slam-calibration, and train-synria-reach.

Covers exit codes, output presence, JSON flags, and key field values.
No GPU required — all commands use mock/synthetic data paths.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from physical_ai_lab.cli import app

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


# ---------------------------------------------------------------------------
# profile (default command)
# ---------------------------------------------------------------------------


def test_profile_exits_zero() -> None:
    result = _invoke("profile")
    assert result.exit_code == 0, result.output


def test_profile_contains_name() -> None:
    result = _invoke("profile", "--name", "test-lab")
    assert result.exit_code == 0, result.output
    assert "test-lab" in result.output


def test_profile_custom_compute_target() -> None:
    result = _invoke("profile", "--compute", "workstation")
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# demo-telemetry
# ---------------------------------------------------------------------------


def test_demo_telemetry_exits_zero() -> None:
    result = _invoke("demo-telemetry")
    assert result.exit_code == 0, result.output


def test_demo_telemetry_json_is_valid(tmp_path: Path) -> None:
    result = _invoke("demo-telemetry", "--json")
    assert result.exit_code == 0, result.output


def test_demo_telemetry_robot_id_propagated() -> None:
    result = _invoke("demo-telemetry", "--robot-id", "test-bot-01")
    assert result.exit_code == 0, result.output
    # robot_id appears in table title
    assert "test-bot-01" in result.output


# ---------------------------------------------------------------------------
# collect-inventory
# ---------------------------------------------------------------------------


def test_collect_inventory_exits_zero(tmp_path: Path) -> None:
    out = tmp_path / "inv.json"
    result = _invoke("collect-inventory", "--output", str(out))
    assert result.exit_code == 0, result.output


def test_collect_inventory_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "inv.json"
    _invoke("collect-inventory", "--output", str(out))
    assert out.exists()


def test_collect_inventory_output_is_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "inv.json"
    _invoke("collect-inventory", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# ops-triage
# ---------------------------------------------------------------------------


def test_ops_triage_exits_zero() -> None:
    result = _invoke("ops-triage")
    assert result.exit_code == 0, result.output


def test_ops_triage_json_output_is_valid() -> None:
    result = _invoke("ops-triage", "--json")
    assert result.exit_code == 0, result.output


def test_ops_triage_json_output_contains_robot_id() -> None:
    """--json output contains the robot_id string (Rich control chars prevent full JSON parse)."""
    result = _invoke("ops-triage", "--robot-id", "synria-01", "--json")
    assert result.exit_code == 0, result.output
    assert "synria-01" in result.output


def test_ops_triage_json_output_contains_status_field() -> None:
    result = _invoke("ops-triage", "--json")
    assert result.exit_code == 0, result.output
    assert '"status"' in result.output


def test_ops_triage_json_output_contains_findings_field() -> None:
    result = _invoke("ops-triage", "--json")
    assert result.exit_code == 0, result.output
    assert '"findings"' in result.output


# ---------------------------------------------------------------------------
# mecanum-calibration
# ---------------------------------------------------------------------------


def test_mecanum_calibration_exits_zero() -> None:
    result = _invoke("mecanum-calibration")
    assert result.exit_code == 0, result.output


def test_mecanum_calibration_json_is_valid() -> None:
    result = _invoke("mecanum-calibration", "--json")
    assert result.exit_code == 0, result.output


def test_mecanum_calibration_json_has_status() -> None:
    result = _invoke("mecanum-calibration", "--json")
    raw = result.output
    data = json.loads(raw[raw.find("{"):])
    assert "status" in data


# ---------------------------------------------------------------------------
# arm-safety-demo
# ---------------------------------------------------------------------------


def test_arm_safety_demo_exits_zero() -> None:
    result = _invoke("arm-safety-demo")
    assert result.exit_code == 0, result.output


def test_arm_safety_demo_zone_left() -> None:
    result = _invoke("arm-safety-demo", "--zone", "left")
    assert result.exit_code == 0, result.output


def test_arm_safety_demo_zone_right() -> None:
    result = _invoke("arm-safety-demo", "--zone", "right")
    assert result.exit_code == 0, result.output


def test_arm_safety_demo_json_output_has_zone() -> None:
    result = _invoke("arm-safety-demo", "--zone", "top", "--json")
    assert result.exit_code == 0, result.output
    raw = result.output
    data = json.loads(raw[raw.find("{"):])
    assert data["zone"] == "top"


def test_arm_safety_demo_json_no_validation_issues() -> None:
    result = _invoke("arm-safety-demo", "--json")
    assert result.exit_code == 0, result.output
    raw = result.output
    data = json.loads(raw[raw.find("{"):])
    assert data["validation_issues"] == 0


# ---------------------------------------------------------------------------
# edge-ai-benchmark
# ---------------------------------------------------------------------------


def test_edge_ai_benchmark_exits_zero(tmp_path: Path) -> None:
    result = _invoke("edge-ai-benchmark", "--report", str(tmp_path / "bm.json"))
    assert result.exit_code == 0, result.output


def test_edge_ai_benchmark_writes_report(tmp_path: Path) -> None:
    rpt = tmp_path / "bm.json"
    _invoke("edge-ai-benchmark", "--report", str(rpt))
    assert rpt.exists()


def test_edge_ai_benchmark_report_valid_json(tmp_path: Path) -> None:
    rpt = tmp_path / "bm.json"
    _invoke("edge-ai-benchmark", "--report", str(rpt))
    data = json.loads(rpt.read_text(encoding="utf-8"))
    assert "workload_name" in data


def test_edge_ai_benchmark_json_flag(tmp_path: Path) -> None:
    result = _invoke(
        "edge-ai-benchmark",
        "--json",
        "--report", str(tmp_path / "bm.json"),
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# slam-calibration
# ---------------------------------------------------------------------------


def test_slam_calibration_exits_zero(tmp_path: Path) -> None:
    result = _invoke("slam-calibration", "--output", str(tmp_path / "slam.json"))
    assert result.exit_code == 0, result.output


def test_slam_calibration_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "slam.json"
    _invoke("slam-calibration", "--output", str(out))
    assert out.exists()


def test_slam_calibration_json_flag(tmp_path: Path) -> None:
    result = _invoke(
        "slam-calibration",
        "--json",
        "--output", str(tmp_path / "slam.json"),
    )
    assert result.exit_code == 0, result.output


def test_slam_calibration_output_has_status(tmp_path: Path) -> None:
    out = tmp_path / "slam.json"
    _invoke("slam-calibration", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "status" in data


# ---------------------------------------------------------------------------
# train-synria-reach
#
# These run the real RTX training path, which needs PyTorch. CI runners do not
# install the ml/GPU stack, so they skip there and run wherever torch exists.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="PyTorch not installed; RTX training path unavailable"
)

# ---------------------------------------------------------------------------


@requires_torch
def test_train_synria_reach_exits_zero(tmp_path: Path) -> None:
    result = _invoke(
        "train-synria-reach",
        "--samples", "512",
        "--epochs", "2",
        "--report", str(tmp_path / "train.json"),
        "--checkpoint", str(tmp_path / "model.pt"),
    )
    assert result.exit_code == 0, result.output


@requires_torch
def test_train_synria_reach_writes_report(tmp_path: Path) -> None:
    rpt = tmp_path / "train.json"
    _invoke(
        "train-synria-reach",
        "--samples", "512",
        "--epochs", "2",
        "--report", str(rpt),
        "--checkpoint", str(tmp_path / "model.pt"),
    )
    assert rpt.exists()


@requires_torch
def test_train_synria_reach_report_valid_json(tmp_path: Path) -> None:
    rpt = tmp_path / "train.json"
    _invoke(
        "train-synria-reach",
        "--samples", "512",
        "--epochs", "2",
        "--report", str(rpt),
        "--checkpoint", str(tmp_path / "model.pt"),
    )
    data = json.loads(rpt.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


@requires_torch
def test_train_synria_reach_json_flag(tmp_path: Path) -> None:
    result = _invoke(
        "train-synria-reach",
        "--json",
        "--samples", "512",
        "--epochs", "2",
        "--report", str(tmp_path / "train.json"),
        "--checkpoint", str(tmp_path / "model.pt"),
    )
    assert result.exit_code == 0, result.output

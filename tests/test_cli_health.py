"""CLI tests for the ops-copilot-health command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import Result
from typer.testing import CliRunner

from physical_ai_lab.cli import app

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def test_ops_copilot_health_exits_zero(tmp_path: Path) -> None:
    result = _invoke("ops-copilot-health", "--output", str(tmp_path / "health.json"))
    assert result.exit_code == 0, result.output


def test_ops_copilot_health_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--output", str(out))
    assert out.exists()


def test_ops_copilot_health_output_is_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_ops_copilot_health_status_is_healthy(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "healthy"


def test_ops_copilot_health_all_checks_pass(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    for check_name, passed in data["checks"].items():
        assert passed is True, f"Check {check_name!r} failed"


def test_ops_copilot_health_expected_checks_present(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("telemetry_generation", "triage", "enrichment", "serialisation"):
        assert key in data["checks"], f"Check {key!r} missing"


def test_ops_copilot_health_robot_id_in_report(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    _invoke("ops-copilot-health", "--robot-id", "synria-arm-01", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["robot_id"] == "synria-arm-01"


def test_ops_copilot_health_json_flag(tmp_path: Path) -> None:
    out = tmp_path / "health.json"
    result = _invoke("ops-copilot-health", "--json", "--output", str(out))
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "healthy"

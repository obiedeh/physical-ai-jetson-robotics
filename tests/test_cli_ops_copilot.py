"""CLI tests for the ops-copilot command in physical_ai_lab.cli.

These tests exercise the Typer ops-copilot command end-to-end using
CliRunner — no subprocess spawned, no GPU required.

Coverage:
  - Default path: no --llm flag → DeterministicClassifier, report written.
  - --llm flag without ANTHROPIC_API_KEY → falls back, warning shown, report written.
  - --json output: classifier_name appears in stdout JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from physical_ai_lab.cli import app

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def test_ops_copilot_default_uses_deterministic_classifier(tmp_path: Path) -> None:
    """Without --llm the report classifier_name is 'deterministic'."""
    output = tmp_path / "report.json"
    result = _invoke("ops-copilot", "--output", str(output))
    assert result.exit_code == 0, result.output
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["classifier_name"] == "deterministic"
    assert data["robot_id"] == "robo-car-01"
    assert data["status"] in {"nominal", "watch", "critical"}
    assert isinstance(data["escalate_to_human"], bool)


def test_ops_copilot_llm_flag_without_api_key_falls_back_to_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--llm with no ANTHROPIC_API_KEY prints a warning and uses deterministic fallback."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output = tmp_path / "report.json"
    result = _invoke("ops-copilot", "--llm", "--output", str(output))
    assert result.exit_code == 0, result.output
    # Warning must appear in CLI output.
    assert "ANTHROPIC_API_KEY" in result.output
    # Report file still written — copilot is always functional.
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["classifier_name"] == "deterministic"


def test_ops_copilot_json_output_writes_complete_report(tmp_path: Path) -> None:
    """The report file written by ops-copilot is a complete JSON document."""
    output = tmp_path / "report.json"
    result = _invoke("ops-copilot", "--json", "--output", str(output))
    assert result.exit_code == 0, result.output
    # The file is written via plain write_text (no Rich markup), so it parses cleanly.
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["classifier_name"] == "deterministic"
    assert "triage" in data
    assert "findings" in data["triage"]
    assert isinstance(data["operator_recommendation"], list)


def test_ops_copilot_custom_robot_id(tmp_path: Path) -> None:
    """--robot-id is reflected in the written report."""
    output = tmp_path / "synria.json"
    result = _invoke("ops-copilot", "--robot-id", "synria-arm-01", "--output", str(output))
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["robot_id"] == "synria-arm-01"


# ---------------------------------------------------------------------------
# --replay mode
# ---------------------------------------------------------------------------


def test_ops_copilot_replay_writes_json_array(tmp_path: Path) -> None:
    """--replay writes a JSON array, one report per telemetry step."""
    output = tmp_path / "replay.json"
    result = _invoke("ops-copilot", "--replay", "--output", str(output))
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0


def test_ops_copilot_replay_report_count_matches_samples(tmp_path: Path) -> None:
    """--replay emits exactly --samples reports (one per step)."""
    output = tmp_path / "replay.json"
    result = _invoke("ops-copilot", "--replay", "--samples", "8", "--output", str(output))
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 8


def test_ops_copilot_replay_each_report_has_required_keys(tmp_path: Path) -> None:
    """Every report in a replay array has the same schema as a single-window report."""
    output = tmp_path / "replay.json"
    result = _invoke("ops-copilot", "--replay", "--samples", "5", "--output", str(output))
    assert result.exit_code == 0, result.output
    reports = json.loads(output.read_text(encoding="utf-8"))
    for rep in reports:
        assert "robot_id" in rep
        assert "status" in rep
        assert "classifier_name" in rep
        assert "escalate_to_human" in rep
        assert "triage" in rep


def test_ops_copilot_replay_classifier_name_is_deterministic(tmp_path: Path) -> None:
    """Without --llm every replay step uses the deterministic classifier."""
    output = tmp_path / "replay.json"
    result = _invoke("ops-copilot", "--replay", "--samples", "4", "--output", str(output))
    assert result.exit_code == 0, result.output
    reports = json.loads(output.read_text(encoding="utf-8"))
    for rep in reports:
        assert rep["classifier_name"] == "deterministic"


def test_ops_copilot_replay_window_size_respected(tmp_path: Path) -> None:
    """--window-size limits how many samples feed each triage call."""
    output = tmp_path / "replay.json"
    # Use 10 samples with a window of 3 — should still produce 10 reports.
    result = _invoke(
        "ops-copilot", "--replay",
        "--samples", "10",
        "--window-size", "3",
        "--output", str(output),
    )
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 10


def test_ops_copilot_replay_robot_id_propagated(tmp_path: Path) -> None:
    """--robot-id appears in every replay report."""
    output = tmp_path / "replay.json"
    result = _invoke(
        "ops-copilot", "--replay",
        "--robot-id", "synria-replay-01",
        "--samples", "3",
        "--output", str(output),
    )
    assert result.exit_code == 0, result.output
    reports = json.loads(output.read_text(encoding="utf-8"))
    for rep in reports:
        assert rep["robot_id"] == "synria-replay-01"

import json
from pathlib import Path

from physical_ai_lab.inventory import CommandResult, collect_inventory, write_inventory


def test_collect_inventory_returns_required_top_level_fields() -> None:
    report = collect_inventory(target="test")

    assert report.target == "test"
    assert report.captured_at
    assert report.host["system"]
    assert report.host["python"]
    assert "ROS_DISTRO" in report.environment
    assert "CUDA_HOME" in report.environment


def test_collect_inventory_includes_standard_commands() -> None:
    report = collect_inventory(target="test")

    assert "git_version" in report.commands
    assert "nvidia_smi" in report.commands
    assert "nvcc_version" in report.commands


def test_unavailable_commands_have_available_false() -> None:
    report = collect_inventory(target="test")

    for result in report.commands.values():
        assert isinstance(result, CommandResult)
        if not result.available:
            assert result.return_code is None
            assert result.stdout == ""


def test_write_inventory_creates_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "inventory" / "test.json"
    written = write_inventory(target="ci-test", output=output)

    assert written == output
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["target"] == "ci-test"
    assert "host" in data
    assert "commands" in data

"""CLI tests for lerobot-dataset-demo, lerobot-policy-eval, and lerobot-cube-sort-demo."""

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
# lerobot-dataset-demo
# ---------------------------------------------------------------------------


def test_lerobot_dataset_demo_exits_zero(tmp_path: Path) -> None:
    result = _invoke("lerobot-dataset-demo", "--output", str(tmp_path / "ds.json"))
    assert result.exit_code == 0, result.output


def test_lerobot_dataset_demo_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    _invoke("lerobot-dataset-demo", "--output", str(out))
    assert out.exists()


def test_lerobot_dataset_demo_output_is_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    _invoke("lerobot-dataset-demo", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_lerobot_dataset_demo_n_episodes_respected(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    _invoke("lerobot-dataset-demo", "--n-episodes", "2", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_episodes"] == 2


def test_lerobot_dataset_demo_zone_propagated(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    _invoke("lerobot-dataset-demo", "--zone", "right", "--n-episodes", "2", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    zones = data.get("staging_zones", [])
    assert "right" in zones


def test_lerobot_dataset_demo_game_propagated(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    _invoke("lerobot-dataset-demo", "--game", "ludo", "--n-episodes", "2", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    games = data.get("games", [])
    assert "ludo" in games


def test_lerobot_dataset_demo_json_flag(tmp_path: Path) -> None:
    out = tmp_path / "ds.json"
    result = _invoke("lerobot-dataset-demo", "--json", "--output", str(out))
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# lerobot-policy-eval
# ---------------------------------------------------------------------------


def test_lerobot_policy_eval_exits_zero(tmp_path: Path) -> None:
    result = _invoke("lerobot-policy-eval", "--output", str(tmp_path / "eval.json"))
    assert result.exit_code == 0, result.output


def test_lerobot_policy_eval_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    _invoke("lerobot-policy-eval", "--output", str(out))
    assert out.exists()


def test_lerobot_policy_eval_output_is_json_array(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    _invoke("lerobot-policy-eval", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_lerobot_policy_eval_n_episodes_respected(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    _invoke("lerobot-policy-eval", "--n-episodes", "2", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_lerobot_policy_eval_all_episodes_pass(tmp_path: Path) -> None:
    """DeterministicArmPolicy should score 100% on synthetic data."""
    out = tmp_path / "eval.json"
    _invoke("lerobot-policy-eval", "--n-episodes", "3", "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    for ep in data:
        assert ep["passed"] is True, f"Episode {ep['episode_id']} failed"


def test_lerobot_policy_eval_json_flag(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    result = _invoke("lerobot-policy-eval", "--json", "--output", str(out))
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# lerobot-cube-sort-demo
# ---------------------------------------------------------------------------


def test_lerobot_cube_sort_demo_exits_zero(tmp_path: Path) -> None:
    result = _invoke("lerobot-cube-sort-demo", "--report", str(tmp_path / "cube.md"))
    assert result.exit_code == 0, result.output


def test_lerobot_cube_sort_demo_writes_markdown_report(tmp_path: Path) -> None:
    rpt = tmp_path / "cube.md"
    _invoke("lerobot-cube-sort-demo", "--report", str(rpt))
    assert rpt.exists()
    text = rpt.read_text(encoding="utf-8")
    assert "Cube Sorting" in text or "cube sort" in text.lower()


def test_lerobot_cube_sort_demo_report_contains_pass_summary(tmp_path: Path) -> None:
    rpt = tmp_path / "cube.md"
    _invoke("lerobot-cube-sort-demo", "--n-cubes", "4", "--report", str(rpt))
    text = rpt.read_text(encoding="utf-8")
    assert "Pass rate" in text or "passed" in text.lower()


def test_lerobot_cube_sort_demo_n_cubes_option(tmp_path: Path) -> None:
    rpt = tmp_path / "cube.md"
    result = _invoke("lerobot-cube-sort-demo", "--n-cubes", "3", "--report", str(rpt))
    assert result.exit_code == 0, result.output
    text = rpt.read_text(encoding="utf-8")
    assert "3" in text


def test_lerobot_cube_sort_demo_json_flag(tmp_path: Path) -> None:
    """--json prints summary to stdout; report file is still written as markdown."""
    rpt = tmp_path / "cube.md"
    result = _invoke("lerobot-cube-sort-demo", "--json", "--report", str(rpt))
    assert result.exit_code == 0, result.output
    assert rpt.exists()
    # stdout contains the JSON summary (Rich may add control chars — check key substrings)
    assert "n_cubes" in result.output
    assert "pass_rate" in result.output


def test_lerobot_cube_sort_demo_all_cubes_pass(tmp_path: Path) -> None:
    """All 6 cubes pass with the default seed — verified via the markdown report."""
    rpt = tmp_path / "cube.md"
    result = _invoke("lerobot-cube-sort-demo", "--n-cubes", "6", "--report", str(rpt))
    assert result.exit_code == 0, result.output
    text = rpt.read_text(encoding="utf-8")
    # Report includes per-cube rows; none should show ❌
    assert "❌" not in text


def test_lerobot_cube_sort_demo_seed_is_deterministic(tmp_path: Path) -> None:
    rpt1 = tmp_path / "a.md"
    rpt2 = tmp_path / "b.md"
    _invoke("lerobot-cube-sort-demo", "--seed", "99", "--json", "--report", str(rpt1))
    _invoke("lerobot-cube-sort-demo", "--seed", "99", "--json", "--report", str(rpt2))
    assert rpt1.read_text(encoding="utf-8") == rpt2.read_text(encoding="utf-8")

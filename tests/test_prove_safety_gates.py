"""Exercise the safety gate proof with its fake follower and clock; no ROS, no hardware."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.prove_safety_gates import CONFIRMATION, main, run

OPERATOR = "Test Operator"
FIXED_NOW = datetime(2026, 10, 18, 16, 7, 8, tzinfo=timezone.utc)


def test_dry_run_refuses_every_probe_and_leaves_the_follower_unchanged(tmp_path: Path) -> None:
    """All four probes refuse or clamp; the follower witness is unchanged; dry run is labelled."""
    said: list[str] = []
    report = run(
        operator=OPERATOR, dry_run=True, topic="/joint_states", gripper="50mm",
        command_period_s=0.4, response_timeout_s=0.1, max_follower_change_deg=0.5,
        output_dir=tmp_path / "reports", ask=lambda _: CONFIRMATION, say=said.append,
        now=lambda: FIXED_NOW,
    )
    assert report["all_passed"] and report["all_probes_refused"] and report["dry_run"]
    assert [p["name"] for p in report["probes"]] == [
        "joint_limit", "velocity_limit", "stale_observation", "per_cycle_step_cap",
    ]
    by_name = {p["name"]: p for p in report["probes"]}
    assert by_name["joint_limit"]["observed"]["state_reason"] == "current state outside limits"
    assert by_name["joint_limit"]["observed"]["clamped_inside_bounds"] is True
    assert by_name["stale_observation"]["observed"]["stale_reason"] == "stale or future observation"
    assert by_name["stale_observation"]["input"]["observation_age_s"] == pytest.approx(0.5)
    step = by_name["per_cycle_step_cap"]
    assert step["observed"]["within_cap"] and step["observed"]["sink_offers"] == 0
    assert step["observed"]["opt_in_refusal"]
    assert step["observed"]["policy_config_verified"] is False
    assert max(step["observed"]["step_per_joint"][:6]) == pytest.approx(0.075 * 0.4)
    assert report["follower_unchanged"] and report["max_follower_change_deg"] == 0
    assert report["command_publishers_before"] is None
    path = Path(report["report_path"])
    assert path.name == "gate_proof_dryrun_20261018T160708Z.json"
    stored = json.loads(path.read_text())
    assert stored["evidence_class"] == "dry run, not evidence"
    assert stored["coverage_note"].startswith("The per-cycle write cap")
    assert sum(line.startswith("PASS") for line in said) == 5


def test_wrong_confirmation_runs_nothing(tmp_path: Path) -> None:
    """Anything but the exact phrase stops before any probe and writes no report."""
    report = run(
        operator=OPERATOR, dry_run=True, topic="/joint_states", gripper="50mm",
        command_period_s=0.4, response_timeout_s=0.1, max_follower_change_deg=0.5,
        output_dir=tmp_path / "reports", ask=lambda _: "yes", say=lambda _: None,
    )
    assert report == {"refused": True}
    assert not (tmp_path / "reports").exists()


def test_cli_requires_the_operator_flag_and_the_typed_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """No flag is exit 2; a wrong phrase is exit 1; the dry run with the phrase is exit 0."""
    assert main(["--dry-run", "--operator", OPERATOR]) == 2
    assert main(["--i-am-the-operator", "--dry-run"]) == 2
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert main(["--i-am-the-operator", "--dry-run", "--operator", OPERATOR,
                 "--output-dir", str(tmp_path / "out")]) == 1
    monkeypatch.setattr("builtins.input", lambda _: CONFIRMATION)
    assert main(["--i-am-the-operator", "--dry-run", "--operator", OPERATOR,
                 "--output-dir", str(tmp_path / "out")]) == 0
    assert "Report written" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("gate_proof_dryrun_*.json"))

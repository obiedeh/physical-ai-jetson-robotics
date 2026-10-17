"""Exercise the limits verification script with a scripted follower; no ROS, no hardware."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.verify_limits import (
    DEFAULT_LIMITS,
    FakeFollowerSource,
    dry_run_states,
    format_table,
    main,
    run,
    stamp_limits_file,
)
from synria_lerobot.physical_contract import PhysicalState
from synria_lerobot.quality_gates import load_limits

OPERATOR = "Test Operator"
FIXED_NOW = datetime(2026, 10, 18, 15, 4, 5, tzinfo=timezone.utc)


def keys(sequence: str):
    """Return a reader that yields one key per character, then aborts."""
    iterator = iter(sequence)

    def read() -> str:
        return next(iterator, "a")

    return read


def limits_copy(tmp_path: Path) -> Path:
    """A private copy of the limits file so a test can stamp it."""
    target = tmp_path / "synria_limits.yaml"
    shutil.copy(DEFAULT_LIMITS, target)
    return target


def test_dry_run_reaches_every_bound_writes_a_dry_run_report_and_never_stamps(
    tmp_path: Path,
) -> None:
    """Fourteen marks pass every row; the report is named as a dry run; the file is untouched."""
    limits = limits_copy(tmp_path)
    before = limits.read_bytes()
    said: list[str] = []
    report = run(
        limits_path=limits, output_dir=tmp_path / "reports", operator=OPERATOR,
        gripper="50mm", topic="/joint_states", tolerance_deg=1.0, gripper_tolerance_m=0.002,
        dry_run=True, read_key=keys("m" * 14), ask=lambda _: OPERATOR, say=said.append,
        now=lambda: FIXED_NOW,
    )
    assert report["all_pass"] and report["all_joints_pass"] and report["dry_run"]
    assert report["verified_fields_written"] is False
    assert limits.read_bytes() == before
    path = Path(report["report_path"])
    assert path.name == "verification_dryrun_20261018T150405Z.json"
    stored = json.loads(path.read_text())
    assert stored["evidence_class"] == "dry run, not evidence"
    assert [row["joint"] for row in stored["joints"]] == [f"Joint{i}" for i in range(1, 7)]
    assert all(row["delta_min_deg"] == 0 and row["delta_max_deg"] == 0 for row in stored["joints"])
    grippers = {row["gripper"]: row for row in stored["grippers"]}
    assert grippers["50mm"]["pass"] is True and grippers["50mm"]["measured_open_m"] == 0.025
    assert grippers["100mm"]["pass"] is None
    assert grippers["100mm"]["note"] == "not installed, unmeasured"
    assert any("never stamped" in line for line in said)
    assert "pass/fail" in format_table(stored["joints"], stored["grippers"])


def test_a_stop_short_of_the_bound_fails_that_joint_and_withholds_the_stamp(
    tmp_path: Path,
) -> None:
    """Joint2 max two degrees short fails; a conservative Joint3 min passes; no stamp offer."""
    limits = limits_copy(tmp_path)
    loaded = load_limits(limits)
    states = dry_run_states(loaded, "50mm")
    short = list(states[3].joint_positions_rad)
    short[1] = loaded.joint_limits_rad[1][1] - math.radians(2.0)
    states[3] = PhysicalState(tuple(short), 0.0, 3.0, 3.0, ros_arrival_stamp_s=3.0)
    beyond = list(states[4].joint_positions_rad)
    beyond[2] = loaded.joint_limits_rad[2][0] - math.radians(3.0)
    states[4] = PhysicalState(tuple(beyond), 0.0, 4.0, 4.0, ros_arrival_stamp_s=4.0)
    asked: list[str] = []
    source = FakeFollowerSource(states)
    report = run(
        limits_path=limits, output_dir=tmp_path / "reports", operator=OPERATOR,
        gripper="50mm", topic="/joint_states", tolerance_deg=1.0, gripper_tolerance_m=0.002,
        dry_run=False, source=source, read_key=keys("m" * 14),
        ask=lambda text: asked.append(text) or OPERATOR, say=lambda _: None,
        now=lambda: FIXED_NOW,
    )
    rows = {row["joint"]: row for row in report["joints"]}
    assert rows["Joint2"]["pass"] is False and rows["Joint2"]["delta_max_deg"] == -2.0
    assert rows["Joint3"]["pass"] is True and rows["Joint3"]["delta_min_deg"] == -3.0
    assert report["all_pass"] is False and report["verified_fields_written"] is False
    assert asked == []
    assert source.closed
    assert Path(report["report_path"]).name == "verification_20261018T150405Z.json"


def test_full_pass_stamps_the_file_only_when_the_operator_retypes_the_name(
    tmp_path: Path,
) -> None:
    """A wrong confirmation leaves the file alone; the exact name rewrites only two fields."""
    limits = limits_copy(tmp_path)
    original = limits.read_text()
    common = dict(
        limits_path=limits, output_dir=tmp_path / "reports", operator=OPERATOR, gripper="50mm",
        topic="/joint_states", tolerance_deg=1.0, gripper_tolerance_m=0.002, dry_run=False,
        say=lambda _: None, now=lambda: FIXED_NOW,
    )
    declined = run(
        **common, source=FakeFollowerSource(dry_run_states(load_limits(limits), "50mm")),
        read_key=keys("m" * 14), ask=lambda _: "someone else",
    )
    assert declined["all_pass"] and declined["verified_fields_written"] is False
    assert limits.read_text() == original
    accepted = run(
        **common, source=FakeFollowerSource(dry_run_states(load_limits(limits), "50mm")),
        read_key=keys("m" * 14), ask=lambda _: OPERATOR,
    )
    assert accepted["verified_fields_written"] is True
    stamped = load_limits(limits)
    assert stamped.verified_by == OPERATOR and stamped.verified_on == "2026-10-18"
    changed = [
        (old, new) for old, new in zip(original.splitlines(), limits.read_text().splitlines(),
                                       strict=True) if old != new
    ]
    assert len(changed) == 2
    assert accepted["limits_sha256_after"] != accepted["limits_sha256"]


def test_skip_and_abort_keys_are_honoured(tmp_path: Path) -> None:
    """A skipped side fails its joint; an abort writes nothing at all."""
    limits = limits_copy(tmp_path)
    skipped = run(
        limits_path=limits, output_dir=tmp_path / "reports", operator=OPERATOR,
        gripper="100mm", topic="/joint_states", tolerance_deg=1.0, gripper_tolerance_m=0.002,
        dry_run=True, read_key=keys("rs" + "m" * 13), ask=lambda _: OPERATOR,
        say=lambda _: None, now=lambda: FIXED_NOW,
    )
    assert skipped["joints"][0]["measured_min_deg"] is None
    assert skipped["joints"][0]["pass"] is False and skipped["all_pass"] is False
    assert {row["gripper"]: row["pass"] for row in skipped["grippers"]} == {
        "50mm": None, "100mm": True,
    }
    aborted = run(
        limits_path=limits, output_dir=tmp_path / "empty", operator=OPERATOR,
        gripper="50mm", topic="/joint_states", tolerance_deg=1.0, gripper_tolerance_m=0.002,
        dry_run=True, read_key=keys("mma"), ask=lambda _: OPERATOR, say=lambda _: None,
    )
    assert aborted == {"aborted": True}
    assert not (tmp_path / "empty").exists()


def test_stamp_helper_refuses_ambiguous_files_and_main_requires_an_operator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The stamp touches exactly one occurrence of each field; the CLI insists on a name."""
    odd = tmp_path / "odd.yaml"
    odd.write_text('{"verified_by": "", "verified_by": ""}')
    with pytest.raises(ValueError, match="exactly once"):
        stamp_limits_file(odd, OPERATOR, "2026-10-18")
    with pytest.raises(SystemExit):
        main([])
    code = main([
        "--operator", OPERATOR, "--dry-run", "--limits", str(limits_copy(tmp_path)),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == 0
    assert "Report written" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("verification_dryrun_*.json"))

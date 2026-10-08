from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "synria_arm_bringup"
sys.path.insert(0, str(PACKAGE_ROOT))

from synria_arm_bringup.validation import validate_leader_port  # noqa: E402


def test_package_metadata_and_launch_files_exist() -> None:
    root = ElementTree.parse(PACKAGE_ROOT / "package.xml").getroot()
    assert root.findtext("name") == "synria_arm_bringup"
    assert (PACKAGE_ROOT / "launch" / "leader_state.launch.py").is_file()


def test_leader_port_has_no_unstable_or_empty_fallback() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_leader_port("")
    with pytest.raises(ValueError, match="/dev/serial/by-id"):
        validate_leader_port("/dev/ttyACM0")
    assert validate_leader_port("/dev/serial/by-id/usb-leader") == (
        "/dev/serial/by-id/usb-leader"
    )


def test_launch_is_isolated_and_read_only() -> None:
    source = (PACKAGE_ROOT / "launch" / "leader_state.launch.py").read_text()
    assert source.count('package="alicia_d_driver"') == 1
    assert '"joint_commands_enabled": False' in source
    assert '"joint_commands_dry_run": False' in source
    assert '"torque_off_on_start": False' in source
    assert '"allow_leader_writes": False' in source
    assert '("/joint_states", "/leader/joint_states")' in source
    assert '("/joint_commands", "/leader/disabled_joint_commands")' in source
    assert "follower" not in source.lower()


def test_launch_reports_missing_external_driver_clearly() -> None:
    source = (PACKAGE_ROOT / "launch" / "leader_state.launch.py").read_text()
    assert "Required external package 'alicia_d_driver' was not found" in source

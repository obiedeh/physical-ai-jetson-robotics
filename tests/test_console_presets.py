"""Verify documented form suggestions without selecting or opening real hardware."""

from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).parents[1] / "synria_lerobot" / "console" / "static"


class FormMarkup(HTMLParser):
    """Read named fields and suggested values from shipped HTML, not a duplicate fixture."""

    def __init__(self) -> None:
        """Collect the minimal structural facts needed to protect operator choices."""
        super().__init__()
        self.fields: dict[str, dict[str, str | None]] = {}
        self.options: dict[str, list[str | None]] = {}
        self.group = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track input constraints and option values while ignoring decorative markup."""
        values = dict(attrs)
        if tag in {"input", "select"} and values.get("name"):
            self.fields[str(values["name"])] = values
        if tag in {"select", "datalist"}:
            self.group = str(values.get("name") or values["id"])
            self.options[self.group] = []
        if tag == "option" and self.group:
            self.options[self.group].append(values.get("value"))

    def handle_endtag(self, tag: str) -> None:
        """End each suggestion group so unrelated options cannot be counted together."""
        if tag in {"select", "datalist"}:
            self.group = ""


@pytest.fixture
def form() -> FormMarkup:
    """Parse the packaged page without running a server or discovering devices."""
    parsed = FormMarkup()
    parsed.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return parsed


@pytest.mark.parametrize(("field", "choices", "default"), [
    ("fps", ["15", "30"], None),
    ("image_width", ["224"], "224"),
    ("image_height", ["224"], "224"),
    ("action_lookahead_steps", ["1"], "1"),
    ("state_startup_timeout_s", ["10"], "10"),
    ("target_episodes", ["10", "50", "100"], "100"),
    ("follower_topic", ["/joint_states"], "/joint_states"),
    ("leader_topic", ["/leader/joint_states"], "/leader/joint_states"),
])
def test_presets_preserve_documented_values_and_defaults(
    form: FormMarkup, field: str, choices: list[str], default: str | None,
) -> None:
    """Suggestions match the reference runbook and never silently select a recording rate."""
    values = form.fields[field]
    assert "data-presets" in values
    assert form.options[str(values["list"])] == choices
    assert values.get("value") == default


def test_unknown_hardware_and_task_windows_are_not_assumed(form: FormMarkup) -> None:
    """Installed gripper, declared source, identities and physical timing remain operator facts."""
    assert form.options["gripper_type"] == ["", "50mm", "100mm"]
    assert form.options["state_source"] == ["", "standalone_driver", "ros2_control"]
    assert form.options["action_source"] == ["next_state", "leader"]
    for name in ("wrist_camera", "front_camera"):
        assert form.options[name] == []
    for name in ("follower_serial", "leader_serial", "power_state_start"):
        assert "value" not in form.fields[name]
    for name in ("fps", "gripper_type"):
        assert "required" in form.fields[name]
    assert "min_episode_s" not in form.fields
    assert "max_episode_s" not in form.fields
    assert "checked" not in form.fields["state_has_velocity"]
    assert form.options["follower_usb_id"] == [""]


def test_preset_script_preserves_custom_inputs_and_form_payload() -> None:
    """Exercise shipped dropdown behavior against deterministic DOM fakes with no device access."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    subprocess.run(
        [executable, str(Path(__file__).with_name("console_form_presets.cjs")),
         str(STATIC / "form-presets.js"), str(STATIC / "app.js")],
        check=True, timeout=15, capture_output=True, text=True,
    )


def test_setup_candidates_never_fill_unknown_physical_facts() -> None:
    """Check automatic suggestions, explicit overrides and USB hints against browser fakes."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    subprocess.run(
        [executable, str(Path(__file__).with_name("console_setup_ui.cjs")),
         str(STATIC / "serial-identity.js")],
        check=True, timeout=15, capture_output=True, text=True,
    )

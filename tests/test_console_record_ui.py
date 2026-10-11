"""Verify the Record screen's plain-language markup and banner states with fakes only."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "synria_lerobot" / "console" / "static"


def test_record_screen_markup_names_actions_and_hides_raw_source_json() -> None:
    """Buttons say what they do, the banner exists, and camera source JSON sits behind a toggle."""
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    for label in (
        "Start recording (Space)", "Stop (Space)", "Yes, save as success (S)",
        "No, save as failure (F)", "Retry save", "Throw away, bad capture", "Cancel countdown",
        "Pause session", "Finish session", "Review last take", "Reset the scene",
    ):
        assert label in markup
    for stale in ("Start · Space", "Success · S", "Failure · F", "Close sources",
                  'id="capture-header"',
                  "Save as success or Save as failure is what saves the episode."):
        assert stale not in markup
    assert 'id="record-banner"' in markup and 'id="record-message"' in markup
    assert 'id="record-countdown-value"' in markup and 'id="record-banner-action"' in markup
    assert 'id="counters" class="counters"' in markup
    assert 'id="wrist-source"' in markup and 'id="front-source"' in markup
    assert '<summary>Source details</summary>' in markup
    assert '<script src="/record-banner.js" defer></script>' in markup
    assert 'id="countdown" type="number" min="0" max="30" value="3"' in markup
    assert 'id="audio" type="checkbox" checked' in markup
    assert "Wrist camera <span" in markup and "Overhead camera <span" in markup
    assert 'id="record-technical"' in markup and 'id="reset-scene"' in markup
    for command in ("start", "stop", "success", "failure", "retry", "cancel_countdown"):
        assert f'data-command="{command}"' in markup


def test_record_banner_states_are_plain_language() -> None:
    """Every banner state carries its single next action, verified through the fake DOM module."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    result = subprocess.run(
        [executable, str(Path(__file__).with_name("console_record_ui.cjs")),
         str(STATIC / "record-banner.js")],
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert "record banner states verified" in result.stdout
    for script in ("app.js", "record-banner.js"):
        subprocess.run([executable, "--check", str(STATIC / script)], check=True, timeout=15)

"""Validate library environment preparation with inert shell fixtures, never ROS or devices."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["physical", "demo", "missing"])
def test_launcher_sources_only_the_selected_library_environment(
    tmp_path: Path, mode: str,
) -> None:
    """Physical launch sources a configured setup, demo skips it, and missing setup is explicit."""
    shell = shutil.which("bash")
    if shell is None or os.name == "nt":
        pytest.skip("RTX shell launcher requires a POSIX host")
    environment = tmp_path / "recording"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/activate").write_text(
        "# Inert activation fixture; executable search path is supplied by the test.\n",
        encoding="utf-8",
    )
    executable = environment / "bin/python"
    executable.write_text('#!/usr/bin/env bash\nprintf "fake console: %s\\n" '
                          '"${FAKE_ROS_SETUP:-unset}"\nprintf "arguments: %s\\n" "$*"\n',
                          encoding="utf-8")
    executable.chmod(0o755)
    setup = tmp_path / "setup.bash"
    if mode != "missing":
        setup.write_text("export FAKE_ROS_SETUP=prepared\n", encoding="utf-8")
    launcher = Path(__file__).parents[1] / "scripts/linux_rtx/synria_teleop_console.sh"
    result = subprocess.run(
        [shell, str(launcher), *(["--demo"] if mode == "demo" else [])],
        env={**os.environ, "RECORDING_VENV": str(environment),
             "RECORDING_ROS_SETUP": str(setup), "FAKE_ROS_SETUP": "unset",
             "PATH": str(environment / "bin") + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert "fake console: " + ("prepared" if mode == "physical" else "unset") in result.stdout
    assert "arguments: -m synria_lerobot.console.app" in result.stdout
    assert ("Launching:" in result.stdout) is (mode != "demo")
    assert ("Not ready:" in result.stderr) is (mode == "missing")

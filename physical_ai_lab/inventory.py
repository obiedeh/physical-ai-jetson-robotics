"""Hardware and software inventory collection for lab machines."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Captured output from a local command."""

    command: list[str]
    available: bool
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class InventoryReport:
    """Machine inventory report for Windows, Linux RTX, and Jetson targets."""

    target: str
    captured_at: str
    host: dict[str, str]
    environment: dict[str, str | None]
    commands: dict[str, CommandResult] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _run(command: list[str], timeout_seconds: int = 15) -> CommandResult:
    executable = command[0]
    if shutil.which(executable) is None:
        return CommandResult(command=command, available=False)

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        return CommandResult(
            command=command,
            available=True,
            return_code=None,
            stdout=stdout or "",
            stderr=f"Command timed out after {timeout_seconds}s",
        )

    return CommandResult(
        command=command,
        available=True,
        return_code=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def collect_inventory(target: str) -> InventoryReport:
    """Collect inventory data without requiring ROS, CUDA, or Jetson packages."""
    commands = {
        "git_version": _run(["git", "--version"]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvcc_version": _run(["nvcc", "--version"]),
        "ros2_version": _run(["ros2", "--version"]),
        "ros2_topic_list": _run(["ros2", "topic", "list"], timeout_seconds=20),
        "tegrastats": _run(["tegrastats"], timeout_seconds=5),
    }

    if platform.system().lower() == "windows":
        commands["windows_camera_devices"] = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
            ]
        )
        commands["windows_usb_devices"] = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -PresentOnly | "
                "Where-Object {$_.InstanceId -match '^USB'} | "
                "Select-Object -First 50 -ExpandProperty FriendlyName",
            ]
        )
    else:
        commands["linux_video_devices"] = _run(["bash", "-lc", "ls -1 /dev/video* 2>/dev/null"])
        commands["linux_usb_devices"] = _run(["lsusb"])
        commands["linux_release"] = _run(["cat", "/etc/os-release"])

    return InventoryReport(
        target=target,
        captured_at=datetime.now(timezone.utc).isoformat(),
        host={
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.replace("\n", " "),
        },
        environment={
            "ROS_DISTRO": os.environ.get("ROS_DISTRO"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PATH": os.environ.get("PATH"),
        },
        commands=commands,
    )


def write_inventory(target: str, output: Path) -> Path:
    """Collect inventory and write it to a JSON file."""
    report = collect_inventory(target=target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    return output

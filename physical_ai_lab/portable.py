"""Portable identifiers for evidence files: board labels, path tokens and address tokens.

Evidence, logs and scripts in this repository name machines by board label
(``rtx-5090-workstation``, ``bench-thor``, ``field-orin``), write ``<home>`` for the
account's home directory and ``<lan-address>`` for any private-range address. Nothing
here knows a host name; the label comes from the environment or the hardware.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

HOME_TOKEN = "<home>"
LAN_TOKEN = "<lan-address>"
BOARD_ENV = "PHYSICAL_AI_BOARD"
BOARD_LABELS = ("rtx-5090-workstation", "bench-thor", "field-orin")
UNKNOWN_BOARD = "unknown-board"

HOME_RE = re.compile(r"/home/[A-Za-z0-9_-]+")

WINDOWS_HOME_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]Users[\\/][^\\/\r\n`\'\"<>]+",
    re.IGNORECASE,
)
PRIVATE_ADDRESS_RE = re.compile(
    r"(?<![0-9.])(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
    r"|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}"
    r"|192\.168\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9.])"
)
_DEVICE_TREE_MODEL = Path("/proc/device-tree/model")


def _gpu_name() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def board_label(environ: dict[str, str] | None = None,
                device_model: Callable[[], str] | None = None,
                gpu_name: Callable[[], str] = _gpu_name) -> str:
    """The board this process runs on, as a label; never a host name.

    ``PHYSICAL_AI_BOARD`` wins when set to a known label. Otherwise a Jetson names
    itself through the device-tree model (Thor or Orin), and a machine with an RTX 5090
    is the workstation. Anything else is ``unknown-board``.
    """
    env = os.environ if environ is None else environ
    label = env.get(BOARD_ENV, "").strip()
    if label in BOARD_LABELS:
        return label
    if device_model is None:
        def device_model() -> str:
            try:
                return _DEVICE_TREE_MODEL.read_text(errors="ignore")
            except OSError:
                return ""
    model = device_model().lower()
    if "thor" in model:
        return "bench-thor"
    if "orin" in model:
        return "field-orin"
    if "5090" in gpu_name():
        return "rtx-5090-workstation"
    return UNKNOWN_BOARD


def portable_text(text: str) -> str:
    """Normalize home metadata while retaining unrelated text and path semantics."""
    if WINDOWS_HOME_RE.match(text):
        text = WINDOWS_HOME_RE.sub(HOME_TOKEN, text, count=1).replace("\\", "/")
    else:
        text = WINDOWS_HOME_RE.sub(HOME_TOKEN, text)
    return PRIVATE_ADDRESS_RE.sub(LAN_TOKEN, HOME_RE.sub(HOME_TOKEN, text))


def portable_path(path: Path | str) -> str:
    """A path as text with the home directory written as ``<home>``."""
    return portable_text(str(path))


def portable_json(value: Any) -> Any:
    """The same JSON-like value with every string made portable; other types untouched."""
    if isinstance(value, str):
        return portable_text(value)
    if isinstance(value, dict):
        return {key: portable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_json(item) for item in value]
    return value


def expand_path(value: str | Path) -> Path:
    """A recorded path back to this account: ``<home>`` becomes the home directory."""
    text = str(value)
    if text.startswith(HOME_TOKEN):
        text = str(Path.home()) + text[len(HOME_TOKEN):]
    return Path(text)


def expand_json(value: Any) -> Any:
    """The same JSON-like value with every leading ``<home>`` written back as the home directory."""
    if isinstance(value, str):
        return str(Path.home()) + value[len(HOME_TOKEN):] if value.startswith(HOME_TOKEN) else value
    if isinstance(value, dict):
        return {key: expand_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_json(item) for item in value]
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """``json.dumps`` of the portable form of ``value``."""
    return json.dumps(portable_json(value), **kwargs)


def dump(value: Any, handle: Any, **kwargs: Any) -> None:
    """``json.dump`` of the portable form of ``value``."""
    json.dump(portable_json(value), handle, **kwargs)

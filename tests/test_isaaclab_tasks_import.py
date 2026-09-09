"""Static parse + Windows-safe import tests for the Isaac Lab task scaffold.

These tests confirm the package imports without Isaac Lab present (so that
ruff / mypy / static-validation CI on Windows + Ubuntu pass), without
attempting any Isaac Lab runtime calls. Runtime validation happens on the
RTX 5090 with Isaac Lab installed; that's covered by `docs/SYNRIA_PICK_AND_PLACE_TASK.md`
and the package's own README.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "isaac" / "isaaclab_tasks" / "synria_pickplace"


def test_isaaclab_task_files_exist() -> None:
    expected = [
        "__init__.py",
        "env_cfg.py",
        "mdp.py",
        "mdp_logic.py",
        "task_geometry.py",
        "trial_state.py",
        "README.md",
    ]
    missing = [name for name in expected if not (PKG / name).exists()]
    assert missing == []


def test_isaaclab_task_modules_parse() -> None:
    parseable = (
        "__init__.py",
        "env_cfg.py",
        "mdp.py",
        "mdp_logic.py",
        "task_geometry.py",
        "trial_state.py",
    )
    for name in parseable:
        text = (PKG / name).read_text(encoding="utf-8")
        # ast.parse raises SyntaxError on malformed Python.
        ast.parse(text, filename=str(PKG / name))


def test_synria_pickplace_imports_without_isaaclab() -> None:
    """The package must import on Windows / any env without Isaac Lab."""
    import importlib
    import sys

    # Ensure we use the on-disk module, not a stale cached copy.
    for mod_name in list(sys.modules):
        if mod_name.startswith("isaac.isaaclab_tasks"):
            del sys.modules[mod_name]

    # Make `isaac/` discoverable on sys.path so `from isaac.isaaclab_tasks...`
    # can resolve without requiring an editable install.
    isaac_parent = str(REPO_ROOT)
    if isaac_parent not in sys.path:
        sys.path.insert(0, isaac_parent)

    # Importing should succeed even without isaaclab / gymnasium installed.
    module = importlib.import_module("isaac.isaaclab_tasks.synria_pickplace")
    # The three config-class names must be exported by __all__.
    for name in (
        "SynriaLudoPickPlaceEnvCfg",
        "SynriaChessPickPlaceEnvCfg",
        "SynriaCheckersPickPlaceEnvCfg",
    ):
        assert hasattr(module, name), f"missing export: {name}"


def test_pure_python_helpers_importable() -> None:
    """task_geometry, trial_state, mdp_logic must import without Isaac Lab or torch."""
    import importlib
    import sys

    for mod_name in list(sys.modules):
        if mod_name.startswith("isaac.isaaclab_tasks"):
            del sys.modules[mod_name]

    isaac_parent = str(REPO_ROOT)
    if isaac_parent not in sys.path:
        sys.path.insert(0, isaac_parent)

    # These three modules must be importable anywhere.
    for submod in ("task_geometry", "trial_state", "mdp_logic"):
        mod = importlib.import_module(f"isaac.isaaclab_tasks.synria_pickplace.{submod}")
        assert mod is not None, f"failed to import {submod}"


def test_task_geometry_zone_center_sanity() -> None:
    """Quick sanity on zone_center_xy to confirm geometry constants loaded."""
    import sys

    isaac_parent = str(REPO_ROOT)
    if isaac_parent not in sys.path:
        sys.path.insert(0, isaac_parent)

    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        BOARD_CENTER_X,
        BOARD_CENTER_Y,
        zone_center_xy,
    )

    # Left and right zones should have the same X as board centre.
    xl, _ = zone_center_xy("left", "chess")
    xr, _ = zone_center_xy("right", "chess")
    assert abs(xl - BOARD_CENTER_X) < 1e-9
    assert abs(xr - BOARD_CENTER_X) < 1e-9

    # Top and bottom zones should have the same Y as board centre.
    _, yt = zone_center_xy("top", "chess")
    _, yb = zone_center_xy("bottom", "chess")
    assert abs(yt - BOARD_CENTER_Y) < 1e-9
    assert abs(yb - BOARD_CENTER_Y) < 1e-9

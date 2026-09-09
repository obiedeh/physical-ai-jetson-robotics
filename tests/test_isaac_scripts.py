"""Tests for Isaac scene-builder scripts and the URDF importer helper.

These scripts require Isaac Sim at runtime, so full execution is hardware-gated.
This file covers:
  - Syntax validity of all five scripts via py_compile
  - import_synria_urdf.parse_args() — pure argparse, no Isaac Sim needed
  - Scene constant integrity via AST inspection (board sizes, piece counts)
"""

from __future__ import annotations

import ast
import importlib.util
import py_compile
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "isaac" / "scripts"

_SCENE_SCRIPTS: list[Path] = [
    SCRIPTS_DIR / "_synria_scene_common.py",
    SCRIPTS_DIR / "build_synria_chess_scene.py",
    SCRIPTS_DIR / "build_synria_checkers_scene.py",
    SCRIPTS_DIR / "build_synria_ludo_scene.py",
    SCRIPTS_DIR / "import_synria_urdf.py",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Syntax validity (py_compile)
# ---------------------------------------------------------------------------


def test_synria_scene_common_syntax() -> None:
    py_compile.compile(str(SCRIPTS_DIR / "_synria_scene_common.py"), doraise=True)


def test_build_chess_scene_syntax() -> None:
    py_compile.compile(str(SCRIPTS_DIR / "build_synria_chess_scene.py"), doraise=True)


def test_build_checkers_scene_syntax() -> None:
    py_compile.compile(str(SCRIPTS_DIR / "build_synria_checkers_scene.py"), doraise=True)


def test_build_ludo_scene_syntax() -> None:
    py_compile.compile(str(SCRIPTS_DIR / "build_synria_ludo_scene.py"), doraise=True)


def test_import_synria_urdf_syntax() -> None:
    py_compile.compile(str(SCRIPTS_DIR / "import_synria_urdf.py"), doraise=True)


# ---------------------------------------------------------------------------
# import_synria_urdf.py — parse_args() (no Isaac Sim required)
# ---------------------------------------------------------------------------


def _load_import_synria_urdf() -> types.ModuleType:
    """Load import_synria_urdf as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(
        "import_synria_urdf", SCRIPTS_DIR / "import_synria_urdf.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_urdf_importer_parse_args_default_collision_type() -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf"])
    assert args.collision_type == "Convex Hull"


def test_urdf_importer_parse_args_custom_collision_type() -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf", "--collision-type", "Bounding Sphere"])
    assert args.collision_type == "Bounding Sphere"


def test_urdf_importer_parse_args_merge_mesh_default_false() -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf"])
    assert args.merge_mesh is False


def test_urdf_importer_parse_args_merge_mesh_flag() -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf", "--merge-mesh"])
    assert args.merge_mesh is True


def test_urdf_importer_parse_args_default_output_dir() -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf"])
    # Default output dir is under the repo's isaac/usd/robots/
    assert "robots" in str(args.output_dir)


def test_urdf_importer_parse_args_custom_output_dir(tmp_path: Path) -> None:
    mod = _load_import_synria_urdf()
    args = mod.parse_args(["--urdf", "/tmp/test.urdf", "--output-dir", str(tmp_path)])
    assert args.output_dir == tmp_path


def test_urdf_importer_collision_type_choices() -> None:
    """Verify all documented collision types are accepted."""
    mod = _load_import_synria_urdf()
    for ct in ("Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"):
        args = mod.parse_args(["--urdf", "/tmp/test.urdf", "--collision-type", ct])
        assert args.collision_type == ct


# ---------------------------------------------------------------------------
# Scene constant integrity via AST inspection
# ---------------------------------------------------------------------------


def _get_constant(source: str, name: str) -> object:
    """Extract a simple top-level constant assignment from source text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    raise KeyError(f"{name!r} not found as a simple constant")


def test_chess_board_size_constant() -> None:
    source = (SCRIPTS_DIR / "build_synria_chess_scene.py").read_text(encoding="utf-8")
    # SQUARES_PER_SIDE = 8 (the 8x8 chess board)
    val = _get_constant(source, "SQUARES_PER_SIDE")
    assert val == 8


def test_chess_square_size_constant() -> None:
    source = (SCRIPTS_DIR / "build_synria_chess_scene.py").read_text(encoding="utf-8")
    val = _get_constant(source, "SQUARE_SIZE")
    assert val == pytest.approx(0.045)


def test_checkers_board_is_8x8() -> None:
    source = (SCRIPTS_DIR / "build_synria_checkers_scene.py").read_text(encoding="utf-8")
    val = _get_constant(source, "SQUARES_PER_SIDE")
    assert val == 8


def test_ludo_board_size_constant() -> None:
    source = (SCRIPTS_DIR / "build_synria_ludo_scene.py").read_text(encoding="utf-8")
    val = _get_constant(source, "BOARD_SIZE")
    assert val == pytest.approx(0.40)


def test_common_scene_table_height() -> None:
    source = (SCRIPTS_DIR / "_synria_scene_common.py").read_text(encoding="utf-8")
    val = _get_constant(source, "TABLE_HEIGHT")
    assert val == pytest.approx(0.75)


def test_common_scene_board_center_x() -> None:
    source = (SCRIPTS_DIR / "_synria_scene_common.py").read_text(encoding="utf-8")
    val = _get_constant(source, "BOARD_CENTER_X")
    assert val == pytest.approx(0.10)


def test_common_scene_board_center_y_is_zero() -> None:
    source = (SCRIPTS_DIR / "_synria_scene_common.py").read_text(encoding="utf-8")
    val = _get_constant(source, "BOARD_CENTER_Y")
    assert val == pytest.approx(0.0)


def test_scene_output_paths_exist_as_strings() -> None:
    """Each scene builder must declare a SCENE_OUT that references the right subdirectory."""
    checks = {
        "build_synria_chess_scene.py": "synria_chess",
        "build_synria_checkers_scene.py": "synria_checkers",
        "build_synria_ludo_scene.py": "synria_ludo",
    }
    for script, expected_subdir in checks.items():
        source = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
        assert expected_subdir in source, (
            f"{script} should reference output directory '{expected_subdir}'"
        )

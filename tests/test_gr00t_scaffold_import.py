"""Windows-safe import + structural tests for the GR00T integration scaffold."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "isaac" / "isaaclab_tasks" / "synria_pickplace" / "gr00t"


def test_gr00t_scaffold_files_exist() -> None:
    expected = [
        "__init__.py",
        "embodiment.py",
        "dataset.py",
        "finetune.py",
        "inference.py",
        "README.md",
    ]
    missing = [name for name in expected if not (PKG / name).exists()]
    assert missing == []


def test_gr00t_scaffold_modules_parse() -> None:
    for name in ("__init__.py", "embodiment.py", "dataset.py", "finetune.py", "inference.py"):
        text = (PKG / name).read_text(encoding="utf-8")
        ast.parse(text, filename=str(PKG / name))


def test_gr00t_scaffold_imports_without_gr00t() -> None:
    """The scaffold must import cleanly without GR00T installed."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for mod_name in list(sys.modules):
        if mod_name.startswith("isaac.isaaclab_tasks.synria_pickplace.gr00t"):
            del sys.modules[mod_name]

    module = importlib.import_module("isaac.isaaclab_tasks.synria_pickplace.gr00t")
    for name in ("SYNRIA_EMBODIMENT_TAG", "SynriaEmbodimentConfig", "synria_modality_config"):
        assert hasattr(module, name), f"missing export: {name}"


def test_synria_modality_config_shape() -> None:
    """The modality config must contain the expected top-level keys + action dims."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import (
        SYNRIA_ACTION_DIM,
        synria_modality_config,
    )

    cfg = synria_modality_config()
    assert cfg["embodiment_tag"] == "new_embodiment"
    assert {"state", "vision", "language", "action"} <= set(cfg)
    # Action head dims should sum to SYNRIA_ACTION_DIM (6 arm joints + 2 fingers = 8).
    action_dims = sum(spec["shape"][0] for spec in cfg["action"].values())
    assert action_dims == SYNRIA_ACTION_DIM


def test_dataset_paths_resolve_under_reports_training() -> None:
    """The on-disk dataset path conventions should land under reports/training/."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import (
        SYNRIA_DEMO_ROOTS,
        SYNRIA_MIMIC_AMPLIFIED_ROOTS,
    )

    for path in (*SYNRIA_DEMO_ROOTS.values(), *SYNRIA_MIMIC_AMPLIFIED_ROOTS.values()):
        # Use as_posix for cross-platform substring matching (Windows uses backslashes).
        assert "reports/training" in path.as_posix(), f"unexpected path: {path}"

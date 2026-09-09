"""Unit tests for the GR00T integration scaffold.

Covers embodiment constants, dataset path conventions, finetune CLI parser
and dry-run execution, and inference CLI parser. All tests run without GR00T,
Isaac Sim, or any hardware — pure Python scaffold verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_gr00t_modules() -> None:
    """Remove cached gr00t submodule imports so tests are hermetic."""
    prefix = "isaac.isaaclab_tasks.synria_pickplace.gr00t"
    for key in list(sys.modules):
        if key.startswith(prefix):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# embodiment.py — constants and config
# ---------------------------------------------------------------------------


def test_embodiment_tag_is_stable() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_EMBODIMENT_TAG

    assert SYNRIA_EMBODIMENT_TAG == "new_embodiment"


def test_arm_joint_count_is_six() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_ARM_JOINT_NAMES

    assert len(SYNRIA_ARM_JOINT_NAMES) == 6


def test_arm_joint_names_match_urdf() -> None:
    """Joint names must align with the SolidWorks-export URDF (Joint1..Joint6)."""
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_ARM_JOINT_NAMES

    assert SYNRIA_ARM_JOINT_NAMES == ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


def test_gripper_joint_count_is_two() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_GRIPPER_JOINT_NAMES

    assert len(SYNRIA_GRIPPER_JOINT_NAMES) == 2


def test_gripper_joint_names() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_GRIPPER_JOINT_NAMES

    assert "left_finger" in SYNRIA_GRIPPER_JOINT_NAMES
    assert "right_finger" in SYNRIA_GRIPPER_JOINT_NAMES


def test_action_dim_is_eight() -> None:
    """6 arm joints + 2 gripper fingers = 8-float action space."""
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_ACTION_DIM

    assert SYNRIA_ACTION_DIM == 8


def test_camera_keys_count() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_CAMERA_KEYS

    assert len(SYNRIA_CAMERA_KEYS) == 2


def test_camera_keys_include_wrist_and_overhead() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_CAMERA_KEYS

    assert "wrist" in SYNRIA_CAMERA_KEYS
    assert "overhead" in SYNRIA_CAMERA_KEYS


def test_camera_resolution_is_gr00t_standard() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_CAMERA_RES

    assert SYNRIA_CAMERA_RES == (224, 224)


def test_embodiment_config_fields() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SynriaEmbodimentConfig

    cfg = SynriaEmbodimentConfig()
    assert cfg.embodiment_tag == "new_embodiment"
    assert cfg.action_dim == 8
    assert len(cfg.arm_joints) == 6
    assert len(cfg.gripper_joints) == 2
    assert len(cfg.camera_keys) == 2
    assert cfg.image_size == (224, 224)


def test_modality_config_top_level_keys() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import synria_modality_config

    cfg = synria_modality_config()
    assert {"embodiment_tag", "state", "vision", "language", "action"} <= set(cfg)


def test_modality_config_state_keys() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import synria_modality_config

    state = synria_modality_config()["state"]
    assert "arm_joint_pos" in state
    assert "arm_joint_vel" in state
    assert "gripper_width" in state


def test_modality_config_vision_cameras_match_camera_keys() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import (
        SYNRIA_CAMERA_KEYS,
        synria_modality_config,
    )

    vision = synria_modality_config()["vision"]
    assert set(vision.keys()) == set(SYNRIA_CAMERA_KEYS)


def test_modality_config_action_dims_sum_to_action_dim() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import (
        SYNRIA_ACTION_DIM,
        synria_modality_config,
    )

    action = synria_modality_config()["action"]
    total = sum(spec["shape"][0] for spec in action.values())
    assert total == SYNRIA_ACTION_DIM


def test_modality_config_language_instruction_key() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import synria_modality_config

    lang = synria_modality_config()["language"]
    assert "instruction" in lang


def test_modality_config_is_json_serialisable() -> None:
    import json

    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import synria_modality_config

    s = json.dumps(synria_modality_config())
    assert len(s) > 0


# ---------------------------------------------------------------------------
# dataset.py — path conventions and error handling
# ---------------------------------------------------------------------------


def test_demo_roots_has_three_games() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_DEMO_ROOTS

    assert set(SYNRIA_DEMO_ROOTS) == {"ludo", "chess", "checkers"}


def test_mimic_amplified_roots_has_three_games() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_MIMIC_AMPLIFIED_ROOTS

    assert set(SYNRIA_MIMIC_AMPLIFIED_ROOTS) == {"ludo", "chess", "checkers"}


def test_recommended_seed_demos() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import RECOMMENDED_SEED_DEMOS_PER_GAME

    assert RECOMMENDED_SEED_DEMOS_PER_GAME == 100


def test_expected_mimic_amplification() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import EXPECTED_MIMIC_AMPLIFICATION

    assert EXPECTED_MIMIC_AMPLIFICATION == 1000


def test_demo_roots_under_reports_training() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_DEMO_ROOTS

    for path in SYNRIA_DEMO_ROOTS.values():
        assert "reports/training" in path.as_posix()


def test_mimic_amplified_roots_under_reports_training() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_MIMIC_AMPLIFIED_ROOTS

    for path in SYNRIA_MIMIC_AMPLIFIED_ROOTS.values():
        assert "reports/training" in path.as_posix()


def test_finetune_unknown_game_rejected() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import main

    with pytest.raises(SystemExit) as exc:
        main(["--game", "monopoly"])
    assert exc.value.code == 2


@pytest.mark.parametrize("amplified", [False, True])
def test_finetune_missing_dataset_stops_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], amplified: bool,
) -> None:
    import subprocess

    from isaac.isaaclab_tasks.synria_pickplace.gr00t import dataset, finetune

    roots = dataset.SYNRIA_MIMIC_AMPLIFIED_ROOTS if amplified else dataset.SYNRIA_DEMO_ROOTS
    monkeypatch.setitem(roots, "chess", tmp_path / "missing")

    def reject_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing dataset must not launch training")

    monkeypatch.setattr(subprocess, "call", reject_launch)
    argv = ["--game", "chess"] + (["--amplified"] if amplified else [])
    with pytest.raises(SystemExit) as exc:
        finetune.main(argv)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert ("amplified" if amplified else "seed") in error
    assert "SYNRIA_GR00T_FINETUNE" in error


# ---------------------------------------------------------------------------
# finetune.py — parser and dry-run
# ---------------------------------------------------------------------------


def test_finetune_parser_has_game_arg() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import _build_parser

    parser = _build_parser()
    # --game is required; parse without it should fail
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_finetune_parser_default_base_model() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import _build_parser

    args = _build_parser().parse_args(["--game", "chess"])
    assert args.base_model == "nvidia/GR00T-N1.7-3B"


def test_finetune_parser_default_max_steps() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import _build_parser

    args = _build_parser().parse_args(["--game", "chess"])
    assert args.max_steps == 10000


def test_finetune_parser_default_global_batch_size() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import _build_parser

    args = _build_parser().parse_args(["--game", "chess"])
    assert args.global_batch_size == 16


def test_finetune_parser_dry_run_flag() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import _build_parser

    args = _build_parser().parse_args(["--game", "chess", "--dry-run"])
    assert args.dry_run is True


def test_finetune_build_launch_cmd_contains_embodiment_tag() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import SYNRIA_EMBODIMENT_TAG
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import (
        _build_launch_cmd,
        _build_parser,
    )

    args = _build_parser().parse_args(["--game", "chess", "--dry-run"])
    cmd = _build_launch_cmd(args)
    assert SYNRIA_EMBODIMENT_TAG in cmd


def test_finetune_build_launch_cmd_contains_dataset_path() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import SYNRIA_DEMO_ROOTS
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import (
        _build_launch_cmd,
        _build_parser,
    )

    args = _build_parser().parse_args(["--game", "chess", "--dry-run"])
    cmd = _build_launch_cmd(args)
    assert str(SYNRIA_DEMO_ROOTS["chess"]) in cmd


def test_finetune_dry_run_returns_zero(tmp_path: Path) -> None:
    """--dry-run must return 0 without touching the dataset or GR00T."""
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import main

    rc = main(["--game", "chess", "--dry-run", "--output-dir", str(tmp_path)])
    assert rc == 0


def test_finetune_dry_run_amplified_returns_zero(tmp_path: Path) -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import main

    rc = main(["--game", "ludo", "--amplified", "--dry-run", "--output-dir", str(tmp_path)])
    assert rc == 0


def test_finetune_missing_dataset_exits_two() -> None:
    """Without --dry-run and missing dataset, main() must raise SystemExit(2)."""
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--game", "chess"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# inference.py — parser
# ---------------------------------------------------------------------------


def test_inference_parser_task_required() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.inference import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--checkpoint", "/some/path"])


def test_inference_parser_checkpoint_required() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.inference import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--task", "Synria-Chess-PickPlace-v0"])


def test_inference_parser_default_num_episodes() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.inference import _build_parser

    args = _build_parser().parse_args(
        ["--task", "Synria-Chess-PickPlace-v0", "--checkpoint", "/some/ckpt"]
    )
    assert args.num_episodes == 20


def test_inference_parser_task_choices() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.inference import _build_parser

    valid_tasks = [
        "Synria-Ludo-PickPlace-v0",
        "Synria-Chess-PickPlace-v0",
        "Synria-Checkers-PickPlace-v0",
    ]
    for task in valid_tasks:
        args = _build_parser().parse_args(["--task", task, "--checkpoint", "/ckpt"])
        assert args.task == task


def test_inference_parser_default_report_path() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.inference import _build_parser

    args = _build_parser().parse_args(
        ["--task", "Synria-Ludo-PickPlace-v0", "--checkpoint", "/ckpt"]
    )
    assert "gr00t_eval" in str(args.report)

"""Tests for the Synria RSL-RL training config and training script parser.

All tests are import-safe — no Isaac Sim, Isaac Lab, or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# train_cfg.py — TASK_TRAIN_CFG registry and config classes
# ---------------------------------------------------------------------------


def test_task_train_cfg_has_three_entries() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG

    assert len(TASK_TRAIN_CFG) == 3


def test_task_train_cfg_contains_all_task_ids() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG

    expected = {
        "Synria-Ludo-PickPlace-v0",
        "Synria-Chess-PickPlace-v0",
        "Synria-Checkers-PickPlace-v0",
    }
    assert set(TASK_TRAIN_CFG.keys()) == expected


def test_chess_config_experiment_name() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaChessPickPlacePPORunnerCfg

    cfg = SynriaChessPickPlacePPORunnerCfg()
    assert "chess" in cfg.experiment_name.lower()


def test_checkers_config_experiment_name() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaCheckersPickPlacePPORunnerCfg

    cfg = SynriaCheckersPickPlacePPORunnerCfg()
    assert "checkers" in cfg.experiment_name.lower()


def test_ludo_config_experiment_name() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaLudoPickPlacePPORunnerCfg

    cfg = SynriaLudoPickPlacePPORunnerCfg()
    assert "ludo" in cfg.experiment_name.lower()


def test_base_config_max_iterations() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaPickPlaceBasePPORunnerCfg

    cfg = SynriaPickPlaceBasePPORunnerCfg()
    assert cfg.max_iterations == 1500


def test_base_config_num_steps_per_env() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaPickPlaceBasePPORunnerCfg

    cfg = SynriaPickPlaceBasePPORunnerCfg()
    assert cfg.num_steps_per_env == 24


def test_base_config_save_interval() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaPickPlaceBasePPORunnerCfg

    cfg = SynriaPickPlaceBasePPORunnerCfg()
    assert cfg.save_interval == 50


def test_base_config_logger_is_tensorboard() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import SynriaPickPlaceBasePPORunnerCfg

    cfg = SynriaPickPlaceBasePPORunnerCfg()
    assert cfg.logger == "tensorboard"


def test_ppo_actor_hidden_dims_are_set() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import _ACTOR_CRITIC_CFG

    assert _ACTOR_CRITIC_CFG.actor_hidden_dims == [512, 256, 128]
    assert _ACTOR_CRITIC_CFG.critic_hidden_dims == [512, 256, 128]


def test_ppo_algorithm_clip_param() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import _PPO_ALGORITHM_CFG

    assert _PPO_ALGORITHM_CFG.clip_param == pytest.approx(0.2)


def test_ppo_algorithm_gamma() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import _PPO_ALGORITHM_CFG

    assert _PPO_ALGORITHM_CFG.gamma == pytest.approx(0.99)


def test_ppo_algorithm_lam() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import _PPO_ALGORITHM_CFG

    assert _PPO_ALGORITHM_CFG.lam == pytest.approx(0.95)


def test_ppo_algorithm_learning_rate() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import _PPO_ALGORITHM_CFG

    assert _PPO_ALGORITHM_CFG.learning_rate == pytest.approx(3e-4)


def test_per_game_configs_have_unique_experiment_names() -> None:
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import (
        SynriaCheckersPickPlacePPORunnerCfg,
        SynriaChessPickPlacePPORunnerCfg,
        SynriaLudoPickPlacePPORunnerCfg,
    )

    names = {
        SynriaLudoPickPlacePPORunnerCfg().experiment_name,
        SynriaChessPickPlacePPORunnerCfg().experiment_name,
        SynriaCheckersPickPlacePPORunnerCfg().experiment_name,
    }
    assert len(names) == 3  # all unique


# ---------------------------------------------------------------------------
# train_synria_pickplace.py — argument parser (no Isaac Sim needed)
# ---------------------------------------------------------------------------


def _load_train_script() -> object:
    """Load train_synria_pickplace without triggering __main__."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_synria_pickplace",
        REPO_ROOT / "isaac" / "scripts" / "train_synria_pickplace.py",
    )
    assert spec is not None and spec.loader is not None
    import types

    mod = importlib.util.module_from_spec(spec)
    # Patch out SimulationApp import so the module loads without Isaac Sim
    sys.modules.setdefault("isaacsim", types.ModuleType("isaacsim"))
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_train_script_parser_task_required() -> None:
    mod = _load_train_script()
    with pytest.raises(SystemExit):
        mod._build_parser().parse_args([])  # type: ignore[attr-defined]


def test_train_script_parser_default_num_envs() -> None:
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0"]
    )
    assert args.num_envs == 4096


def test_train_script_parser_delegates_device_to_app_launcher() -> None:
    # --device is injected by AppLauncher.add_app_launcher_args() at runtime,
    # so the base parser must NOT define it (a duplicate raises ArgumentError).
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0"]
    )
    assert not hasattr(args, "device")


def test_train_script_parser_default_seed() -> None:
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0"]
    )
    assert args.seed == 42


def test_train_script_parser_delegates_headless_to_app_launcher() -> None:
    # --headless is injected by AppLauncher.add_app_launcher_args() at runtime,
    # so the base parser must NOT define it (a duplicate raises ArgumentError).
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0"]
    )
    assert not hasattr(args, "headless")


def test_train_script_parser_max_iterations_override() -> None:
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0", "--max_iterations", "5"]
    )
    assert args.max_iterations == 5


def test_train_script_parser_task_choices() -> None:
    mod = _load_train_script()
    parser = mod._build_parser()  # type: ignore[attr-defined]
    valid_tasks = [
        "Synria-Ludo-PickPlace-v0",
        "Synria-Chess-PickPlace-v0",
        "Synria-Checkers-PickPlace-v0",
    ]
    for task in valid_tasks:
        args = parser.parse_args(["--task", task])
        assert args.task == task


def test_train_script_parser_resume_flag() -> None:
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0", "--resume"]
    )
    assert args.resume is True


def test_train_script_parser_num_envs_smoke() -> None:
    """num_envs=16 + max_iterations=5 is the documented smoke-test config."""
    mod = _load_train_script()
    args = mod._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--task", "Synria-Chess-PickPlace-v0", "--num_envs", "16", "--max_iterations", "5"]
    )
    assert args.num_envs == 16
    assert args.max_iterations == 5

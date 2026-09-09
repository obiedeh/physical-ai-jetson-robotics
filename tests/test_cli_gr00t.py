"""CLI tests for the GR00T-related commands in physical_ai_lab.cli.

Tests cover both commands that are executable on any machine without GPU,
GR00T, or Isaac Lab:

    physical-ai-lab gr00t-embodiment [--json]
    physical-ai-lab gr00t-finetune-dry-run [--game ...] [--amplified] ...

All tests use Typer's CliRunner so no subprocess is spawned and no
hardware is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_ai_lab.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> Result:
    """Invoke the CLI with the given arguments and return the Result."""
    return runner.invoke(app, list(args))


# ===========================================================================
# gr00t-embodiment
# ===========================================================================


class TestGr00tEmbodiment:
    """Tests for `physical-ai-lab gr00t-embodiment`."""

    def test_exits_zero(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert result.exit_code == 0

    def test_table_output_contains_embodiment_tag(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert "new_embodiment" in result.output

    def test_table_output_contains_all_six_arm_joints(self) -> None:
        result = _invoke("gr00t-embodiment")
        for joint in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"):
            assert joint in result.output

    def test_table_output_contains_gripper_joints(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert "left_finger" in result.output
        assert "right_finger" in result.output

    def test_table_output_contains_action_dim_8(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert "8" in result.output

    def test_table_output_contains_camera_keys(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert "wrist" in result.output
        assert "overhead" in result.output

    def test_table_output_mentions_recipe_doc(self) -> None:
        result = _invoke("gr00t-embodiment")
        assert "SYNRIA_GR00T_FINETUNE" in result.output

    # --- JSON mode ---

    def test_json_flag_exits_zero(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        assert result.exit_code == 0

    def test_json_output_is_valid_json(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_json_embodiment_tag(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert data["embodiment_tag"] == "new_embodiment"

    def test_json_arm_joints_list(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert data["arm_joints"] == [
            "Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"
        ]

    def test_json_gripper_joints_list(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert data["gripper_joints"] == ["left_finger", "right_finger"]

    def test_json_action_dim_is_8(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert data["action_dim"] == 8

    def test_json_camera_keys(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert "wrist" in data["camera_keys"]
        assert "overhead" in data["camera_keys"]

    def test_json_image_size_224x224(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        assert data["image_size"] == [224, 224]

    def test_json_modality_config_top_level_keys(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        modality = data["modality_config"]
        assert {"embodiment_tag", "state", "vision", "language", "action"} <= set(modality)

    def test_json_modality_action_dim_matches(self) -> None:
        """Sum of action head shapes must equal action_dim (8)."""
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        action_section = data["modality_config"]["action"]
        total = sum(spec["shape"][0] for spec in action_section.values())
        assert total == data["action_dim"]

    def test_json_modality_state_covers_joints_and_gripper(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        state = data["modality_config"]["state"]
        assert "arm_joint_pos" in state
        assert "arm_joint_vel" in state
        assert "gripper_width" in state

    def test_json_modality_vision_has_both_cameras(self) -> None:
        result = _invoke("gr00t-embodiment", "--json")
        data = json.loads(result.output)
        vision = data["modality_config"]["vision"]
        assert "wrist" in vision
        assert "overhead" in vision


# ===========================================================================
# gr00t-finetune-dry-run
# ===========================================================================


class TestGr00tFinetuneDir:
    """Tests for `physical-ai-lab gr00t-finetune-dry-run`."""

    # --- Exit codes ---

    def test_default_exits_zero(self) -> None:
        result = _invoke("gr00t-finetune-dry-run")
        assert result.exit_code == 0

    def test_all_games_exit_zero(self) -> None:
        for game in ("chess", "checkers", "ludo"):
            result = _invoke("gr00t-finetune-dry-run", "--game", game)
            assert result.exit_code == 0, f"game={game!r} failed: {result.output}"

    def test_invalid_game_exits_nonzero(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "monopoly")
        assert result.exit_code != 0

    # --- Output content: per-game dataset path ---

    def test_chess_output_contains_chess_dataset_path(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "synria_chess" in result.output

    def test_checkers_output_contains_checkers_dataset_path(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "checkers")
        assert "synria_checkers" in result.output

    def test_ludo_output_contains_ludo_dataset_path(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "ludo")
        assert "synria_ludo" in result.output

    # --- Seed vs amplified dataset path ---

    def test_seed_path_used_by_default(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "lerobot_demos" in result.output
        assert "mimic_amplified" not in result.output

    def test_amplified_flag_switches_to_mimic_path(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess", "--amplified")
        assert "mimic_amplified" in result.output

    # --- Embodiment tag and model ---

    def test_output_contains_embodiment_tag(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "new_embodiment" in result.output

    def test_default_base_model_in_output(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "GR00T-N1.7-3B" in result.output

    def test_custom_base_model_propagated(self) -> None:
        result = _invoke(
            "gr00t-finetune-dry-run",
            "--game", "chess",
            "--base-model", "nvidia/GR00T-N1.7-2B",
        )
        assert "GR00T-N1.7-2B" in result.output

    # --- Hyperparameter propagation ---

    def test_custom_max_steps_in_output(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess", "--max-steps", "500")
        assert result.exit_code == 0
        assert "--max-steps 500" in result.output

    def test_custom_global_batch_size_in_output(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess", "--global-batch-size", "32")
        assert result.exit_code == 0
        assert "--global-batch-size 32" in result.output

    # --- Dry-run markers ---

    def test_launch_command_line_present(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "launch_finetune" in result.output

    def test_dry_run_confirmation_line_present(self) -> None:
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "dry run" in result.output.lower()

    def test_does_not_actually_invoke_training(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run must not launch a training subprocess."""
        import subprocess

        def reject_launch(*args: object, **kwargs: object) -> None:
            pytest.fail("Dry-run attempted to launch training")

        monkeypatch.setattr(subprocess, "call", reject_launch)
        result = _invoke("gr00t-finetune-dry-run", "--game", "chess")
        assert "not invoking" in result.output
        assert "training started" not in result.output.lower()

    # --- Amplified + game combination ---

    def test_ludo_amplified_combination(self) -> None:
        result = _invoke(
            "gr00t-finetune-dry-run",
            "--game", "ludo",
            "--amplified",
        )
        assert result.exit_code == 0
        assert "synria_ludo" in result.output
        assert "mimic_amplified" in result.output


# ===========================================================================
# gr00t-inference-dry-run
# ===========================================================================


class TestGr00tInferenceDryRun:
    """Tests for `physical-ai-lab gr00t-inference-dry-run`."""

    # --- Exit codes ---

    def test_default_exits_zero(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert result.exit_code == 0

    def test_all_valid_tasks_exit_zero(self) -> None:
        for task in (
            "Synria-Ludo-PickPlace-v0",
            "Synria-Chess-PickPlace-v0",
            "Synria-Checkers-PickPlace-v0",
        ):
            result = _invoke("gr00t-inference-dry-run", "--task", task)
            assert result.exit_code == 0, f"task={task!r} failed: {result.output}"

    def test_invalid_task_exits_nonzero(self) -> None:
        result = _invoke("gr00t-inference-dry-run", "--task", "Invalid-Task-v99")
        assert result.exit_code != 0

    # --- Default task in output ---

    def test_default_task_chess_in_output(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "Chess" in result.output

    def test_ludo_task_in_output(self) -> None:
        result = _invoke("gr00t-inference-dry-run", "--task", "Synria-Ludo-PickPlace-v0")
        assert "Ludo" in result.output

    def test_checkers_task_in_output(self) -> None:
        result = _invoke("gr00t-inference-dry-run", "--task", "Synria-Checkers-PickPlace-v0")
        assert "Checkers" in result.output

    # --- Checkpoint propagation ---

    def test_custom_checkpoint_appears_in_output(self) -> None:
        result = _invoke(
            "gr00t-inference-dry-run",
            "--checkpoint", "/tmp/my_checkpoint",
        )
        assert "my_checkpoint" in result.output

    # --- Base model propagation ---

    def test_default_base_model_in_output(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "GR00T-N1.7-3B" in result.output

    def test_custom_base_model_in_output(self) -> None:
        result = _invoke(
            "gr00t-inference-dry-run",
            "--base-model", "nvidia/GR00T-N1.7-2B",
        )
        assert "GR00T-N1.7-2B" in result.output

    # --- Num-episodes propagation ---

    def test_custom_num_episodes_in_output(self) -> None:
        result = _invoke("gr00t-inference-dry-run", "--num-episodes", "5")
        assert "5" in result.output

    # --- Dry-run markers ---

    def test_dry_run_header_present(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "dry run" in result.output.lower()

    def test_not_invoking_present(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "Not invoking" in result.output

    def test_does_not_say_rollout_started(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "rollout started" not in result.output.lower()

    # --- Recipe reference ---

    def test_recipe_doc_mentioned(self) -> None:
        result = _invoke("gr00t-inference-dry-run")
        assert "SYNRIA_GR00T_FINETUNE" in result.output

"""The practice run length is an operator setting for one disposable smoke run, 30 s by default."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from test_console_service import ConsoleFixture, values
from test_task_registry import synthetic_registry

from synria_lerobot import recorder
from synria_lerobot.task_registry import (
    DEFAULT_SMOKE_WINDOW_S,
    DISPOSABLE_SMOKE,
    load_task_registry,
    validate_smoke_window,
)


def test_default_window_is_thirty_seconds_and_never_enters_the_task_hash() -> None:
    """No length set means 30 s practice takes; setting one leaves the task identity alone."""
    task = load_task_registry(Path("config/synria_tasks.json"))["die_into_cup"]
    assert task.recording_window(DISPOSABLE_SMOKE) == {"min_episode_s": 30.0,
                                                       "max_episode_s": 30.0}
    longer = task.with_smoke_window(45)
    assert longer.recording_window(DISPOSABLE_SMOKE) == {"min_episode_s": 45.0,
                                                         "max_episode_s": 45.0}
    assert longer.sha256 == task.sha256 and longer.as_dict() == task.as_dict()
    assert "smoke_window_s" not in longer.metadata(DISPOSABLE_SMOKE)
    assert longer.metadata(DISPOSABLE_SMOKE)["max_episode_s"] == 45.0
    assert DEFAULT_SMOKE_WINDOW_S == 30.0


@pytest.mark.parametrize("bad", [4, 121, 0, -5, float("nan"), float("inf"), True, "30", None])
def test_lengths_outside_five_to_one_hundred_twenty_seconds_are_refused(bad: object) -> None:
    """Only a finite number of seconds within the bounds is accepted."""
    with pytest.raises(ValueError, match="practice run length"):
        validate_smoke_window(bad)


def test_recorder_configuration_applies_the_length_to_smoke_only(tmp_path: Path) -> None:
    """The length reaches the contract for a smoke run and is ignored for a real dataset."""
    registry = synthetic_registry(tmp_path / "tasks.json", 20, 30)
    base = dict(task_registry=registry, task_id="die_into_cup", gripper_type="50mm",
                action_source="next_state", state_has_velocity=False, action_lookahead_steps=1,
                repo_id="local/x", dataset_path=tmp_path / "dataset", fps=15, image_width=224,
                image_height=224, state_source="standalone_driver", follower_topic="/joint_states",
                guard_command_topic=[])
    config = recorder.recording_config_from_args(
        argparse.Namespace(**base, smoke=True, practice_seconds=12))
    assert config.contract.max_episode_s == 12.0 and config.contract.min_episode_s == 12.0
    config = recorder.recording_config_from_args(argparse.Namespace(**base, smoke=True))
    assert config.contract.max_episode_s == 30.0
    # The console checks a qualifying configuration from the same settings: the task
    # window rules there and the practice length is ignored, not an error.
    config = recorder.recording_config_from_args(
        argparse.Namespace(**base, smoke=False, practice_seconds=12))
    assert config.contract.max_episode_s == 30.0 and config.contract.min_episode_s == 20.0


def test_console_accepts_the_length_for_practice_runs_and_refuses_it_for_sessions(
    tmp_path: Path,
) -> None:
    """The page sends practice_seconds with a practice run; a session payload may not carry it."""
    physical = {**values(), "state_source": "standalone_driver", "wrist_camera": "fake-wrist",
                "front_camera": "fake-front", "follower_serial": "SERIAL",
                "power_state_start": "on"}
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings({**physical, "practice_seconds": 45}, smoke=True)
        assert settings["practice_seconds"] == 45.0
        assert service._settings(physical, smoke=True)["practice_seconds"] == 30.0
        with pytest.raises(ValueError, match="between 5 and 120"):
            service._settings({**physical, "practice_seconds": 3}, smoke=True)
        with pytest.raises(ValueError, match="practice runs only"):
            service._settings({**physical, "practice_seconds": 45}, smoke=False)

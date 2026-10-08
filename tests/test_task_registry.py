"""Fixed-scene identities and windows; all configured times here are synthetic."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from synria_lerobot.task_registry import (
    TASK_IDS,
    TaskDefinition,
    load_task_registry,
    select_recording_task,
)

REGISTRY = Path(__file__).resolve().parents[1] / "config" / "synria_tasks.json"


def synthetic_task(minimum: float, maximum: float, task_id: str = "die_into_cup") -> TaskDefinition:
    return replace(
        load_task_registry(REGISTRY)[task_id], min_episode_s=minimum, max_episode_s=maximum
    )


def test_registry_is_exactly_three_unconfigured_fixed_scene_tasks() -> None:
    tasks = load_task_registry(REGISTRY)
    assert tuple(tasks) == TASK_IDS
    for name, task in tasks.items():
        assert task.min_episode_s is task.max_episode_s is None
        assert task.as_dict()["declared_fixed_scene"] is True
        assert task.as_dict()["requires_unobserved_goal"] is False
        with pytest.raises(ValueError, match="operator timing required"):
            select_recording_task(REGISTRY, name)


@pytest.mark.parametrize("minimum,maximum", [
    (None, 30), (20, None), (0, 30), (-1, 30), (40, 30), (True, 30), (20, False),
    (float("nan"), 30), (20, float("inf")), ("20", 30),
])
def test_registry_rejects_invalid_or_partial_window(minimum: object, maximum: object) -> None:
    with pytest.raises(ValueError, match="episode window"):
        synthetic_task(minimum, maximum)


@pytest.mark.parametrize("field,value", [
    ("task_id", "token_move"), ("task_text", ""), ("success_rule", None),
    ("scene_requirements", []), ("scene_requirements", [False]),
    ("declared_fixed_scene", False), ("declared_fixed_scene", 1),
    ("requires_unobserved_goal", True), ("requires_unobserved_goal", 0),
])
def test_registry_rejects_unknown_or_unobserved_goal_tasks(field: str, value: object) -> None:
    payload = synthetic_task(40, 90).as_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        TaskDefinition.from_dict(payload)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_registry_requires_exact_task_set(tmp_path: Path, change: str) -> None:
    payload = json.loads(REGISTRY.read_text())
    if change == "missing":
        payload["tasks"].pop()
    elif change == "duplicate":
        payload["tasks"][1] = copy.deepcopy(payload["tasks"][0])
    else:
        payload["tasks"].append(copy.deepcopy(payload["tasks"][0]))
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly the three"):
        load_task_registry(path)


def test_registry_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(REGISTRY.read_text().replace(
        '"requires_unobserved_goal": false',
        '"requires_unobserved_goal": true, "requires_unobserved_goal": false', 1,
    ))
    with pytest.raises(ValueError, match="duplicate task registry field"):
        load_task_registry(path)


def test_task_hash_binds_full_entry_not_other_registry_entries(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text())
    payload["tasks"][0]["episode_window"] = {"min_episode_s": 40, "max_episode_s": 90}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    task = select_recording_task(path, "die_into_cup")
    assert task.require_configured() == {"min_episode_s": 40, "max_episode_s": 90}
    assert TaskDefinition.from_dict(task.as_dict()).sha256 == task.sha256
    payload["tasks"][1]["episode_window"] = {"min_episode_s": 10, "max_episode_s": 80}
    path.write_text(json.dumps(payload))
    assert select_recording_task(path, "die_into_cup").sha256 == task.sha256
    for field, value in (
        ("task_text", "Changed instruction"), ("success_rule", "Changed rule"),
        ("scene_requirements", ("Changed scene",)), ("max_episode_s", 91),
    ):
        assert replace(task, **{field: value}).sha256 != task.sha256
    with pytest.raises(ValueError, match="unknown"):
        select_recording_task(path, "token_move")


@pytest.mark.parametrize("field,value", [
    ("task_id", "cup_return"), ("task_text", "move a token"),
    ("task_definition_sha256", "0" * 64), ("min_episode_s", 20),
    ("max_episode_s", 90), ("min_episode_s", True),
])
def test_snapshot_identity_cannot_be_relabelled(field: str, value: object) -> None:
    metadata = synthetic_task(40, 80).metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match="task id|episode window"):
        TaskDefinition.from_metadata(metadata)


def test_contract_binds_snapshot_without_registry_io(monkeypatch: pytest.MonkeyPatch) -> None:
    from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract

    task = synthetic_task(40, 80)
    contract = PhysicalDatasetContract(
        "50mm", ActionSource.NEXT_STATE, False, task_id=task.task_id, task_definition=task,
    )
    payload = contract.as_dict(fps=15)
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: pytest.fail("unexpected file read")
    )
    assert PhysicalDatasetContract.from_dict(payload) == contract
    assert contract.min_episode_s == 40 and contract.max_episode_s == 80
    with pytest.raises(ValueError, match="matching"):
        replace(contract, task_id="cup_return")
    payload.pop("task_definition")
    with pytest.raises(ValueError, match="task definition"):
        PhysicalDatasetContract.from_dict(payload)

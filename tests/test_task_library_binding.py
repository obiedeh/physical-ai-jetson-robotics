"""Library tasks beside the published registry: ids, snapshot files, contract binding, counting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_quality_gates import LIMITS_PATH, _baseline_episode, _provenance
from test_task_registry import REGISTRY, synthetic_task

from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract
from synria_lerobot.quality_gates import GateConfig, load_limits, write_session_artifacts
from synria_lerobot.task_registry import (
    LIBRARY_SNAPSHOT_VERSION,
    TaskDefinition,
    is_library_task_id,
    library_snapshot_fields,
    load_task_registry,
    load_task_source,
    select_recording_task,
)


def library_task(minimum: float | None = 8, maximum: float | None = 25) -> TaskDefinition:
    """One library task version as the task library would snapshot it."""
    return TaskDefinition(
        "lib_roll_a_die_1",
        "Pick the die from its start zone, release it into the upright cup on its mark, "
        "and return to ready.",
        "The die is inside the cup.",
        ("The cup is upright on its mark.", "The die is in its start zone."),
        minimum, maximum,
    )


def snapshot(path: Path, task: TaskDefinition | None = None, **overrides: object) -> Path:
    """Write a one-task library snapshot file."""
    payload: dict[str, object] = {
        "version": LIBRARY_SNAPSHOT_VERSION, "task_source": "library",
        "goal_id": "lib_roll_a_die", "goal_title": "Roll a die", "library_version": 1,
        "chain_position": 1, "task": (task or library_task()).as_dict(),
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("task_id", ["lib_roll_a_die_1", "lib_put_block_in_box_2", "lib_x"])
def test_library_ids_are_accepted_beside_the_three_registry_ids(task_id: str) -> None:
    """A library id follows one pattern and never collides with the published ids."""
    assert is_library_task_id(task_id)
    assert task_id not in load_task_registry(REGISTRY)
    assert replace(library_task(), task_id=task_id).task_id == task_id


@pytest.mark.parametrize("task_id", [
    "library_x", "Lib_1", "lib_", "lib-roll", "die_into_cup_2", "",
])
def test_malformed_ids_are_refused(task_id: str) -> None:
    """Anything outside the registry ids and the library pattern is still unknown."""
    assert not is_library_task_id(task_id)
    with pytest.raises(ValueError, match="unknown fixed-scene task id"):
        replace(library_task(), task_id=task_id)


def test_registry_file_still_requires_exactly_the_four_tasks(tmp_path: Path) -> None:
    """The published registry cannot smuggle a library task in."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["tasks"].append(library_task().as_dict())
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the four"):
        load_task_registry(path)
    with pytest.raises(ValueError, match="exactly the four"):
        load_task_source(path)


def test_snapshot_file_loads_one_library_task_with_its_binding(tmp_path: Path) -> None:
    """A snapshot is a second task source: one task, its goal, version and chain position."""
    path = snapshot(tmp_path / "snapshot.json")
    tasks = load_task_source(path)
    assert list(tasks) == ["lib_roll_a_die_1"]
    assert tasks["lib_roll_a_die_1"] == library_task()
    assert library_snapshot_fields(path) == {
        "task_source": "library", "goal_id": "lib_roll_a_die", "goal_title": "Roll a die",
        "library_version": 1, "chain_position": 1,
    }
    assert library_snapshot_fields(REGISTRY) == {}, "the registry has no library binding"
    selected = select_recording_task(path, "lib_roll_a_die_1")
    assert selected.sha256 == library_task().sha256
    with pytest.raises(ValueError, match="unknown fixed-scene task id"):
        select_recording_task(path, "die_into_cup")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("task_source", "registry", "missing or unknown"),
    ("goal_id", "roll a die", "goal id"),
    ("goal_title", " ", "goal title"),
    ("library_version", 0, "library_version"),
    ("chain_position", "1", "chain_position"),
])
def test_snapshot_binding_is_validated(
    tmp_path: Path, field: str, value: object, message: str,
) -> None:
    """Every binding field is checked before a snapshot can select a task."""
    path = snapshot(tmp_path / "bad.json", **{field: value})
    with pytest.raises(ValueError, match=message):
        library_snapshot_fields(path)


def test_snapshot_task_must_carry_a_library_id(tmp_path: Path) -> None:
    """A registry task inside a snapshot would let a dataset claim the wrong source."""
    path = snapshot(tmp_path / "wrong.json", task=synthetic_task(8, 25))
    with pytest.raises(ValueError, match="library task id"):
        load_task_source(path)


def test_untimed_library_task_cannot_qualify_but_can_practice(tmp_path: Path) -> None:
    """Timing stays operator confirmed: no window, no qualifying session, as for the registry."""
    path = snapshot(tmp_path / "untimed.json", task=library_task(None, None))
    with pytest.raises(ValueError, match="operator timing required"):
        select_recording_task(path, "lib_roll_a_die_1")
    assert select_recording_task(path, "lib_roll_a_die_1", purpose="disposable_smoke")


def test_registry_contract_payload_is_unchanged() -> None:
    """Existing datasets keep their exact contract payload and hash binding."""
    task = synthetic_task(40, 80)
    contract = PhysicalDatasetContract(
        "50mm", ActionSource.NEXT_STATE, False, task_id=task.task_id, task_definition=task,
    )
    payload = contract.as_dict(fps=15)
    assert "task_source" not in payload and "goal_id" not in payload
    assert PhysicalDatasetContract.from_dict(payload) == contract
    assert contract.task_source == "registry"
    with pytest.raises(ValueError, match="no library goal binding"):
        replace(contract, goal_id="lib_roll_a_die")
    with pytest.raises(ValueError, match="no library goal binding"):
        PhysicalDatasetContract(
            "50mm", ActionSource.NEXT_STATE, False, task_id="lib_roll_a_die_1",
            task_definition=library_task(),
        )


def test_library_contract_round_trips_with_its_binding() -> None:
    """A library dataset records the exact task snapshot, its hash and the goal binding."""
    task = library_task()
    contract = PhysicalDatasetContract(
        "50mm", ActionSource.NEXT_STATE, False, task_id=task.task_id, task_definition=task,
        task_source="library", goal_id="lib_roll_a_die", goal_title="Roll a die",
        library_version=2, chain_position=1,
    )
    payload = contract.as_dict(fps=15)
    assert payload["task_source"] == "library" and payload["goal_id"] == "lib_roll_a_die"
    assert payload["library_version"] == 2 and payload["chain_position"] == 1
    assert payload["task_definition_sha256"] == task.sha256
    assert PhysicalDatasetContract.from_dict(payload) == contract
    for field, value, message in (
        ("goal_id", None, "goal id"), ("goal_title", "", "goal title"),
        ("library_version", 0, "positive version"), ("chain_position", None, "positive version"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(contract, **{field: value})
    with pytest.raises(ValueError, match="library task id"):
        replace(contract, task_source="library", task_id="die_into_cup",
                task_definition=synthetic_task(8, 25))


def test_library_takes_run_every_gate_but_never_count_toward_milestones(tmp_path: Path) -> None:
    """Quality is judged as always; the milestone count stays tied to the published registry."""
    task = library_task(20, 30)
    limits = replace(load_limits(LIMITS_PATH), verified_by="test", verified_on="2026-10-11")
    binding = {"task_source": "library", "goal_id": "lib_roll_a_die", "goal_title": "Roll a die",
               "library_version": 1, "chain_position": 1}
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    summary = write_session_artifacts(
        session_dir=tmp_path / "reports" / "library-session", dataset_path=dataset,
        provenance={**_provenance("leader"), **task.metadata("qualifying"), **binding},
        episodes=[replace(_baseline_episode(), task_definition=task)], limits=limits,
        gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
    )
    assert summary["episode_count"] == 1
    assert summary["quality_valid_episode_count"] == 1, summary["gate_results"]
    assert summary["qualifying_episode_count"] == 0, "library takes never count toward milestones"
    assert summary["task_source"] == "library" and summary["goal_id"] == "lib_roll_a_die"
    assert summary["chain_position"] == 1 and summary["library_version"] == 1
    registry_summary = write_session_artifacts(
        session_dir=tmp_path / "reports" / "registry-session", dataset_path=tmp_path / "other",
        provenance={**_provenance("leader"), **synthetic_task(20, 30).metadata("qualifying")},
        episodes=[_baseline_episode()], limits=limits,
        gate_config=GateConfig(min_episode_s=20, max_episode_s=30),
    )
    assert registry_summary["task_source"] == "registry"
    assert registry_summary["qualifying_episode_count"] == 1

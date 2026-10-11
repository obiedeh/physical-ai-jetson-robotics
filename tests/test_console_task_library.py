"""The task library in the console: catalog versions and freezing, service routes, sessions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from test_console_service import ConsoleFixture, values

from synria_lerobot.console.catalog import Catalog
from synria_lerobot.task_planner import ExternalPlanner, builtin_plan
from synria_lerobot.task_registry import load_task_source

REPOSITORY = Path(__file__).resolve().parents[1]
GOAL = {"id": "lib_roll_a_die", "title": "Roll a die", "sentence": "train to roll a dice"}


@contextmanager
def opened(workspace: Path) -> Iterator[Catalog]:
    """A catalog that is always closed."""
    catalog = Catalog(workspace, repository=REPOSITORY)
    try:
        yield catalog
    finally:
        catalog.close()


def seed_payload() -> dict[str, Any]:
    """The seed breakdown as the Save tasks screen would post it."""
    result = builtin_plan("train to roll a dice")
    assert result.breakdown is not None
    return result.breakdown.as_dict()


def catalog_tasks(breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    """Sub-task rows as the service hands them to the catalog."""
    return [{"task_id": f"lib_roll_a_die_{position}", **step,
             "episode_window": {"min_episode_s": None, "max_episode_s": None}}
            for position, step in enumerate(breakdown["sub_tasks"], start=1)]


def schema_version(catalog: Catalog) -> int:
    """The stored schema version."""
    return int(catalog.connection.execute("PRAGMA user_version").fetchone()[0])


def test_catalog_schema_upgrades_to_version_two(tmp_path: Path) -> None:
    """A fresh database lands on version two; a version one database is upgraded in place."""
    with opened(tmp_path / "fresh") as catalog:
        assert schema_version(catalog) == 2
        assert catalog.goals() == []
    old = tmp_path / "old"
    with opened(old) as catalog:
        catalog.connection.executescript(
            "DROP TABLE library_tasks; DROP TABLE goals; PRAGMA user_version=1;"
        )
        catalog.connection.commit()
    with opened(old) as catalog:
        assert schema_version(catalog) == 2
        assert catalog.goals() == []


def test_save_goal_versions_and_freezes(tmp_path: Path) -> None:
    """Editing an unfrozen version replaces it; editing a frozen version writes the next one."""
    with opened(tmp_path / "workspace") as catalog:
        breakdown = seed_payload()
        goal = {**GOAL, "rationale": breakdown["goal"]["rationale"]}
        saved = catalog.save_goal(goal, catalog_tasks(breakdown), planner_source="built-in planner")
        assert [task["task_id"] for task in saved["tasks"]] == [
            "lib_roll_a_die_1", "lib_roll_a_die_2", "lib_roll_a_die_3", "lib_roll_a_die_4"]
        assert all(task["version"] == 1 and not task["frozen"] for task in saved["tasks"])
        assert saved["tasks"][0]["chain_position"] == 1 and saved["tasks"][0]["kind"] == "skill"
        assert saved["sentence"] == "train to roll a dice", "the sentence stays with the goal"
        # An edit before any take replaces version 1 in place.
        tasks = catalog_tasks(breakdown)
        tasks[0]["name"] = "Die into the cup"
        again = catalog.save_goal(goal, tasks, planner_source="operator")
        assert again["tasks"][0]["name"] == "Die into the cup"
        assert again["tasks"][0]["version"] == 1
        # Once frozen, the same edit creates version 2 and version 1 stays readable.
        catalog.freeze_library_task("lib_roll_a_die_1", 1)
        tasks[0]["name"] = "Die goes into the cup"
        third = catalog.save_goal(goal, tasks, planner_source="operator")
        assert third["tasks"][0]["version"] == 2
        assert third["tasks"][0]["name"] == "Die goes into the cup"
        assert third["tasks"][0]["frozen"] is False
        assert catalog.library_task("lib_roll_a_die_1", 1)["name"] == "Die into the cup"
        assert catalog.library_task("lib_roll_a_die_1", 1)["frozen"] is True
        assert catalog.library_task("lib_roll_a_die_2")["version"] == 1, "untouched steps keep 1"
        kinds = {row["kind"] for row in catalog.events()}
        assert {"goal_saved", "library_task_version_created", "library_task_edited",
                "library_task_frozen"} <= kinds
        with pytest.raises(KeyError):
            catalog.library_task("lib_roll_a_die_9")


def test_window_is_stored_with_the_version(tmp_path: Path) -> None:
    """Time it stores the confirmed window; it never rewrites a version that has takes."""
    with opened(tmp_path / "workspace") as catalog:
        catalog.save_goal(GOAL, catalog_tasks(seed_payload()), planner_source="built-in planner")
        timed = catalog.set_library_window("lib_roll_a_die_1", 8, 25, runs=[{"duration_s": 12.5}])
        assert timed["version"] == 1
        assert timed["episode_window"] == {"min_episode_s": 8, "max_episode_s": 25}
        assert timed["timing_runs"] == [{"duration_s": 12.5}]
        catalog.freeze_library_task("lib_roll_a_die_1", 1)
        retimed = catalog.set_library_window("lib_roll_a_die_1", 10, 30)
        assert retimed["version"] == 2 and retimed["frozen"] is False
        first = catalog.library_task("lib_roll_a_die_1", 1)
        assert first["episode_window"]["max_episode_s"] == 25
        with pytest.raises(ValueError, match="operator timing required"):
            catalog.set_library_window("lib_roll_a_die_1", 30, 10)


def test_service_plans_saves_groups_exports_and_records_a_library_session(
    tmp_path: Path,
) -> None:
    """Plan, save, see the grouped dropdown, export a fragment, create and open a session."""
    with ConsoleFixture(tmp_path, demo=True) as service:
        before = service.read(["tasks"], {})
        assert [task["task_source"] for task in before] == ["registry"] * 3
        planned = service.mutate(["library", "plan"], {"sentence": "train to roll a dice"})
        assert planned["source"] == "built-in planner" and planned["fallback_reason"] == ""
        assert planned["proposed"]["goal_id"] == "lib_roll_a_die"
        assert planned["warnings"] == []
        assert service.read(["library"], {})["planner"] == {
            "external_configured": False, "source": "built-in planner"}
        asked = service.mutate(["library", "plan"], {"sentence": "do the thing"})
        assert asked["breakdown"] is None and len(asked["questions"]) == 3
        with pytest.raises(ValueError, match="one sentence"):
            service.mutate(["library", "plan"], {"sentence": "x" * 401})
        saved = service.mutate(["library", "save"], {
            "breakdown": planned["breakdown"], "planner_source": planned["source"]})
        assert saved["goal"]["id"] == "lib_roll_a_die" and len(saved["goal"]["tasks"]) == 4
        choices = service.read(["tasks"], {})
        assert [task["task_source"] for task in choices] == ["registry"] * 3 + ["library"] * 4
        library = [task for task in choices if task["task_source"] == "library"]
        assert library[0]["display_name"] == "Roll a die: 1 Die into cup"
        assert library[0]["group"] == "Roll a die" and library[0]["task_id"] == "lib_roll_a_die_1"
        assert library[0]["episode_window"] == {"min_episode_s": None, "max_episode_s": None}
        export = service.read(["library-export", "lib_roll_a_die"], {})
        assert "published registry" in export["note"] and len(export["tasks"]) == 4
        assert export["tasks"][0]["task_id"] == "lib_roll_a_die_1"
        assert export["tasks"][0]["declared_fixed_scene"] is True
        # A library session binds the exact version, freezes it, and opens through the snapshot.
        session = service.create_session({**values(), "task_id": "lib_roll_a_die_1",
                                          "name": "Library run"})
        assert session["task_source"] == "library" and session["goal_id"] == "lib_roll_a_die"
        assert session["library_version"] == 1 and session["chain_position"] == 1
        snapshot = Path(session["task_snapshot_path"])
        assert snapshot.is_file() and snapshot.parent == service.workspace / "library-snapshots"
        stored = load_task_source(snapshot)["lib_roll_a_die_1"]
        assert stored.sha256 == session["task_definition_sha256"]
        assert service.catalog.library_task("lib_roll_a_die_1", 1)["frozen"] is True
        service.mutate(["session", session["id"], "open"], {})
        assert service.state()["active_session"]["id"] == session["id"]
        contract_path = Path(session["dataset_path"]) / "physical_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert contract["task_source"] == "library" and contract["goal_id"] == "lib_roll_a_die"
        assert contract["chain_position"] == 1 and contract["library_version"] == 1
        assert contract["task_text"].startswith("Pick the die from its start zone")
        service.mutate(["session", session["id"], "close"], {})
        # The frozen version cannot be edited in place: saving an edit makes version 2.
        edited = json.loads(json.dumps(planned["breakdown"]))
        edited["sub_tasks"][0]["suggested_takes"] = 30
        resaved = service.mutate(["library", "save"], {
            "breakdown": edited, "planner_source": "operator"})
        assert resaved["goal"]["tasks"][0]["version"] == 2
        assert service.catalog.library_task("lib_roll_a_die_1", 1)["suggested_takes"] == 20
        timed = service.mutate(["library", "lib_roll_a_die_2", "window"], {
            "min_episode_s": 8, "max_episode_s": 25, "runs": []})
        assert timed["episode_window"] == {"min_episode_s": 8, "max_episode_s": 25}
        with pytest.raises(ValueError, match="only library tasks"):
            service.mutate(["library", "die_into_cup", "window"], {
                "min_episode_s": 8, "max_episode_s": 25})


def test_external_planner_is_reported_and_falls_back(tmp_path: Path) -> None:
    """A configured command that fails is reported, and the built-in planner answers."""
    planner = ExternalPlanner((str(tmp_path / "missing-planner"),), timeout_s=1.0)
    with ConsoleFixture(tmp_path / "workspace", demo=True, planner=planner) as service:
        assert service.read(["library"], {})["planner"]["external_configured"] is True
        planned = service.mutate(["library", "plan"], {"sentence": "roll a die"})
        assert planned["source"] == "built-in planner"
        assert "could not be run" in planned["fallback_reason"]

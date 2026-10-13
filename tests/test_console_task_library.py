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
        assert [task["task_source"] for task in before] == ["registry"] * 4
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
        assert [task["task_source"] for task in choices] == ["registry"] * 4 + ["library"] * 4
        library = [task for task in choices if task["task_source"] == "library"]
        assert library[0]["display_name"] == "1. Die into cup"
        assert library[3]["display_name"] == "Optional: Stand the cup up"
        assert library[3]["optional"] is True and library[0]["optional"] is False
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


class FakeArmFeed:
    """A read-only arm feed that rests, moves three times and rests, from a fake clock."""

    def __init__(self) -> None:
        """Lay out the samples in advance so the timing thread just walks through them."""
        import math

        from synria_lerobot.physical_contract import PhysicalState

        rest = (1.55, 1.56, -0.60, 0.10, -0.81, -0.17, 0.0001)
        self.samples: list[PhysicalState] = []
        self.closed = False
        now = 0.0

        def push(values: tuple[float, ...]) -> None:
            nonlocal now
            now += 0.02
            self.samples.append(PhysicalState(values[:6], values[6], now, now, None, now))

        for _ in range(int(3.5 / 0.02)):
            push(rest)
        for move_s in (10.0, 12.0, 9.0):
            count = int(move_s / 0.02)
            for index in range(count):
                phase = index / (count - 1)
                swing = math.radians(40.0) * math.sin(math.pi * phase)
                push((rest[0] + swing, rest[1], rest[2], rest[3], rest[4], rest[5],
                      rest[6] + 0.016 * math.sin(math.pi * phase)))
            for _ in range(int(3.0 / 0.02)):
                push(rest)
        self._index = 0

    def read(self) -> Any:
        """Hand out the next sample; after the last one keep returning it."""
        sample = self.samples[min(self._index, len(self.samples) - 1)]
        self._index += 1
        return sample

    def close(self) -> None:
        """Record that the subscription was released."""
        self.closed = True


def test_time_it_listens_read_only_and_stores_the_confirmed_window(tmp_path: Path) -> None:
    """Three moves are timed from the arm feed, a window is proposed, the operator confirms."""
    import time

    feeds: list[FakeArmFeed] = []

    def factory(topic: str) -> FakeArmFeed:
        assert topic == "/joint_states"
        feed = FakeArmFeed()
        feeds.append(feed)
        return feed

    with ConsoleFixture(tmp_path, timing_source_factory=factory,
                        timing_poll_s=0.0005) as service:
        service.catalog.save_goal(GOAL, catalog_tasks(seed_payload()), planner_source="operator")
        with pytest.raises(ValueError, match="only library tasks"):
            service.mutate(["library", "die_into_cup", "time", "start"], {})
        status = service.mutate(["library", "lib_roll_a_die_1", "time", "start"], {})
        assert status["running"] is True and status["phase"] == "learning rest"
        with pytest.raises(RuntimeError, match="already running"):
            service.mutate(["library", "lib_roll_a_die_1", "time", "start"], {})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = service.read(["library", "lib_roll_a_die_1", "time"], {})
            if len(status["runs"]) >= 3:
                break
            time.sleep(0.05)
        assert len(status["runs"]) == 3, status
        assert [round(run["duration_s"]) for run in status["runs"]] == [10, 12, 9]
        assert all(run["gripper_cycled"] for run in status["runs"])
        assert status["proposed_window"] == {"min_episode_s": 7.0, "max_episode_s": 16.0}
        assert status["error"] is None
        confirmed = service.mutate(["library", "lib_roll_a_die_1", "time", "confirm"],
                                   {"min_episode_s": 8, "max_episode_s": 16})
        assert confirmed["episode_window"] == {"min_episode_s": 8, "max_episode_s": 16}
        assert len(confirmed["timing_runs"]) == 3
        assert feeds and feeds[0].closed, "the subscription is released after confirming"
        kinds = [row["kind"] for row in service.catalog.events()]
        assert "timing_started" in kinds and "library_task_timed" in kinds
        assert service.read(["library", "lib_roll_a_die_1", "time"], {})["running"] is False
    with ConsoleFixture(tmp_path / "demo", demo=True) as demo:
        with pytest.raises(ValueError, match="not available in demo mode"):
            demo.mutate(["library", "lib_roll_a_die_1", "time", "start"], {})


def test_every_label_reads_cleanly_from_the_real_task_payload(tmp_path: Path) -> None:
    """Hand the page's label helper the service's own task list for a saved goal, not a fake."""
    import shutil
    import subprocess

    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    with ConsoleFixture(tmp_path, demo=True) as service:
        planned = service.mutate(["library", "plan"], {"sentence": "train to roll a dice"})
        service.mutate(["library", "save"], {"breakdown": planned["breakdown"],
                                             "planner_source": planned["source"]})
        payload = service.read(["tasks"], {})
    taught = [task for task in payload if task["task_source"] == "library"]
    assert [task["display_name"] for task in taught] == [
        "1. Die into cup", "2. Roll and set the cup down", "3. Die back to start",
        "Optional: Stand the cup up"]
    assert all(task["name"] for task in taught), "the short name travels with the payload"
    payload_path = tmp_path / "tasks.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    static = REPOSITORY / "synria_lerobot" / "console" / "static"
    result = subprocess.run(
        [executable, str(Path(__file__).with_name("console_task_labels.cjs")),
         str(static / "record-banner.js"), str(payload_path)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert "task labels verified" in result.stdout

"""Validated fixed-scene task identities and operator-configured recording windows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_VERSION = "synria_fixed_scene_tasks_v1"
TASK_IDS = ("die_into_cup", "roll_and_dump", "cup_return")
EPISODE_WINDOW_KEYS = ("min_episode_s", "max_episode_s")
TASK_METADATA_KEYS = (
    "task_id", "task_text", "task_definition", "task_definition_sha256", *EPISODE_WINDOW_KEYS,
)


def episode_window_metadata(min_episode_s: Any, max_episode_s: Any) -> dict[str, float]:
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (min_episode_s, max_episode_s)
    ) or min_episode_s > max_episode_s:
        raise ValueError(
            "episode window requires finite positive min <= max; operator timing required"
        )
    return {"min_episode_s": min_episode_s, "max_episode_s": max_episode_s}


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    task_text: str
    success_rule: str
    scene_requirements: tuple[str, ...]
    min_episode_s: float | None
    max_episode_s: float | None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or self.task_id not in TASK_IDS:
            raise ValueError("unknown fixed-scene task id")
        if any(type(value) is not str or not value.strip() for value in (
            self.task_text, self.success_rule,
        )):
            raise ValueError("task text and success rule must be nonempty strings")
        if (
            type(self.scene_requirements) is not tuple or not self.scene_requirements
            or any(type(value) is not str or not value.strip() for value in self.scene_requirements)
        ):
            raise ValueError("task scene requirements must be a nonempty immutable string sequence")
        if self.min_episode_s is not None or self.max_episode_s is not None:
            self.require_configured()

    def require_configured(self) -> dict[str, float]:
        return episode_window_metadata(self.min_episode_s, self.max_episode_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "task_text": self.task_text, "success_rule": self.success_rule,
            "scene_requirements": list(self.scene_requirements), "declared_fixed_scene": True,
            "requires_unobserved_goal": False,
            "episode_window": {"min_episode_s": self.min_episode_s,
                               "max_episode_s": self.max_episode_s},
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "task_text": self.task_text,
            "task_definition": self.as_dict(), "task_definition_sha256": self.sha256,
            **self.require_configured(),
        }

    @classmethod
    def from_metadata(cls, payload: Any) -> TaskDefinition:
        if type(payload) is not dict:
            raise ValueError("task identity metadata is required")
        task = cls.from_dict(payload.get("task_definition"))
        if any(payload.get(key) != value for key, value in task.metadata().items()):
            raise ValueError("task id, text, window or definition hash differs from its snapshot")
        episode_window_metadata(payload.get("min_episode_s"), payload.get("max_episode_s"))
        return task

    @classmethod
    def from_dict(cls, value: Any) -> TaskDefinition:
        keys = {"task_id", "task_text", "success_rule", "scene_requirements",
                "declared_fixed_scene", "requires_unobserved_goal", "episode_window"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("task definition has missing or unknown fields")
        if (
            value["declared_fixed_scene"] is not True
            or value["requires_unobserved_goal"] is not False
        ):
            raise ValueError(
                "only declared fixed-scene tasks without an unobserved goal are supported"
            )
        window = value["episode_window"]
        if type(window) is not dict or set(window) != set(EPISODE_WINDOW_KEYS):
            raise ValueError("task requires an explicit episode window")
        if type(value["scene_requirements"]) is not list:
            raise ValueError("task scene requirements must be a list")
        return cls(
            value["task_id"], value["task_text"], value["success_rule"],
            tuple(value["scene_requirements"]), window["min_episode_s"], window["max_episode_s"],
        )


def load_task_registry(path: Path) -> dict[str, TaskDefinition]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate task registry field: {name}")
            result[name] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    if type(payload) is not dict or set(payload) != {"version", "tasks"} or (
        payload["version"] != REGISTRY_VERSION or type(payload["tasks"]) is not list
    ):
        raise ValueError("unsupported fixed-scene task registry")
    tasks = [TaskDefinition.from_dict(value) for value in payload["tasks"]]
    if len(tasks) != len(TASK_IDS) or {task.task_id for task in tasks} != set(TASK_IDS):
        raise ValueError("registry must contain exactly the three distinct fixed-scene task ids")
    return {task.task_id: task for task in tasks}


def select_recording_task(path: Path, task_id: str) -> TaskDefinition:
    tasks = load_task_registry(path)
    if type(task_id) is not str or task_id not in tasks:
        raise ValueError("unknown fixed-scene task id")
    selected = tasks[task_id]
    selected.require_configured()
    return selected

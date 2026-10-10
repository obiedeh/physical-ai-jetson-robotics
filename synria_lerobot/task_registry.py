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
QUALIFYING = "qualifying"
DISPOSABLE_SMOKE = "disposable_smoke"
SYNTHETIC_DEMO = "synthetic_demo"
TASK_METADATA_KEYS = (
    "task_id",
    "task_text",
    "task_definition",
    "task_definition_sha256",
    *EPISODE_WINDOW_KEYS,
    "recording_purpose",
    "episode_window_override",
)


def recording_purpose(smoke: bool, declared: str | None = None) -> str:
    """Validate explicit nonphysical purpose without relabeling ordinary recordings."""
    if type(smoke) is not bool:
        raise ValueError("smoke must be an explicit boolean")
    if declared == SYNTHETIC_DEMO and not smoke:
        return SYNTHETIC_DEMO
    if declared is not None and declared != (DISPOSABLE_SMOKE if smoke else QUALIFYING):
        raise ValueError("smoke flag differs from the recorded purpose")
    return DISPOSABLE_SMOKE if smoke else QUALIFYING


def episode_window_metadata(min_episode_s: Any, max_episode_s: Any) -> dict[str, float]:
    """Validate positive ordered timing bounds before publishing task metadata."""
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (min_episode_s, max_episode_s)
        )
        or min_episode_s > max_episode_s
    ):
        raise ValueError(
            "episode window requires finite positive min <= max; operator timing required"
        )
    return {"min_episode_s": min_episode_s, "max_episode_s": max_episode_s}


def validate_fixed_scene_schedule(trials: Any, task_id: str) -> None:
    """Freeze marked geometry/start setup; only declared die position/face may vary."""
    if task_id not in TASK_IDS or type(trials) is not list or not trials:
        raise ValueError("fixed physical skill/scene trials required")
    identifiers = set()
    geometry = None
    fields = {
        "scene_id",
        "cup_mark",
        "die_start_zone",
        "landing_tray",
        "start_state",
        "die_position_in_zone",
    }
    for trial in trials:
        if type(trial) is not dict or set(trial) != {"trial_id", "task_id", "scene"}:
            raise ValueError("physical trial needs only a fixed task and scene, no square goals")
        identifier = trial["trial_id"]
        if (
            type(identifier) is not str
            or not identifier
            or not identifier[0].isalnum()
            or not all(
                character.isascii() and (character.isalnum() or character in "_-")
                for character in identifier
            )
            or identifier in identifiers
            or trial["task_id"] != task_id
        ):
            raise ValueError("unique trial id and the same trained task required")
        identifiers.add(identifier)
        scene = trial["scene"]
        if (
            type(scene) is not dict
            or set(scene) != {*fields, "die_face_up"}
            or any(type(scene[key]) is not str or not scene[key].strip() for key in fields)
            or type(scene["die_face_up"]) is not int
            or not 1 <= scene["die_face_up"] <= 6
        ):
            raise ValueError("fixed scene and explicit die position/face variation required")
        current = {key: scene[key] for key in fields - {"die_position_in_zone"}}
        if geometry is not None and current != geometry:
            raise ValueError("trial geometry and start state must stay fixed")
        geometry = current


@dataclass(frozen=True)
class TaskDefinition:
    """Freeze one skill instruction, scene and prospective operator timing window."""
    task_id: str
    task_text: str
    success_rule: str
    scene_requirements: tuple[str, ...]
    min_episode_s: float | None
    max_episode_s: float | None

    def __post_init__(self) -> None:
        """Reject malformed identity, dimensions or timing at the data boundary."""
        if type(self.task_id) is not str or self.task_id not in TASK_IDS:
            raise ValueError("unknown fixed-scene task id")
        if any(
            type(value) is not str or not value.strip()
            for value in (
                self.task_text,
                self.success_rule,
            )
        ):
            raise ValueError("task text and success rule must be nonempty strings")
        if (
            type(self.scene_requirements) is not tuple
            or not self.scene_requirements
            or any(type(value) is not str or not value.strip() for value in self.scene_requirements)
        ):
            raise ValueError("task scene requirements must be a nonempty immutable string sequence")
        if self.min_episode_s is not None or self.max_episode_s is not None:
            self.require_configured()

    def require_configured(self) -> dict[str, float]:
        """Refuse qualifying collection until the operator supplies task timing."""
        return episode_window_metadata(self.min_episode_s, self.max_episode_s)

    def as_dict(self) -> dict[str, Any]:
        """Serialize canonical evidence fields for exact contract and resume comparison."""
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "success_rule": self.success_rule,
            "scene_requirements": list(self.scene_requirements),
            "declared_fixed_scene": True,
            "requires_unobserved_goal": False,
            "episode_window": {
                "min_episode_s": self.min_episode_s,
                "max_episode_s": self.max_episode_s,
            },
        }

    @property
    def sha256(self) -> str:
        """Hash the canonical task snapshot so later registry changes cannot relabel data."""
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def recording_window(self, purpose: str) -> dict[str, float]:
        """Resolve purpose-specific bounds without changing the stored task snapshot."""
        if purpose == QUALIFYING:
            return self.require_configured()
        if purpose == DISPOSABLE_SMOKE:
            return episode_window_metadata(20.0, 20.0)
        if purpose == SYNTHETIC_DEMO:
            return episode_window_metadata(1.0, 30.0)
        raise ValueError("unknown recording purpose")

    def metadata(self, purpose: str = QUALIFYING) -> dict[str, Any]:
        """Attach immutable task identity and explicit nonqualifying window overrides."""
        window = self.recording_window(purpose)
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "task_definition": self.as_dict(),
            "task_definition_sha256": self.sha256,
            **window,
            "recording_purpose": purpose,
            "episode_window_override": (
                {**window, "reason": "disposable smoke; never qualifying"}
                if purpose == DISPOSABLE_SMOKE
                else ({**window, "reason": "synthetic demo; never qualifying"}
                      if purpose == SYNTHETIC_DEMO else None)
            ),
        }

    @classmethod
    def from_metadata(cls, payload: Any) -> TaskDefinition:
        """Validate task snapshot, purpose and hash before trusting persisted evidence."""
        if type(payload) is not dict:
            raise ValueError("task identity metadata is required")
        task = cls.from_dict(payload.get("task_definition"))
        purpose = payload.get("recording_purpose")
        if type(purpose) is not str:
            raise ValueError("explicit recording purpose is required")
        if any(
            key not in payload or payload[key] != value
            for key, value in task.metadata(purpose).items()
        ):
            raise ValueError("task id, text, window or definition hash differs from its snapshot")
        episode_window_metadata(payload.get("min_episode_s"), payload.get("max_episode_s"))
        override = payload["episode_window_override"]
        if override is not None:
            episode_window_metadata(override.get("min_episode_s"), override.get("max_episode_s"))
        if "smoke" in payload and recording_purpose(payload["smoke"], purpose) != purpose:
            raise ValueError("smoke flag differs from the recorded purpose")
        return task

    @classmethod
    def from_dict(cls, value: Any) -> TaskDefinition:
        """Construct a validated object from explicitly serialized evidence."""
        keys = {
            "task_id",
            "task_text",
            "success_rule",
            "scene_requirements",
            "declared_fixed_scene",
            "requires_unobserved_goal",
            "episode_window",
        }
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
            value["task_id"],
            value["task_text"],
            value["success_rule"],
            tuple(value["scene_requirements"]),
            window["min_episode_s"],
            window["max_episode_s"],
        )


def load_task_registry(path: Path) -> dict[str, TaskDefinition]:
    """Read the complete fixed-scene registry while rejecting ambiguous duplicate fields."""
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate object fields instead of silently selecting one value."""
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate task registry field: {name}")
            result[name] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    if (
        type(payload) is not dict
        or set(payload) != {"version", "tasks"}
        or (payload["version"] != REGISTRY_VERSION or type(payload["tasks"]) is not list)
    ):
        raise ValueError("unsupported fixed-scene task registry")
    tasks = [TaskDefinition.from_dict(value) for value in payload["tasks"]]
    if len(tasks) != len(TASK_IDS) or {task.task_id for task in tasks} != set(TASK_IDS):
        raise ValueError("registry must contain exactly the three distinct fixed-scene task ids")
    return {task.task_id: task for task in tasks}


def select_recording_task(
    path: Path,
    task_id: str,
    *,
    purpose: str = QUALIFYING,
) -> TaskDefinition:
    """Select and validate one task for its declared recording purpose."""
    tasks = load_task_registry(path)
    if type(task_id) is not str or task_id not in tasks:
        raise ValueError("unknown fixed-scene task id")
    selected = tasks[task_id]
    selected.recording_window(purpose)
    return selected

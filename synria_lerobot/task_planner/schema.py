"""One validated shape for a task breakdown, shared by every planner backend.

A breakdown is a goal in the operator's own words plus an ordered chain of
sub-tasks. Each sub-task can be trained on its own and chained later, so the
schema carries what the chain check needs: what is held at the start and the
end, which objects are grasped, and whether the fixed camera can confirm the
end state. Nothing here touches hardware or storage.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "synria_task_breakdown_v1"
KINDS = ("skill", "reset", "recovery")
PRIMITIVES = (
    "pick", "place", "drop into", "pour out", "shake", "push", "open", "close", "return to ready",
)
MAX_TEXT = 400
MAX_STEPS = 12
_SLUG = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

BREAKDOWN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Synria task breakdown",
    "type": "object",
    "required": ["schema_version", "goal", "sub_tasks"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "goal": {
            "type": "object",
            "required": ["title", "sentence"],
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                "sentence": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                "rationale": {"type": "string", "maxLength": 2000},
            },
        },
        "sub_tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_STEPS,
            "items": {
                "type": "object",
                "required": [
                    "name", "instruction", "start_state", "end_state", "success_rule",
                    "scene_before_take", "kind", "objects_grasped", "holds_at_start",
                    "holds_at_end", "camera_can_confirm", "suggested_takes",
                ],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "instruction": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    "start_state": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    "end_state": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    "success_rule": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    "scene_before_take": {
                        "type": "array", "minItems": 1, "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    },
                    "kind": {"enum": list(KINDS)},
                    "optional": {"type": "boolean"},
                    "objects_grasped": {
                        "type": "array", "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 60},
                    },
                    "holds_at_start": {"type": "boolean"},
                    "holds_at_end": {"type": "boolean"},
                    "camera_can_confirm": {"type": "boolean"},
                    "suggested_takes": {"type": "integer", "minimum": 1, "maximum": 500},
                    "primitives": {
                        "type": "array", "maxItems": 12, "items": {"enum": list(PRIMITIVES)},
                    },
                },
            },
        },
    },
}


class BreakdownError(ValueError):
    """A breakdown payload does not match the schema; say exactly where."""


@dataclass(frozen=True)
class SubTask:
    """One trainable step with the facts the chain check and the recorder need."""

    name: str
    instruction: str
    start_state: str
    end_state: str
    success_rule: str
    scene_before_take: tuple[str, ...]
    kind: str
    objects_grasped: tuple[str, ...]
    holds_at_start: bool
    holds_at_end: bool
    camera_can_confirm: bool
    suggested_takes: int
    optional: bool = False
    primitives: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize with lists so the result round-trips through JSON unchanged."""
        value = asdict(self)
        for key in ("scene_before_take", "objects_grasped", "primitives"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class Goal:
    """What the operator typed, with a title and the reason the chain is cut where it is."""

    title: str
    sentence: str
    rationale: str = ""


@dataclass(frozen=True)
class Breakdown:
    """A goal and its ordered sub-tasks."""

    goal: Goal
    sub_tasks: tuple[SubTask, ...]
    schema_version: str = SCHEMA_VERSION
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the documented schema."""
        return {
            "schema_version": self.schema_version,
            "goal": {"title": self.goal.title, "sentence": self.goal.sentence,
                     "rationale": self.goal.rationale},
            "sub_tasks": [step.as_dict() for step in self.sub_tasks],
        }


def _text(value: Any, where: str, *, maximum: int = MAX_TEXT) -> str:
    """Require a nonempty string that fits on a card."""
    if type(value) is not str or not value.strip():
        raise BreakdownError(f"{where} must be a nonempty string")
    if len(value) > maximum:
        raise BreakdownError(f"{where} is longer than {maximum} characters")
    return value.strip()


def _strings(value: Any, where: str, *, minimum: int, maximum: int) -> tuple[str, ...]:
    """Require a list of short nonempty strings within the allowed count."""
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise BreakdownError(f"{where} must list between {minimum} and {maximum} items")
    return tuple(_text(item, f"{where}[{index}]", maximum=MAX_TEXT)
                 for index, item in enumerate(value))


def _flag(value: Any, where: str) -> bool:
    """Require an explicit boolean, never a truthy stand-in."""
    if type(value) is not bool:
        raise BreakdownError(f"{where} must be true or false")
    return value


def validate_sub_task(value: Any, where: str = "sub_task") -> SubTask:
    """Validate one step against the schema and return it as a frozen object."""
    if type(value) is not dict:
        raise BreakdownError(f"{where} must be an object")
    required = set(BREAKDOWN_SCHEMA["properties"]["sub_tasks"]["items"]["required"])
    allowed = set(BREAKDOWN_SCHEMA["properties"]["sub_tasks"]["items"]["properties"])
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise BreakdownError(f"{where} is missing {', '.join(missing)}")
    if unknown:
        raise BreakdownError(f"{where} has unknown fields {', '.join(unknown)}")
    kind = value["kind"]
    if kind not in KINDS:
        raise BreakdownError(f"{where}.kind must be one of {', '.join(KINDS)}")
    takes = value["suggested_takes"]
    if type(takes) is not int or not 1 <= takes <= 500:
        raise BreakdownError(f"{where}.suggested_takes must be a whole number from 1 to 500")
    primitives = tuple(value.get("primitives", []))
    if any(item not in PRIMITIVES for item in primitives) or len(primitives) > 12:
        raise BreakdownError(f"{where}.primitives must use the known vocabulary")
    return SubTask(
        name=_text(value["name"], f"{where}.name", maximum=80),
        instruction=_text(value["instruction"], f"{where}.instruction"),
        start_state=_text(value["start_state"], f"{where}.start_state"),
        end_state=_text(value["end_state"], f"{where}.end_state"),
        success_rule=_text(value["success_rule"], f"{where}.success_rule"),
        scene_before_take=_strings(value["scene_before_take"], f"{where}.scene_before_take",
                                   minimum=1, maximum=8),
        kind=kind,
        objects_grasped=_strings(value["objects_grasped"], f"{where}.objects_grasped",
                                 minimum=0, maximum=4),
        holds_at_start=_flag(value["holds_at_start"], f"{where}.holds_at_start"),
        holds_at_end=_flag(value["holds_at_end"], f"{where}.holds_at_end"),
        camera_can_confirm=_flag(value["camera_can_confirm"], f"{where}.camera_can_confirm"),
        suggested_takes=takes,
        optional=_flag(value.get("optional", False), f"{where}.optional"),
        primitives=primitives,
    )


def validate_breakdown(payload: Any) -> Breakdown:
    """Validate a whole payload, from any backend, before it reaches a screen or a store."""
    if type(payload) is not dict:
        raise BreakdownError("breakdown must be an object")
    allowed = set(BREAKDOWN_SCHEMA["properties"])
    unknown = sorted(set(payload) - allowed)
    missing = sorted(set(BREAKDOWN_SCHEMA["required"]) - set(payload))
    if missing:
        raise BreakdownError(f"breakdown is missing {', '.join(missing)}")
    if unknown:
        raise BreakdownError(f"breakdown has unknown fields {', '.join(unknown)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise BreakdownError(f"breakdown schema_version must be {SCHEMA_VERSION}")
    goal = payload["goal"]
    if type(goal) is not dict or set(goal) - {"title", "sentence", "rationale"}:
        raise BreakdownError("goal must have title, sentence and optionally rationale")
    steps = payload["sub_tasks"]
    if type(steps) is not list or not 1 <= len(steps) <= MAX_STEPS:
        raise BreakdownError(f"sub_tasks must list between 1 and {MAX_STEPS} steps")
    return Breakdown(
        goal=Goal(
            title=_text(goal.get("title"), "goal.title"),
            sentence=_text(goal.get("sentence"), "goal.sentence"),
            rationale=str(goal.get("rationale", ""))[:2000],
        ),
        sub_tasks=tuple(validate_sub_task(step, f"sub_tasks[{index}]")
                        for index, step in enumerate(steps)),
    )


def slug(value: str) -> str:
    """Lowercase identifier from free text, safe for ids and file names."""
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text or "task"


def is_slug(value: str) -> bool:
    """Whether a string is already a valid identifier."""
    return bool(_SLUG.match(value))

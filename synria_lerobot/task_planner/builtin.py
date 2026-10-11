"""The built-in planner: works offline from a primitive vocabulary and a small recipe set.

It recognises a few goal shapes. When it cannot parse a sentence it does not guess:
it returns the questions to ask (which object, which container, which destination)
and plans again once they are answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from synria_lerobot.task_planner.schema import Breakdown, Goal, validate_breakdown

READY = "gripper empty, arm at ready"
AT_READY = "Gripper empty, arm at ready."

STOP_WORDS = {
    "a", "an", "the", "to", "train", "teach", "learn", "arm", "robot", "it", "how", "please",
    "and", "then", "into", "in", "onto", "on", "from", "of", "with", "up", "down", "out",
}


@dataclass(frozen=True)
class Question:
    """One thing the planner needs to know before it can break the goal down."""

    key: str
    prompt: str
    example: str


@dataclass(frozen=True)
class PlanResult:
    """What a planner returns: a breakdown, or the questions that stand in its way."""

    source: str
    breakdown: Breakdown | None
    questions: tuple[Question, ...] = ()
    recipe: str = ""
    fallback_reason: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the console API."""
        return {
            "source": self.source,
            "breakdown": self.breakdown.as_dict() if self.breakdown else None,
            "questions": [{"key": q.key, "prompt": q.prompt, "example": q.example}
                          for q in self.questions],
            "recipe": self.recipe,
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
        }


def _step(name: str, instruction: str, start: str, end: str, success: str, scene: list[str],
          kind: str, objects: list[str], takes: int, primitives: list[str],
          optional: bool = False) -> dict[str, Any]:
    """Build one schema-shaped step; every built-in step starts and ends empty at ready."""
    return {
        "name": name, "instruction": instruction, "start_state": start, "end_state": end,
        "success_rule": success, "scene_before_take": scene, "kind": kind,
        "objects_grasped": objects, "holds_at_start": False, "holds_at_end": False,
        "camera_can_confirm": True, "suggested_takes": takes, "primitives": primitives,
        "optional": optional,
    }


def _breakdown(
    title: str, sentence: str, rationale: str, steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap steps in the documented envelope."""
    return {
        "schema_version": "synria_task_breakdown_v1",
        "goal": {"title": title, "sentence": sentence, "rationale": rationale},
        "sub_tasks": steps,
    }


def roll_a_die_recipe(sentence: str) -> dict[str, Any]:
    """The seed breakdown for rolling a die with a cup, a start zone and a landing tray."""
    start_scene = ["The cup is upright on its mark.", "The die is in its start zone.",
                   "The tray is on its mark.", AT_READY]
    rationale = (
        "Three steps, each starting and ending with the gripper empty at ready, so each can be "
        "trained alone and chained in any order. The cup is picked up, shaken, poured and set "
        "down in one step so no hand-off happens while the gripper holds the cup. The reset "
        "step puts the die back so cycles repeat without a hand reset."
    )
    steps = [
        _step("Die into cup",
              "Pick the die from its start zone, release it into the upright cup on its mark, "
              "and return to ready.",
              f"Die in its start zone, cup upright on its mark, {READY}",
              f"Die inside the cup, cup upright on its mark, {READY}",
              "The die is inside the cup.",
              start_scene, "skill", ["die"], 20, ["pick", "drop into", "return to ready"]),
        _step("Roll and set the cup down",
              "Pick up the cup, shake it, pour the die onto the tray, set the cup upright back "
              "on its mark, and return to ready.",
              f"Die inside the cup, cup upright on its mark, {READY}",
              f"Die at rest on the tray, cup upright on its mark, {READY}",
              "The die is at rest on the tray and the cup is upright on its mark.",
              ["The cup is upright on its mark with the die inside.", "The tray is on its mark.",
               AT_READY],
              "skill", ["cup"], 20, ["pick", "shake", "pour out", "place", "return to ready"]),
        _step("Die back to start",
              "Pick the die from the tray, place it in the start zone, and return to ready.",
              f"Die at rest on the tray, cup upright on its mark, {READY}",
              f"Die in its start zone, cup upright on its mark, {READY}",
              "The die is in the start zone.",
              ["The die is on the tray.", "The cup is upright on its mark.", AT_READY],
              "reset", ["die"], 15, ["pick", "place", "return to ready"]),
        _step("Stand the cup up",
              "Stand a tipped cup upright on its mark, and return to ready.",
              f"Cup tipped over near its mark, {READY}",
              f"Cup upright on its mark, {READY}",
              "The cup is upright on its mark.",
              ["The cup is tipped over near its mark.", AT_READY],
              "recovery", ["cup"], 10, ["pick", "place", "return to ready"], optional=True),
    ]
    return _breakdown("Roll a die", sentence, rationale, steps)


def put_into_recipe(sentence: str, item: str, container: str) -> dict[str, Any]:
    """A two-step breakdown: put an item into a container, then put it back."""
    name = item.capitalize()
    rationale = (
        "One grasp per step and a reset step so the cycle can repeat. Each step starts and "
        "ends with the gripper empty at ready."
    )
    steps = [
        _step(f"{name} into {container}",
              f"Pick the {item} from its start zone, release it into the {container} on its "
              "mark, and return to ready.",
              f"{name} in its start zone, {container} on its mark, {READY}",
              f"{name} inside the {container}, {container} on its mark, {READY}",
              f"The {item} is inside the {container}.",
              [f"The {container} is on its mark.", f"The {item} is in its start zone.", AT_READY],
              "skill", [item], 20, ["pick", "drop into", "return to ready"]),
        _step(f"{name} back to start",
              f"Pick the {item} from the {container}, place it in its start zone, and return "
              "to ready.",
              f"{name} inside the {container}, {container} on its mark, {READY}",
              f"{name} in its start zone, {container} on its mark, {READY}",
              f"The {item} is in its start zone.",
              [f"The {item} is inside the {container}.", AT_READY],
              "reset", [item], 15, ["pick", "place", "return to ready"]),
    ]
    return _breakdown(f"Put the {item} into the {container}", sentence, rationale, steps)


def move_to_recipe(sentence: str, item: str, destination: str) -> dict[str, Any]:
    """A two-step breakdown: move an item to a marked destination, then back."""
    name = item.capitalize()
    steps = [
        _step(f"{name} to {destination}",
              f"Pick the {item} from its start zone, place it on the {destination} mark, and "
              "return to ready.",
              f"{name} in its start zone, {READY}",
              f"{name} on the {destination} mark, {READY}",
              f"The {item} is at rest on the {destination} mark.",
              [f"The {item} is in its start zone.", f"The {destination} mark is clear.", AT_READY],
              "skill", [item], 20, ["pick", "place", "return to ready"]),
        _step(f"{name} back to start",
              f"Pick the {item} from the {destination} mark, place it in its start zone, and "
              "return to ready.",
              f"{name} on the {destination} mark, {READY}",
              f"{name} in its start zone, {READY}",
              f"The {item} is in its start zone.",
              [f"The {item} is on the {destination} mark.", AT_READY],
              "reset", [item], 15, ["pick", "place", "return to ready"]),
    ]
    return _breakdown(f"Move the {item} to the {destination}", sentence,
                      "One grasp per step and a reset step so the cycle can repeat.", steps)


_DIE = re.compile(r"\b(roll|rolls|rolling|throw|throws|throwing)\b.*\b(die|dice)\b",
                  re.IGNORECASE)
_PUT = re.compile(
    r"\b(?:put|drop|place)\s+(?:the\s+|a\s+)?(?P<item>[a-z]+)\s+(?:into|in|inside)\s+"
    r"(?:the\s+|a\s+)?(?P<container>[a-z]+)", re.IGNORECASE,
)
_MOVE = re.compile(
    r"\b(?:move|take|bring|carry)\s+(?:the\s+|a\s+)?(?P<item>[a-z]+)\s+(?:to|onto|on)\s+"
    r"(?:the\s+|a\s+)?(?P<destination>[a-z]+)", re.IGNORECASE,
)


def guided_questions() -> tuple[Question, ...]:
    """What to ask when a sentence cannot be parsed."""
    return (
        Question("object", "Which object does the arm pick up?", "die"),
        Question("container", "Which container does it go into, if any?", "cup"),
        Question("destination",
                 "Which marked destination does it go to, if there is no container?", "tray"),
    )


def builtin_plan(sentence: str, answers: dict[str, str] | None = None) -> PlanResult:
    """Break a sentence down from the recipe set, or return the questions that are needed."""
    source = "built-in planner"
    text = " ".join(str(sentence or "").split())
    if not text:
        return PlanResult(source, None, guided_questions(),
                          notes=("Type what you want to teach the arm, in one sentence.",))
    clean = {key: " ".join(str(value).lower().split())
             for key, value in (answers or {}).items() if str(value).strip()}
    if _DIE.search(text):
        return PlanResult(source, validate_breakdown(roll_a_die_recipe(text)), recipe="roll_a_die")
    match = _PUT.search(text)
    if match and match.group("item").lower() not in STOP_WORDS:
        recipe = put_into_recipe(text, match.group("item").lower(),
                                 match.group("container").lower())
        return PlanResult(source, validate_breakdown(recipe), recipe="put_into")
    match = _MOVE.search(text)
    if match and match.group("item").lower() not in STOP_WORDS:
        recipe = move_to_recipe(text, match.group("item").lower(),
                                match.group("destination").lower())
        return PlanResult(source, validate_breakdown(recipe), recipe="move_to")
    item = clean.get("object")
    if item and clean.get("container"):
        recipe = put_into_recipe(text, item, clean["container"])
        return PlanResult(source, validate_breakdown(recipe), recipe="put_into",
                          notes=("Planned from your answers.",))
    if item and clean.get("destination"):
        recipe = move_to_recipe(text, item, clean["destination"])
        return PlanResult(source, validate_breakdown(recipe), recipe="move_to",
                          notes=("Planned from your answers.",))
    return PlanResult(source, None, guided_questions(),
                      notes=("I could not work out the object and where it goes from that "
                             "sentence.",))


def goal_from(sentence: str, breakdown: Breakdown) -> Goal:
    """Keep the operator's sentence with the goal, whatever the planner titled it."""
    return Goal(breakdown.goal.title, sentence, breakdown.goal.rationale)

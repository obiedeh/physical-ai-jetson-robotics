"""The breakdown rules every planner must follow, and the chain check that enforces them.

The rules are shown to the operator as guidance and passed to the external planner
command verbatim. The chain check turns them into warnings on a proposed breakdown;
it never edits the breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from synria_lerobot.task_planner.schema import Breakdown, SubTask

RULES: tuple[str, ...] = (
    "Every sub-task starts and ends with the gripper empty and the arm at the ready pose, "
    "so sub-tasks stand alone and chain in any order.",
    "One object grasped per sub-task.",
    "One success condition that the overhead camera can show.",
    "The end state of each sub-task is the start state of the next.",
    "Aim for roughly 8 to 25 seconds per sub-task.",
    "Scene fixed by marks at first.",
    "Add reset sub-tasks that put the scene back so cycles can repeat.",
    "Never define a sub-task whose success depends on something the cameras cannot see.",
)

UNSEEN_WORDS = re.compile(
    r"\b(feel|feels|felt|weight|weighs|heavy|light|warm|cold|smell|sound|inside the drawer|"
    r"behind|under the lid|hidden|out of view|out of sight)\b", re.IGNORECASE,
)


@dataclass(frozen=True)
class ChainWarning:
    """One thing the operator should look at before saving, with the step it concerns."""

    step: int | None
    code: str
    message: str

    def as_dict(self) -> dict[str, object]:
        """Serialize for the result screen."""
        return {"step": self.step, "code": self.code, "message": self.message}


def _normal(text: str) -> str:
    """Compare states loosely: lowercase, single spaces, no trailing full stop."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def _step_warnings(index: int, step: SubTask) -> list[ChainWarning]:
    """Warnings that concern one step on its own."""
    found: list[ChainWarning] = []
    if step.holds_at_start:
        found.append(ChainWarning(index, "holds_at_start",
                                  f"Step {index} starts holding something; every step starts "
                                  "with the gripper empty."))
    if step.holds_at_end:
        found.append(ChainWarning(index, "holds_at_end",
                                  f"Step {index} ends holding something; every step ends with "
                                  "the gripper empty at ready."))
    if len(step.objects_grasped) > 1:
        found.append(ChainWarning(index, "two_objects",
                                  f"Step {index} grasps {len(step.objects_grasped)} objects "
                                  f"({', '.join(step.objects_grasped)}); grasp one object "
                                  "per step."))
    if not step.camera_can_confirm or UNSEEN_WORDS.search(step.success_rule):
        found.append(ChainWarning(index, "unseen_goal",
                                  f"Step {index} has a success condition the overhead camera "
                                  f"may not show: {step.success_rule}"))
    if "ready" not in _normal(step.end_state):
        found.append(ChainWarning(index, "end_not_ready",
                                  f"Step {index} does not say the arm returns to the ready "
                                  "pose at the end."))
    return found


def chain_check(breakdown: Breakdown) -> list[ChainWarning]:
    """Warn where a breakdown breaks a rule; the operator decides what to do about it."""
    warnings: list[ChainWarning] = []
    steps = breakdown.sub_tasks
    for index, step in enumerate(steps, start=1):
        warnings.extend(_step_warnings(index, step))
    chain = [step for step in steps if not step.optional]
    for index in range(1, len(chain)):
        previous, current = chain[index - 1], chain[index]
        if _normal(previous.end_state) != _normal(current.start_state):
            position = steps.index(current) + 1
            warnings.append(ChainWarning(
                position, "chain_gap",
                f"The end of step {steps.index(previous) + 1} is not the start of step "
                f"{position}: \"{previous.end_state}\" then \"{current.start_state}\".",
            ))
    if chain and not any(step.kind == "reset" for step in chain):
        warnings.append(ChainWarning(
            None, "no_reset",
            "No reset step puts the scene back, so cycles cannot repeat without a hand reset.",
        ))
    return warnings


def describe_rules() -> list[str]:
    """The rules as a list, for screens and for the external command's input."""
    return list(RULES)


def step_summary(step: SubTask) -> str:
    """One line per step for logs and the activity record."""
    return f"{step.name} ({step.kind}): {step.instruction}"

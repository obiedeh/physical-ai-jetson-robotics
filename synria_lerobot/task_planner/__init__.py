"""Break a sentence such as "train to roll a dice" into sub-tasks that can be trained alone.

Two backends sit behind one interface and one validated schema: a built-in planner
that works offline from a primitive vocabulary and a small recipe set, and an optional
external task planner command configured only at launch. Nothing in this package
touches hardware, storage or the network.
"""

from synria_lerobot.task_planner.builtin import PlanResult, Question, builtin_plan, guided_questions
from synria_lerobot.task_planner.external import (
    ENVIRONMENT_VARIABLE,
    EXTERNAL_SOURCE,
    ExternalPlanner,
    PlannerError,
    plan_goal,
)
from synria_lerobot.task_planner.rules import RULES, ChainWarning, chain_check, describe_rules
from synria_lerobot.task_planner.schema import (
    BREAKDOWN_SCHEMA,
    KINDS,
    PRIMITIVES,
    SCHEMA_VERSION,
    Breakdown,
    BreakdownError,
    Goal,
    SubTask,
    slug,
    validate_breakdown,
    validate_sub_task,
)

__all__ = [
    "BREAKDOWN_SCHEMA", "ENVIRONMENT_VARIABLE", "EXTERNAL_SOURCE", "KINDS", "PRIMITIVES",
    "RULES", "SCHEMA_VERSION", "Breakdown", "BreakdownError", "ChainWarning", "ExternalPlanner",
    "Goal", "PlanResult", "PlannerError", "Question", "SubTask", "builtin_plan", "chain_check",
    "describe_rules", "guided_questions", "plan_goal", "slug", "validate_breakdown",
    "validate_sub_task",
]

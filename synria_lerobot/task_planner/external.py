"""An optional external task planner command behind the same interface as the built-in planner.

The command is configured only through one environment variable read at launch,
never from the browser. The console passes the goal sentence, the rules and the
schema on stdin as JSON, with no shell, waits for a bounded time, expects JSON on
stdout and validates it. On any failure the built-in planner answers instead and
the result says so.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass

from synria_lerobot.task_planner.builtin import PlanResult, builtin_plan
from synria_lerobot.task_planner.rules import describe_rules
from synria_lerobot.task_planner.schema import BREAKDOWN_SCHEMA, BreakdownError, validate_breakdown

ENVIRONMENT_VARIABLE = "SYNRIA_TASK_PLANNER_COMMAND"
DEFAULT_TIMEOUT_S = 20.0
MAX_OUTPUT_BYTES = 200_000
EXTERNAL_SOURCE = "external task planner command"


class PlannerError(RuntimeError):
    """The external command did not produce a valid breakdown."""


@dataclass(frozen=True)
class ExternalPlanner:
    """Run a configured local command and validate what it returns."""

    command: tuple[str, ...]
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_environment(
        cls, value: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> ExternalPlanner | None:
        """Parse the launch-time setting; empty means the feature stays off."""
        if value is None or not value.strip():
            return None
        parts = tuple(shlex.split(value))
        if not parts:
            return None
        return cls(parts, timeout_s)

    def request(self, sentence: str, answers: dict[str, str] | None = None) -> bytes:
        """The exact bytes the command receives on stdin."""
        payload = {
            "goal_sentence": sentence,
            "answers": dict(answers or {}),
            "rules": describe_rules(),
            "schema": BREAKDOWN_SCHEMA,
            "reply": "Write one JSON object matching schema to stdout and nothing else.",
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def plan(self, sentence: str, answers: dict[str, str] | None = None) -> PlanResult:
        """Run the command once; raise PlannerError on timeout, bad exit, bad JSON or bad shape."""
        try:
            completed = subprocess.run(
                list(self.command), input=self.request(sentence, answers), capture_output=True,
                timeout=self.timeout_s, check=False, shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise PlannerError(
                f"the planner command did not answer within {self.timeout_s:g} s"
            ) from error
        except (OSError, ValueError) as error:
            raise PlannerError(f"the planner command could not be run: {error}") from error
        if completed.returncode != 0:
            raise PlannerError(f"the planner command exited with status {completed.returncode}")
        if len(completed.stdout) > MAX_OUTPUT_BYTES:
            raise PlannerError("the planner command returned more output than allowed")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PlannerError("the planner command did not return valid JSON") from error
        try:
            breakdown = validate_breakdown(payload)
        except BreakdownError as error:
            raise PlannerError(
                f"the planner command's answer does not match the schema: {error}"
            ) from error
        return PlanResult(EXTERNAL_SOURCE, breakdown)


def plan_goal(sentence: str, *, external: ExternalPlanner | None = None,
              answers: dict[str, str] | None = None) -> PlanResult:
    """Plan with the external command when configured, else built in; always say which answered."""
    if external is None:
        return builtin_plan(sentence, answers)
    try:
        return external.plan(sentence, answers)
    except PlannerError as error:
        fallback = builtin_plan(sentence, answers)
        return PlanResult(fallback.source, fallback.breakdown, fallback.questions, fallback.recipe,
                          fallback_reason=str(error), notes=fallback.notes)

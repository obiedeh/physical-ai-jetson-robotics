"""The task planner with fakes only: schema, recipes, chain check, external command and fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from synria_lerobot.task_planner import (
    EXTERNAL_SOURCE,
    RULES,
    Breakdown,
    BreakdownError,
    ExternalPlanner,
    PlannerError,
    builtin_plan,
    chain_check,
    plan_goal,
    slug,
    validate_breakdown,
)
from synria_lerobot.task_planner.builtin import roll_a_die_recipe


def seed() -> Breakdown:
    """The seed recipe as a validated breakdown."""
    return validate_breakdown(roll_a_die_recipe("train to roll a dice"))


def test_seed_recipe_matches_the_recommended_breakdown() -> None:
    """Three chained steps plus an optional recovery, each empty at ready at both ends."""
    breakdown = seed()
    assert breakdown.goal.title == "Roll a die"
    assert breakdown.goal.sentence == "train to roll a dice"
    names = [step.name for step in breakdown.sub_tasks]
    assert names == ["Die into cup", "Roll and set the cup down", "Die back to start",
                     "Stand the cup up"]
    kinds = [step.kind for step in breakdown.sub_tasks]
    assert kinds == ["skill", "skill", "reset", "recovery"]
    assert [step.optional for step in breakdown.sub_tasks] == [False, False, False, True]
    assert all(len(step.objects_grasped) == 1 for step in breakdown.sub_tasks)
    assert all(not step.holds_at_start and not step.holds_at_end
               for step in breakdown.sub_tasks)
    assert breakdown.sub_tasks[1].success_rule == (
        "The die is at rest on the tray and the cup is upright on its mark."
    )
    assert "no hand-off happens while the gripper holds the cup" in breakdown.goal.rationale
    assert chain_check(breakdown) == [], "the seed passes its own chain check"
    assert validate_breakdown(breakdown.as_dict()) == breakdown, "round trip through JSON shape"


@pytest.mark.parametrize("sentence", [
    "train to roll a dice", "Roll a die", "teach the arm to throw dice",
])
def test_builtin_planner_recognises_the_die_goal(sentence: str) -> None:
    """Different phrasings of the die goal all reach the seed recipe and keep the sentence."""
    result = builtin_plan(sentence)
    assert result.source == "built-in planner" and result.recipe == "roll_a_die"
    assert result.breakdown is not None and result.breakdown.goal.sentence == sentence
    assert result.questions == ()


def test_builtin_planner_handles_put_and_move_sentences() -> None:
    """Simple put and move sentences become a skill step and a reset step."""
    put = builtin_plan("put the block into the box")
    assert put.recipe == "put_into" and put.breakdown is not None
    assert [step.kind for step in put.breakdown.sub_tasks] == ["skill", "reset"]
    first = put.breakdown.sub_tasks[0].instruction
    assert "block" in first and "box" in first
    assert chain_check(put.breakdown) == []
    move = builtin_plan("move the cup to the tray")
    assert move.recipe == "move_to" and move.breakdown is not None
    assert move.breakdown.sub_tasks[0].success_rule == "The cup is at rest on the tray mark."
    assert chain_check(move.breakdown) == []


def test_builtin_planner_asks_instead_of_guessing() -> None:
    """An unparsable sentence returns the three guided questions and no breakdown."""
    result = builtin_plan("do the thing we discussed")
    assert result.breakdown is None
    assert [question.key for question in result.questions] == ["object", "container", "destination"]
    assert builtin_plan("").breakdown is None
    answered = builtin_plan("do the thing we discussed", {"object": "Die", "container": "cup"})
    assert answered.breakdown is not None and answered.recipe == "put_into"
    assert answered.breakdown.sub_tasks[0].name == "Die into cup"
    moved = builtin_plan("do the thing", {"object": "cup", "destination": "shelf"})
    assert moved.recipe == "move_to"


def test_schema_rejects_bad_shapes_with_a_place() -> None:
    """Every refusal names the field so a bad external answer can be reported plainly."""
    good = seed().as_dict()
    with pytest.raises(BreakdownError, match="schema_version"):
        validate_breakdown({**good, "schema_version": "other"})
    with pytest.raises(BreakdownError, match="missing goal"):
        validate_breakdown({"schema_version": good["schema_version"],
                            "sub_tasks": good["sub_tasks"]})
    broken = json.loads(json.dumps(good))
    broken["sub_tasks"][0]["kind"] = "core"
    with pytest.raises(BreakdownError, match=r"sub_tasks\[0\].kind"):
        validate_breakdown(broken)
    broken = json.loads(json.dumps(good))
    broken["sub_tasks"][1]["holds_at_end"] = "yes"
    with pytest.raises(BreakdownError, match=r"sub_tasks\[1\].holds_at_end must be true or false"):
        validate_breakdown(broken)
    broken = json.loads(json.dumps(good))
    broken["sub_tasks"][2]["extra"] = 1
    with pytest.raises(BreakdownError, match="unknown fields extra"):
        validate_breakdown(broken)
    broken = json.loads(json.dumps(good))
    broken["sub_tasks"][0]["suggested_takes"] = 0
    with pytest.raises(BreakdownError, match="suggested_takes"):
        validate_breakdown(broken)
    broken = json.loads(json.dumps(good))
    broken["sub_tasks"][0]["primitives"] = ["juggle"]
    with pytest.raises(BreakdownError, match="vocabulary"):
        validate_breakdown(broken)
    with pytest.raises(BreakdownError, match="must be an object"):
        validate_breakdown([good])


def test_chain_check_warns_on_each_rule() -> None:
    """Gaps, two objects, holding at either end, unseen goals and no reset step are all named."""
    payload = seed().as_dict()
    steps = payload["sub_tasks"]
    steps[0]["holds_at_end"] = True
    steps[1]["objects_grasped"] = ["cup", "die"]
    steps[1]["start_state"] = "Something else entirely"
    steps[2]["camera_can_confirm"] = False
    steps[2]["kind"] = "skill"
    steps[3]["success_rule"] = "The cup feels heavy."
    warnings = chain_check(validate_breakdown(payload))
    codes = {(warning.step, warning.code) for warning in warnings}
    assert (1, "holds_at_end") in codes
    assert (2, "two_objects") in codes
    assert (2, "chain_gap") in codes
    assert (3, "unseen_goal") in codes
    assert (4, "unseen_goal") in codes
    assert (None, "no_reset") in codes
    assert all(warning.message for warning in warnings)
    assert all(isinstance(warning.as_dict(), dict) for warning in warnings)


def test_chain_check_ignores_optional_steps_in_the_chain() -> None:
    """An optional recovery step does not have to join the chain, but a required step does."""
    payload = seed().as_dict()
    payload["sub_tasks"][3]["optional"] = False
    warnings = chain_check(validate_breakdown(payload))
    assert any(warning.code == "chain_gap" and warning.step == 4 for warning in warnings)


def test_rules_are_the_eight_breakdown_rules() -> None:
    """The guidance shown to the operator is the rule list the planner follows."""
    assert len(RULES) == 8
    assert RULES[0].startswith("Every sub-task starts and ends with the gripper empty")
    assert RULES[-1].startswith("Never define a sub-task whose success depends")


def test_slug_is_an_identifier() -> None:
    """Free text becomes a lowercase identifier and empty text does not become empty."""
    assert slug("Roll a die!") == "roll_a_die"
    assert slug("  ") == "task"


def write_command(tmp_path: Path, name: str, body: str) -> ExternalPlanner:
    """Write a tiny Python script and wrap it as the external planner command."""
    script = tmp_path / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    return ExternalPlanner((sys.executable, "-I", str(script)), timeout_s=5.0)


def test_external_command_receives_goal_rules_and_schema(tmp_path: Path) -> None:
    """The command gets JSON on stdin with no shell, and its JSON answer is validated."""
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(seed().as_dict()), encoding="utf-8")
    planner = write_command(tmp_path, "echo_plan", (
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['goal_sentence'] == 'stack the rings'\n"
        "assert len(request['rules']) == 8 and request['schema']['title']\n"
        "assert request['answers'] == {'object': 'ring'}\n"
        f"print(open({str(answer)!r}, encoding='utf-8').read())\n"
    ))
    result = plan_goal("stack the rings", external=planner, answers={"object": "ring"})
    assert result.source == EXTERNAL_SOURCE
    assert result.breakdown is not None and result.fallback_reason == ""
    assert planner.command[0] == sys.executable, "the command runs directly, not through a shell"


@pytest.mark.parametrize(("body", "reason"), [
    ("print('not json')", "valid JSON"),
    ("import json; print(json.dumps({'schema_version': 'synria_task_breakdown_v1', "
     "'goal': {'title': 'x', 'sentence': 'y'}, 'sub_tasks': []}))", "schema"),
    ("import sys; sys.exit(3)", "status 3"),
    ("import time; time.sleep(30)", "did not answer"),
])
def test_external_failures_fall_back_to_the_builtin_planner(
    tmp_path: Path, body: str, reason: str,
) -> None:
    """Bad JSON, a bad shape, a bad exit and a timeout all fall back, and the result says why."""
    planner = ExternalPlanner((sys.executable, "-I", "-c", body), timeout_s=1.0)
    with pytest.raises(PlannerError, match=reason):
        planner.plan("roll a die")
    result = plan_goal("roll a die", external=planner)
    assert result.source == "built-in planner" and result.recipe == "roll_a_die"
    assert reason.split()[0] in result.fallback_reason


def test_external_planner_is_off_unless_configured() -> None:
    """An empty or missing setting keeps the feature off; a set one is parsed without a shell."""
    assert ExternalPlanner.from_environment(None) is None
    assert ExternalPlanner.from_environment("   ") is None
    planner = ExternalPlanner.from_environment("/usr/bin/env python3 /opt/plan.py --fast")
    assert planner is not None
    assert planner.command == ("/usr/bin/env", "python3", "/opt/plan.py", "--fast")
    assert plan_goal("roll a die", external=None).source == "built-in planner"


def test_missing_command_falls_back(tmp_path: Path) -> None:
    """A command that does not exist is a planner error, not a crash."""
    planner = ExternalPlanner((str(tmp_path / "no-such-planner"),), timeout_s=1.0)
    with pytest.raises(PlannerError, match="could not be run"):
        planner.plan("roll a die")
    assert plan_goal("roll a die", external=planner).breakdown is not None

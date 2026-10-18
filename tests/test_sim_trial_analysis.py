"""Sorting trial records into failure modes and writing the table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synria_lerobot.sim_trial_analysis import (
    CUP_UNTOUCHED,
    HELD_BY_SAFETY,
    LIFTED_AND_DROPPED,
    MODES,
    NEVER_REACHED,
    NUDGED,
    PLACED_BUT_NOT_HELD,
    REACHED_NO_GRASP,
    SUCCESS,
    classify_trial,
    render_markdown,
    summarize_report,
    summarize_reports,
    superseded_reports,
    write_summary,
)


def trial(**fields: Any) -> dict[str, Any]:
    """A trial record with every field a mode could need, overridden by ``fields``."""
    base = {"success": False, "steps": 100, "holds": 0, "max_lift_m": 0.0, "recoveries": 0,
            "min_gripper_cup_distance_m": None, "min_cup_target_distance_m": None}
    return {**base, **fields}


def report(trials: list[dict[str, Any]], *, score: str = "lift", policy_id: str = "p",
           step: int = 20000) -> dict[str, Any]:
    """A campaign report around the given trials."""
    config = {"lift_m": 0.03, "place_radius_m": 0.03}
    return {"kind": "simulation_policy_trials", "lane": "simulation", "utc": "2026-10-18T00:00:00Z",
            "task_id": "pick_cup", "score": score, "config": config,
            "policy": {"policy_id": policy_id, "step": step}, "trials": trials,
            "successes": sum(1 for t in trials if t["success"]),
            "success_rate": sum(1 for t in trials if t["success"]) / len(trials)}


def test_every_mode_is_reached_in_order() -> None:
    """Success first, then holds, then the lift, then the recorded reach."""
    assert classify_trial(trial(success=True, holds=99), 0.03) == SUCCESS
    assert classify_trial(trial(holds=51), 0.03) == HELD_BY_SAFETY
    assert classify_trial(trial(holds=50, max_lift_m=0.04), 0.03) == LIFTED_AND_DROPPED
    assert classify_trial(trial(max_lift_m=0.04, min_cup_target_distance_m=0.02), 0.03,
                          place_radius_m=0.03) == PLACED_BUT_NOT_HELD
    assert classify_trial(trial(max_lift_m=0.04, min_cup_target_distance_m=0.02), 0.03,
                          place_radius_m=None) == LIFTED_AND_DROPPED
    assert classify_trial(trial(max_lift_m=0.01), 0.03) == NUDGED
    assert classify_trial(trial(max_lift_m=0.001), 0.03) == CUP_UNTOUCHED
    assert classify_trial(trial(min_gripper_cup_distance_m=0.02), 0.03) == REACHED_NO_GRASP
    assert classify_trial(trial(min_gripper_cup_distance_m=0.08), 0.03) == NEVER_REACHED


def test_summary_counts_every_mode_and_totals_over_reports(tmp_path: Path) -> None:
    """Counts per report, totals over reports, the table shows only the modes that occurred."""
    first = report([trial(success=True), trial(min_gripper_cup_distance_m=0.1),
                    trial(min_gripper_cup_distance_m=0.01), trial(holds=80)])
    second = report([trial(max_lift_m=0.05), trial()], policy_id="q", step=5000)
    summary = summarize_report(first)
    assert summary["modes"][SUCCESS] == 1 and summary["modes"][NEVER_REACHED] == 1
    assert summary["modes"][REACHED_NO_GRASP] == 1 and summary["modes"][HELD_BY_SAFETY] == 1
    assert summary["reach_recorded"] and set(summary["modes"]) == set(MODES)
    paths = [tmp_path / "trials_a.json", tmp_path / "trials_b.json"]
    paths[0].write_text(json.dumps(first))
    paths[1].write_text(json.dumps(second))
    combined = summarize_reports(paths)
    assert combined["trials"] == 6 and combined["totals"][LIFTED_AND_DROPPED] == 1
    assert combined["totals"][CUP_UNTOUCHED] == 1 and combined["physical_evidence"] is False
    assert [row["policy_id"] for row in combined["rows"]] == ["p", "q"]
    table = render_markdown(combined)
    assert "| p | 20000 | 4 |" in table and "| all current | | 6 |" in table
    assert NUDGED not in table and PLACED_BUT_NOT_HELD not in table
    # A study's superseded campaign is listed, marked, and left out of the totals.
    (tmp_path / "study.json").write_text(json.dumps(
        {"rows": [{"trials": {"report": str(paths[1])},
                   "superseded_trials": [{"report": str(paths[0])}]}]}))
    assert superseded_reports(tmp_path) == {paths[0].resolve()}
    assert superseded_reports(tmp_path / "none") == set()
    marked = summarize_reports(paths, superseded=superseded_reports(tmp_path))
    assert marked["trials"] == 2 and marked["totals"][SUCCESS] == 0
    assert "| p (superseded) | 20000 | 4 |" in render_markdown(marked)
    json_path, markdown_path = write_summary(combined, tmp_path / "out")
    assert json.loads(json_path.read_text())["trials"] == 6
    assert markdown_path.read_text() == table
    with pytest.raises(ValueError, match="not a simulation trial report"):
        summarize_report({**first, "lane": "physical"})
    with pytest.raises(ValueError, match="no trial reports"):
        summarize_reports([])


def test_held_share_is_reported_beside_the_modes() -> None:
    """Holds short of half a trial still show up as a share of all steps."""
    summary = summarize_report(report([trial(holds=30, steps=150), trial(holds=0, steps=150),
                                       trial(holds=15, steps=100)]))
    assert summary["holds"] == 45 and summary["steps"] == 400
    assert summary["held_share"] == pytest.approx(45 / 400)
    text = render_markdown(summarize_reports_from([summary]))
    assert "| steps held |" in text and "[95% interval]" in text
    assert "| p | 20000 | 3 |" in text and "| 11.2% |" in text
    assert "0% [0, 56]" in text


def test_wilson_interval_widths() -> None:
    """Twenty trials span about forty points; fifty about twenty-five; none gives zeros."""
    from synria_lerobot.sim_trial_analysis import wilson_interval

    low, high = wilson_interval(7, 20)
    assert 0.18 < low < 0.19 and 0.56 < high < 0.58
    low, high = wilson_interval(28, 50)
    assert 0.42 < low < 0.43 and 0.68 < high < 0.70
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(0, 50)[0] == 0.0 and wilson_interval(50, 50)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def summarize_reports_from(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A summary record around already summarized rows, as the writer would see it."""
    from collections import Counter

    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row["modes"])
    return {"rows": [{"report": "r", "superseded": False, **row} for row in rows],
            "totals": {mode: totals.get(mode, 0) for mode in MODES},
            "trials": sum(row["trials"] for row in rows)}

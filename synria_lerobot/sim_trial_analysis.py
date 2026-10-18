"""Failure modes of simulation trial campaigns, counted from their reports.

A success rate says how often a policy picked the cup; it does not say what the
failures looked like. Every trial record carries enough to tell them apart: the
cup's highest lift says whether the cup was ever lifted, moved or left alone; the
closest gripper-to-cup distance, when the campaign recorded it, says whether the
gripper ever reached the cup; the hold count says whether the policy spent the trial
refused by the safety client. This module sorts every trial of one or more reports
into one of those modes and writes a table per report, so the next change can be
aimed at the mode that dominates. For the place score it also counts how many trials
got the cup to the mark. Simulation lane only; never physical evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_KIND = "simulation_policy_trials"
# A cup that rose less than this is counted as not moved: the physics drive settles by
# about a millimetre after every reset.
NUDGE_M = 0.002
# The gripper came within this of the cup's centre: a reach, whether or not it closed.
REACH_M = 0.03
SUCCESS = "success"
LIFTED_AND_DROPPED = "lifted, then dropped"
PLACED_BUT_NOT_HELD = "reached the mark, not held there"
NUDGED = "moved the cup, no lift"
REACHED_NO_GRASP = "reached the cup, no grasp"
NEVER_REACHED = "never reached the cup"
CUP_UNTOUCHED = "cup untouched, reach not recorded"
HELD_BY_SAFETY = "held by the safety client most of the trial"
MODES = (SUCCESS, LIFTED_AND_DROPPED, PLACED_BUT_NOT_HELD, NUDGED, REACHED_NO_GRASP,
         NEVER_REACHED, CUP_UNTOUCHED, HELD_BY_SAFETY)


def classify_trial(trial: dict[str, Any], lift_m: float, *, place_radius_m: float | None = None,
                   nudge_m: float = NUDGE_M, reach_m: float = REACH_M) -> str:
    """The one mode a trial record falls into, checked in the order of :data:`MODES`.

    A success is a success whatever else happened. Then a trial the safety client held
    for more than half its steps is a hold, since the policy barely ran. Then the cup's
    highest lift: at or above the lift height means it was lifted and dropped (for the
    place score, a cup that came within the mark's radius after that is counted apart);
    above the nudge height means it was moved without a lift. A cup that never moved is
    sorted by the closest gripper distance when the campaign recorded one.
    """
    if trial["success"]:
        return SUCCESS
    if trial["steps"] > 0 and trial["holds"] > trial["steps"] / 2:
        return HELD_BY_SAFETY
    lift = float(trial["max_lift_m"])
    if lift >= lift_m:
        closest = trial.get("min_cup_target_distance_m")
        if place_radius_m is not None and closest is not None and closest <= place_radius_m:
            return PLACED_BUT_NOT_HELD
        return LIFTED_AND_DROPPED
    if lift > nudge_m:
        return NUDGED
    distance = trial.get("min_gripper_cup_distance_m")
    if distance is None:
        return CUP_UNTOUCHED
    return REACHED_NO_GRASP if distance <= reach_m else NEVER_REACHED


def summarize_report(report: dict[str, Any], *, nudge_m: float = NUDGE_M,
                     reach_m: float = REACH_M) -> dict[str, Any]:
    """Mode counts for one campaign report, with the rule values that sorted them.

    The share of steps the safety client held is reported beside the modes: a hold
    short of half a trial does not make a mode, but a policy that spends a sixth of
    every trial pulled back from a joint bound is pressing the operator limits, and
    the share says so even when the trial is sorted by what the cup did.
    """
    if report.get("kind") != REPORT_KIND or report.get("lane") != "simulation":
        raise ValueError("not a simulation trial report")
    config = report["config"]
    place_radius = config.get("place_radius_m") if report.get("score") == "place" else None
    modes = [classify_trial(trial, float(config["lift_m"]), place_radius_m=place_radius,
                            nudge_m=nudge_m, reach_m=reach_m) for trial in report["trials"]]
    counts = Counter(modes)
    holds = sum(int(trial.get("holds", 0)) for trial in report["trials"])
    steps = sum(int(trial.get("steps", 0)) for trial in report["trials"])
    recorded_reach = any(trial.get("min_gripper_cup_distance_m") is not None
                         for trial in report["trials"])
    return {
        "policy_id": report["policy"].get("policy_id"),
        "step": report["policy"].get("step"),
        "utc": report["utc"],
        "task_id": report.get("task_id"),
        "score": report.get("score", "lift"),
        "trials": len(modes),
        "successes": report["successes"],
        "success_rate": report["success_rate"],
        "modes": {mode: counts.get(mode, 0) for mode in MODES},
        "reach_recorded": recorded_reach,
        "recoveries": sum(int(trial.get("recoveries", 0)) for trial in report["trials"]),
        "holds": holds,
        "steps": steps,
        "held_share": holds / steps if steps else 0.0,
        "rules": {"lift_m": config["lift_m"], "nudge_m": nudge_m, "reach_m": reach_m,
                  "place_radius_m": place_radius},
    }


def superseded_reports(study: Path) -> set[Path]:
    """The campaign reports a study's rows list as superseded by a rerun, resolved."""
    record = study / "study.json"
    if not record.exists():
        return set()
    rows = json.loads(record.read_text(encoding="utf-8")).get("rows", [])
    return {Path(entry["report"]).resolve() for row in rows
            for entry in row.get("superseded_trials") or []}


def summarize_reports(paths: list[Path], *, nudge_m: float = NUDGE_M, reach_m: float = REACH_M,
                      superseded: set[Path] | None = None) -> dict[str, Any]:
    """Summaries for every report, in the order given, plus the totals over all of them.

    Reports named in ``superseded`` are shown but left out of the totals: a study keeps
    a campaign that was rerun under a changed rule, and the totals should count one
    campaign per policy.
    """
    if not paths:
        raise ValueError("no trial reports given")
    rows = []
    old = {path.resolve() for path in superseded or set()}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"report": str(path), "superseded": path.resolve() in old,
                     **summarize_report(report, nudge_m=nudge_m, reach_m=reach_m)})
    totals: Counter[str] = Counter()
    for row in rows:
        if not row["superseded"]:
            totals.update(row["modes"])
    return {
        "kind": "simulation_trial_failure_modes",
        "lane": "simulation",
        "physical_evidence": False,
        "utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "totals": {mode: totals.get(mode, 0) for mode in MODES},
        "trials": sum(row["trials"] for row in rows if not row["superseded"]),
        "note": ("Simulation lane; sorted from the campaign reports; superseded campaigns "
                 "listed but not totalled; never physical evidence."),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """One row per campaign, one column per mode, totals last."""
    used = [mode for mode in MODES if any(row["modes"][mode] for row in summary["rows"])]
    lines = [
        "# Failure modes of the simulation trial campaigns",
        "",
        "Simulation lane; never physical evidence. Every trial of every report is sorted",
        "into one mode. A lift counts from the report's own lift height; a nudge is a",
        f"rise above {NUDGE_M * 1000:.0f} mm; a reach is the gripper centre within "
        f"{REACH_M * 100:.0f} cm of the cup's",
        "centre, known only for campaigns that recorded the gripper pose; a hold is a",
        "trial the safety client refused for more than half its steps; the last column",
        "is the share of all steps the client held, whatever the mode. A campaign a",
        "study reran under a changed rule is marked superseded and left out of the totals.",
        "",
        "| policy | step | trials | " + " | ".join(used) + " | steps held |",
        "|---|---|---|" + "---|" * (len(used) + 1),
    ]
    for row in summary["rows"]:
        cells = " | ".join(str(row["modes"][mode]) for mode in used)
        name = row["policy_id"] + (" (superseded)" if row.get("superseded") else "")
        held = f"{100 * row.get('held_share', 0.0):.1f}%"
        lines.append(f"| {name} | {row['step']} | {row['trials']} | {cells} | {held} |")
    totals = " | ".join(str(summary["totals"][mode]) for mode in used)
    lines.append(f"| all current | | {summary['trials']} | {totals} | |")
    lines.append("")
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Write the JSON record and the Markdown table; both are rewritten on every run."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "failure_modes.json"
    markdown_path = directory / "FAILURE_MODES.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, markdown_path

"""Audit a recorded dataset against the operator joint and gripper limits.

The recording quality gate accepts a frame whose joints sit up to the recording
tolerance beyond the operator bounds, because the bounds are whole degrees and a
resting joint can read a fraction of a degree past them. The policy client has no
such tolerance: it clamps every commanded joint to the exact bound and holds when
the observed state is outside it. A demonstration that presses a bound therefore
teaches the policy a target the client will clip, and in the simulation lane the arm
that sags past the bound is held and pulled back on every such step. This module
counts, per episode and per joint, how many frames press a bound, how many lie
outside it and how many lie beyond the recording tolerance, so a recording batch can
be judged before training and a campaign's holds can be traced to the takes that
caused them. It reads the dataset's parquet files directly and never changes them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synria_lerobot.quality_gates import PhysicalLimits

AUDIT_KIND = "dataset_limits_audit"
# A joint this close to a bound, on either side, counts as pressing it.
DEFAULT_PRESS_DEG = 0.5
# LeRobot stores states and actions as 32-bit floats, so a value recorded exactly on a
# bound can read a few billionths past it; that much is never counted as outside.
STORAGE_SLACK = 1e-6
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EPISODE_KEY = "episode_index"


def _rows(root: Path) -> Iterable[tuple[int, list[float], list[float]]]:
    """Yield (episode index, state, action) for every frame of every data file, in order."""
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("dataset has no data files")
    # Imported after the file check so an empty dataset is refused without pyarrow.
    from pyarrow import parquet

    for path in files:
        table = parquet.read_table(path, columns=[EPISODE_KEY, STATE_KEY, ACTION_KEY])
        for row in table.to_pylist():
            yield (int(row[EPISODE_KEY]), [float(v) for v in row[STATE_KEY]],
                   [float(v) for v in row[ACTION_KEY]])


def _side(value: float, lower: float, upper: float, margin: float) -> str | None:
    """Which bound the value presses within ``margin``, or None."""
    if value <= lower + margin:
        return "lower"
    if value >= upper - margin:
        return "upper"
    return None


def audit_dataset(root: Path, limits: PhysicalLimits, gripper_type: str, *,
                  press_deg: float = DEFAULT_PRESS_DEG) -> dict[str, Any]:
    """Count the frames of every episode that press, leave or exceed the tolerance of a bound.

    A frame presses a joint bound when its commanded or observed value lies within
    ``press_deg`` of that bound on either side. It is outside when the value is past
    the exact bound, and beyond tolerance when it is past the bound by more than the
    limits file's recording tolerance, which is what the quality gate would reject.
    The gripper is checked against its exact stroke on both counts. Nothing here
    judges a take; the counts are reported so the operator can.
    """
    if not math.isfinite(press_deg) or press_deg < 0:
        raise ValueError("press_deg must be a finite, non-negative number of degrees")
    if gripper_type not in limits.gripper_limits_m:
        raise ValueError(f"unknown gripper type {gripper_type!r}")
    margin = math.radians(press_deg)
    tolerance = limits.recording_tolerance_rad
    names = (*limits.joint_names, "Gripper")
    bounds = (*limits.joint_limits_rad, limits.gripper_limits_m[gripper_type])
    episodes: dict[int, dict[str, Any]] = {}
    for episode, state, action in _rows(root):
        if len(state) < 7 or len(action) != 7:
            raise ValueError(f"episode {episode}: a frame does not carry seven values")
        record = episodes.setdefault(episode, {
            "frames": 0, "pressing": 0, "outside": 0, "beyond_tolerance": 0,
            "joints": {name: {"pressing": 0, "outside": 0, "beyond_tolerance": 0,
                              "lower": 0, "upper": 0,
                              "action_min": math.inf, "action_max": -math.inf,
                              "state_min": math.inf, "state_max": -math.inf}
                       for name in names},
        })
        record["frames"] += 1
        pressing = outside = beyond = False
        for index, name in enumerate(names):
            lower, upper = bounds[index]
            joint = record["joints"][name]
            commanded, observed = action[index], state[index]
            joint["action_min"] = min(joint["action_min"], commanded)
            joint["action_max"] = max(joint["action_max"], commanded)
            joint["state_min"] = min(joint["state_min"], observed)
            joint["state_max"] = max(joint["state_max"], observed)
            slack = 0.0 if name == "Gripper" else tolerance
            press_margin = 0.0 if name == "Gripper" else margin
            sides = {_side(v, lower, upper, press_margin) for v in (commanded, observed)}
            sides.discard(None)
            if sides:
                joint["pressing"] += 1
                pressing = True
                for side in sides:
                    joint[side] += 1
            if any(not lower - STORAGE_SLACK <= v <= upper + STORAGE_SLACK
                   for v in (commanded, observed)):
                joint["outside"] += 1
                outside = True
            if any(not lower - slack - STORAGE_SLACK <= v <= upper + slack + STORAGE_SLACK
                   for v in (commanded, observed)):
                joint["beyond_tolerance"] += 1
                beyond = True
        record["pressing"] += pressing
        record["outside"] += outside
        record["beyond_tolerance"] += beyond
    if not episodes:
        raise ValueError("dataset has no frames")
    frames = sum(record["frames"] for record in episodes.values())
    joints = {}
    for index, name in enumerate(names):
        lower, upper = bounds[index]
        per = [record["joints"][name] for record in episodes.values()]
        joints[name] = {
            "lower": lower, "upper": upper,
            "episodes_pressing": sum(1 for j in per if j["pressing"]),
            "frames_pressing": sum(j["pressing"] for j in per),
            "frames_outside": sum(j["outside"] for j in per),
            "frames_beyond_tolerance": sum(j["beyond_tolerance"] for j in per),
            "lower_presses": sum(j["lower"] for j in per),
            "upper_presses": sum(j["upper"] for j in per),
            "action_min": min(j["action_min"] for j in per),
            "action_max": max(j["action_max"] for j in per),
            "state_min": min(j["state_min"] for j in per),
            "state_max": max(j["state_max"] for j in per),
        }
    return {
        "kind": AUDIT_KIND,
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "gripper_type": gripper_type,
        "limits": {"verified_by": limits.verified_by, "verified_on": limits.verified_on,
                   "recording_tolerance_deg": limits.recording_tolerance_deg},
        "press_deg": press_deg,
        "episodes": {str(index): episodes[index] for index in sorted(episodes)},
        "episode_count": len(episodes),
        "frames": frames,
        "episodes_pressing": sum(1 for r in episodes.values() if r["pressing"]),
        "frames_pressing": sum(r["pressing"] for r in episodes.values()),
        "frames_outside": sum(r["outside"] for r in episodes.values()),
        "frames_beyond_tolerance": sum(r["beyond_tolerance"] for r in episodes.values()),
        "joints": joints,
        "rules": {
            "pressing": (f"a commanded or observed joint within {press_deg:g} deg of an "
                         "operator bound, on either side; the gripper only at its stroke"),
            "outside": ("a commanded or observed value past the exact operator bound, beyond "
                    f"the {STORAGE_SLACK:g} storage slack of 32-bit floats"),
            "beyond_tolerance": ("past the bound by more than the recording tolerance; "
                                 "what the recording quality gate rejects"),
        },
        "note": ("Counts from the dataset's own frames against the operator limits; the policy "
                 "client clamps commands to the exact bounds and holds on an observed state "
                 "outside them, with no recording tolerance."),
    }


def _deg(value: float) -> str:
    """Radians as whole-tenth degrees for the table."""
    return f"{math.degrees(value):.1f}"


def render_markdown(audit: dict[str, Any]) -> str:
    """A joint table, then one row per episode that pressed any bound."""
    tolerance = audit["limits"]["recording_tolerance_deg"]
    lines = [
        "# Dataset limits audit",
        "",
        f"Dataset `{audit['dataset_root']}`, gripper {audit['gripper_type']}, against the",
        f"operator limits verified by {audit['limits']['verified_by'] or 'nobody'} on "
        f"{audit['limits']['verified_on'] or 'no date'}.",
        f"A frame presses a bound when a commanded or observed joint is within "
        f"{audit['press_deg']:g} deg of it;",
        "it is outside when past the exact bound; it is beyond tolerance when past the",
        f"bound by more than the {tolerance:g} deg recording tolerance. The policy client "
        "clamps every",
        "command to the exact bound and holds on an observed state outside it.",
        "",
        f"Episodes: {audit['episode_count']}; frames: {audit['frames']}; episodes pressing a "
        f"bound: {audit['episodes_pressing']}; frames pressing: {audit['frames_pressing']}; "
        f"frames outside: {audit['frames_outside']}; frames beyond tolerance: "
        f"{audit['frames_beyond_tolerance']}.",
        "",
        "| joint | bounds | episodes pressing | frames pressing | lower / upper | "
        "frames outside | beyond tolerance | commanded range | observed range |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, joint in audit["joints"].items():
        if name == "Gripper":
            bounds = f"{joint['lower']:.3f} to {joint['upper']:.3f} m"
            commanded = f"{joint['action_min']:.4f} to {joint['action_max']:.4f} m"
            observed = f"{joint['state_min']:.4f} to {joint['state_max']:.4f} m"
        else:
            bounds = f"{_deg(joint['lower'])} to {_deg(joint['upper'])} deg"
            commanded = f"{_deg(joint['action_min'])} to {_deg(joint['action_max'])} deg"
            observed = f"{_deg(joint['state_min'])} to {_deg(joint['state_max'])} deg"
        lines.append(
            f"| {name} | {bounds} | {joint['episodes_pressing']} | {joint['frames_pressing']} | "
            f"{joint['lower_presses']} / {joint['upper_presses']} | {joint['frames_outside']} | "
            f"{joint['frames_beyond_tolerance']} | {commanded} | {observed} |"
        )
    pressing = [(index, record) for index, record in audit["episodes"].items()
                if record["pressing"]]
    lines.append("")
    if not pressing:
        lines.append("No episode pressed a bound.")
    else:
        lines.extend([
            "| episode | frames | pressing | outside | beyond tolerance | joints pressed |",
            "|---|---|---|---|---|---|",
        ])
        for index, record in pressing:
            pressed = ", ".join(
                f"{name} {side}" for name, joint in record["joints"].items()
                for side in ("lower", "upper") if joint[side]
            )
            lines.append(
                f"| {index} | {record['frames']} | {record['pressing']} | {record['outside']} | "
                f"{record['beyond_tolerance']} | {pressed} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_audit(audit: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Write the JSON record and the Markdown table; both are rewritten on every run."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "limits_audit.json"
    markdown_path = directory / "LIMITS_AUDIT.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, markdown_path

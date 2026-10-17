"""Verify the configured Synria joint limits against the physical stops, read-only.

The operator jogs each joint to its physical stop with the leader arm while this
script subscribes to the follower's joint state. One key marks each stop. The
measured angles are compared with ``config/synria_limits.yaml`` and the result is
written to ``reports/limits/verification_<timestamp>.json`` and printed as a table.

Pass rule, per joint and per side: the measured physical stop lies at or beyond the
configured bound, within ``--tolerance-deg``. A stop that is reached before the
configured bound means the limit gate would allow a command past the real stop, so
that side fails. The installed gripper passes when its closed and open readings are
within ``--gripper-tolerance-m`` of 0 and of the configured stroke.

Nothing is ever published. The script creates a depth-one joint-state subscriber and
no publisher; ``--dry-run`` replaces the subscriber with a fake follower so the flow
and the report format are exercised without hardware.

Usage::

    python scripts/verify_limits.py --operator "Your Name" --gripper 50mm
    python scripts/verify_limits.py --operator "Your Name" --gripper 50mm --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.physical_contract import DRIVER_JOINT_NAMES, PhysicalState  # noqa: E402
from synria_lerobot.quality_gates import PhysicalLimits, load_limits  # noqa: E402

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY / "reports" / "limits"
GRIPPER_TYPES = ("50mm", "100mm")
KEY_HELP = "m = mark this stop, r = re-read the current angle, s = skip, a = abort"
PASS_RULE = (
    "measured stop at or beyond the configured bound within the tolerance; gripper closed and "
    "open readings within the gripper tolerance of 0 and of the configured stroke"
)


class FollowerSource(Protocol):
    """The one capability this script needs from the follower: a fresh state on demand."""

    def read(self) -> PhysicalState:
        """Return the newest follower state."""
        ...

    def close(self) -> None:
        """Release the subscription."""
        ...


class LiveFollowerSource:
    """A read-only joint-state subscriber; created lazily so dry runs never import ROS."""

    def __init__(self, topic: str) -> None:
        """Open the depth-one subscriber on ``topic``; no publisher is created."""
        from synria_lerobot.recorder import RosJointStateSource

        self._source = RosJointStateSource(topic, node_name="synria_limits_verification")

    def read(self) -> PhysicalState:
        """Return the newest follower state, waiting briefly for the first sample."""
        return self._source.read()

    def close(self) -> None:
        """Shut the subscriber and its executor down."""
        self._source.close()


class FakeFollowerSource:
    """A scripted follower for dry runs and tests: it holds one pose per stop.

    Like a real arm resting at a stop, repeated reads return the same state. The pose
    moves on to the next scripted state only when ``advance`` is called, which ``run``
    does after each mark or skip.
    """

    def __init__(self, states: list[PhysicalState]) -> None:
        """Serve ``states`` one stop at a time, repeating the last one when exhausted."""
        if not states:
            raise ValueError("at least one fake state is required")
        self._states = list(states)
        self._index = 0
        self.closed = False

    def read(self) -> PhysicalState:
        """Return the state for the current stop."""
        return self._states[min(self._index, len(self._states) - 1)]

    def advance(self) -> None:
        """Move the scripted arm to the next stop."""
        self._index += 1

    def close(self) -> None:
        """Record closure so tests can assert the source was released."""
        self.closed = True


def dry_run_states(limits: PhysicalLimits, gripper: str) -> list[PhysicalState]:
    """States that reach every configured bound exactly, in the order the stops are marked.

    The dry run marks Joint1 min, Joint1 max, ..., Joint6 max, then gripper closed and open.
    Each state places the joint under test at its bound and every other joint at zero.
    """
    stroke = limits.gripper_limits_m[gripper][1]
    states: list[PhysicalState] = []
    stamp = 0.0
    for index in range(6):
        for side in (0, 1):
            positions = [0.0] * 6
            positions[index] = limits.joint_limits_rad[index][side]
            states.append(PhysicalState(tuple(positions), 0.0, stamp, stamp,
                                        ros_arrival_stamp_s=stamp))
            stamp += 1.0
    for gripper_m in (0.0, stroke):
        states.append(PhysicalState((0.0,) * 6, gripper_m, stamp, stamp,
                                    ros_arrival_stamp_s=stamp))
        stamp += 1.0
    return states


@dataclass(frozen=True)
class Stop:
    """One physical stop to mark: a joint side or a gripper end."""

    name: str
    side: str
    configured: float
    unit: str

    @property
    def label(self) -> str:
        """Human wording for the prompt."""
        if self.name == "Gripper":
            return f"gripper fully {self.side}"
        direction = "negative" if self.side == "min" else "positive"
        return f"{self.name} {direction} stop ({self.side})"


def stops_for(limits: PhysicalLimits, gripper: str) -> list[Stop]:
    """The marking order: every joint min then max, then the installed gripper closed and open."""
    stops: list[Stop] = []
    for name, (low, high) in zip(limits.joint_names, limits.joint_limits_rad, strict=True):
        stops.append(Stop(name, "min", math.degrees(low), "deg"))
        stops.append(Stop(name, "max", math.degrees(high), "deg"))
    low_m, high_m = limits.gripper_limits_m[gripper]
    stops.append(Stop("Gripper", "closed", low_m, "m"))
    stops.append(Stop("Gripper", "open", high_m, "m"))
    return stops


def reading_for(stop: Stop, state: PhysicalState) -> float:
    """The follower value that corresponds to one stop, in the stop's unit."""
    if stop.name == "Gripper":
        return state.gripper_m
    index = DRIVER_JOINT_NAMES.index(stop.name)
    return math.degrees(state.joint_positions_rad[index])


def _advance(source: FollowerSource) -> None:
    """Tell a scripted follower to move to its next stop; a live follower has no such hook."""
    advance = getattr(source, "advance", None)
    if callable(advance):
        advance()


def default_read_key() -> str:
    """Read one key: raw mode on a POSIX terminal, else the first character of a line."""
    if sys.stdin.isatty():
        try:
            import termios
            import tty

            descriptor = sys.stdin.fileno()
            saved = termios.tcgetattr(descriptor)
            try:
                tty.setcbreak(descriptor)
                return sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
        except (ImportError, OSError):
            try:
                import msvcrt

                return str(msvcrt.getwch()).lower()  # type: ignore[attr-defined]
            except (ImportError, OSError):
                pass
    line = sys.stdin.readline()
    if not line:
        return "a"
    return line.strip()[:1].lower()


def git_sha() -> str:
    """The repository commit the verification ran on, or an explicit unknown marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def evaluate_joints(
    limits: PhysicalLimits, measured: dict[tuple[str, str], float | None], tolerance_deg: float,
) -> list[dict[str, Any]]:
    """Compare each joint's measured stops with its configured bounds, in degrees."""
    rows: list[dict[str, Any]] = []
    for name, (low, high) in zip(limits.joint_names, limits.joint_limits_rad, strict=True):
        yaml_min, yaml_max = math.degrees(low), math.degrees(high)
        measured_min = measured.get((name, "min"))
        measured_max = measured.get((name, "max"))
        delta_min = None if measured_min is None else measured_min - yaml_min
        delta_max = None if measured_max is None else measured_max - yaml_max
        passed = (
            measured_min is not None and measured_max is not None
            and measured_min <= yaml_min + tolerance_deg
            and measured_max >= yaml_max - tolerance_deg
        )
        rows.append({
            "joint": name,
            "yaml_min_deg": round(yaml_min, 3), "yaml_max_deg": round(yaml_max, 3),
            "measured_min_deg": None if measured_min is None else round(measured_min, 3),
            "measured_max_deg": None if measured_max is None else round(measured_max, 3),
            "delta_min_deg": None if delta_min is None else round(delta_min, 3),
            "delta_max_deg": None if delta_max is None else round(delta_max, 3),
            "pass": passed,
        })
    return rows


def evaluate_grippers(
    limits: PhysicalLimits, installed: str, measured: dict[tuple[str, str], float | None],
    tolerance_m: float,
) -> list[dict[str, Any]]:
    """One row per gripper type; only the installed one carries measurements."""
    rows: list[dict[str, Any]] = []
    for gripper in GRIPPER_TYPES:
        low, high = limits.gripper_limits_m[gripper]
        if gripper != installed:
            rows.append({
                "gripper": gripper, "yaml_min_m": low, "yaml_max_m": high,
                "measured_closed_m": None, "measured_open_m": None,
                "delta_closed_m": None, "delta_open_m": None,
                "pass": None, "note": "not installed, unmeasured",
            })
            continue
        closed = measured.get(("Gripper", "closed"))
        opened = measured.get(("Gripper", "open"))
        delta_closed = None if closed is None else closed - low
        delta_open = None if opened is None else opened - high
        passed = (
            closed is not None and opened is not None
            and abs(closed - low) <= tolerance_m and abs(opened - high) <= tolerance_m
        )
        rows.append({
            "gripper": gripper, "yaml_min_m": low, "yaml_max_m": high,
            "measured_closed_m": None if closed is None else round(closed, 5),
            "measured_open_m": None if opened is None else round(opened, 5),
            "delta_closed_m": None if delta_closed is None else round(delta_closed, 5),
            "delta_open_m": None if delta_open is None else round(delta_open, 5),
            "pass": passed, "note": "installed",
        })
    return rows


def _cell(value: Any, width: int) -> str:
    """Format one table cell; None reads as a dash."""
    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "pass" if value else "FAIL"
    elif isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return text.rjust(width)


def format_table(joints: list[dict[str, Any]], grippers: list[dict[str, Any]]) -> str:
    """The results table in the runbook's column order."""
    header = ("joint", "yaml min", "yaml max", "measured min", "measured max", "delta min",
              "delta max", "pass/fail")
    widths = (8, 10, 10, 13, 13, 10, 10, 10)
    lines = ["".join(h.rjust(w) for h, w in zip(header, widths, strict=True))]
    for row in joints:
        cells = (row["joint"], row["yaml_min_deg"], row["yaml_max_deg"],
                 row["measured_min_deg"], row["measured_max_deg"], row["delta_min_deg"],
                 row["delta_max_deg"], row["pass"])
        lines.append("".join(_cell(c, w) for c, w in zip(cells, widths, strict=True)))
    lines.append("(joint angles in degrees)")
    lines.append("")
    gripper_header = ("gripper", "yaml min", "yaml max", "meas closed", "meas open",
                      "delta cl", "delta op", "pass/fail")
    lines.append("".join(h.rjust(w) for h, w in zip(gripper_header, widths, strict=True)))
    for row in grippers:
        verdict: Any = row["pass"] if row["pass"] is not None else row["note"]
        cells = (row["gripper"], row["yaml_min_m"], row["yaml_max_m"], row["measured_closed_m"],
                 row["measured_open_m"], row["delta_closed_m"], row["delta_open_m"], verdict)
        lines.append("".join(_cell(c, w) for c, w in zip(cells, widths, strict=True)))
    lines.append("(gripper positions in metres)")
    return "\n".join(lines)


def _keep_key(replacement: str) -> Callable[[re.Match[str]], str]:
    """A substitution that keeps the key and the spacing around the colon."""
    def substitute(match: re.Match[str]) -> str:
        """Replace only the quoted value."""
        return match.group(1) + replacement
    return substitute


def stamp_limits_file(path: Path, operator: str, date: str) -> None:
    """Rewrite only the two verification fields, keeping the file's layout byte for byte."""
    text = path.read_text(encoding="utf-8")
    for field, value in (("verified_by", operator), ("verified_on", date)):
        pattern = re.compile(r'("' + re.escape(field) + r'"\s*:\s*)"[^"]*"')
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"{field} must appear exactly once in {path}")
        text = pattern.sub(_keep_key(json.dumps(value)), text, count=1)
    path.write_text(text, encoding="utf-8")


def _write_report(output_dir: Path, report: dict[str, Any], stamp: str, dry_run: bool) -> Path:
    """Write the JSON report; dry runs carry ``dryrun`` in the name so they are never evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"verification_{'dryrun_' if dry_run else ''}{stamp}.json"
    path = output_dir / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    *,
    limits_path: Path,
    output_dir: Path,
    operator: str,
    gripper: str,
    topic: str,
    tolerance_deg: float,
    gripper_tolerance_m: float,
    dry_run: bool,
    source: FollowerSource | None = None,
    read_key: Callable[[], str] = default_read_key,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Walk the operator through every stop, write the report and return it.

    ``source`` defaults to the live subscriber, or to a fake that reaches every bound when
    ``dry_run`` is set. Dry runs never offer to stamp the limits file and refuse to write a
    report anywhere but under ``output_dir``.
    """
    if gripper not in GRIPPER_TYPES:
        raise ValueError("gripper must be 50mm or 100mm")
    if not operator.strip():
        raise ValueError("an operator name is required; it is what the report attributes")
    limits = load_limits(limits_path)
    if dry_run and source is None:
        source = FakeFollowerSource(dry_run_states(limits, gripper))
    elif source is None:
        say(f"Subscribing read-only to {topic}; no publisher is created.")
        source = LiveFollowerSource(topic)
    started = now()
    measured: dict[tuple[str, str], float | None] = {}
    aborted = False
    try:
        say("Limits verification" + (" (DRY RUN, fake follower)" if dry_run else ""))
        say(f"Limits file: {limits_path}")
        say(f"Currently recorded: verified_by={limits.verified_by!r} "
            f"verified_on={limits.verified_on!r}")
        say(KEY_HELP)
        for stop in stops_for(limits, gripper):
            say("")
            say(f"Jog to the {stop.label}. Configured bound: {stop.configured:.3f} {stop.unit}.")
            while True:
                state = source.read()
                say(f"  current reading: {reading_for(stop, state):.3f} {stop.unit}")
                key = read_key()
                if key == "m":
                    # Record the reading the operator just saw, not a later one.
                    value = reading_for(stop, state)
                    measured[(stop.name, stop.side)] = value
                    say(f"  marked {stop.name} {stop.side} = {value:.3f} {stop.unit}")
                    _advance(source)
                    break
                if key == "s":
                    measured[(stop.name, stop.side)] = None
                    say(f"  skipped {stop.name} {stop.side}; this side will fail")
                    _advance(source)
                    break
                if key == "a":
                    aborted = True
                    break
                if key != "r":
                    say("  unknown key")
            if aborted:
                break
    finally:
        source.close()
    if aborted:
        say("Aborted by the operator; no report written and the limits file is unchanged.")
        return {"aborted": True}
    joints = evaluate_joints(limits, measured, tolerance_deg)
    grippers = evaluate_grippers(limits, gripper, measured, gripper_tolerance_m)
    all_joints_pass = all(row["pass"] for row in joints)
    installed = next(row for row in grippers if row["gripper"] == gripper)
    all_pass = all_joints_pass and bool(installed["pass"])
    finished = now()
    stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "kind": "synria_limits_verification",
        "dry_run": dry_run,
        "evidence_class": ("dry run, not evidence" if dry_run
                           else "operator measurement on hardware"),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "operator": operator,
        "topic": None if dry_run else topic,
        "git_sha": git_sha(),
        "limits_path": str(limits_path),
        "limits_sha256": hashlib.sha256(limits_path.read_bytes()).hexdigest(),
        "verified_fields_before": {"verified_by": limits.verified_by,
                                   "verified_on": limits.verified_on},
        "installed_gripper": gripper,
        "tolerance_deg": tolerance_deg,
        "gripper_tolerance_m": gripper_tolerance_m,
        "pass_rule": PASS_RULE,
        "joints": joints,
        "grippers": grippers,
        "all_joints_pass": all_joints_pass,
        "all_pass": all_pass,
        "verified_fields_written": False,
    }
    say("")
    say(format_table(joints, grippers))
    say("")
    if dry_run:
        say("Dry run: the limits file is never stamped from a fake follower.")
    elif not all_pass:
        failing = [row["joint"] for row in joints if not row["pass"]]
        if not installed["pass"]:
            failing.append(f"gripper {gripper}")
        say("Not all stops passed (" + ", ".join(failing) + "); verified_by and verified_on "
            "are left unchanged. Fix the configured bounds or the measurement and rerun.")
    else:
        date = finished.strftime("%Y-%m-%d")
        say(f"All six joints and the {gripper} gripper passed.")
        answer = ask(
            f"Type your name exactly ({operator}) to write verified_by={operator!r} and "
            f"verified_on={date!r} into {limits_path}, or press Enter to leave it: "
        ).strip()
        if answer == operator:
            stamp_limits_file(limits_path, operator, date)
            report["verified_fields_written"] = True
            report["limits_sha256_after"] = hashlib.sha256(limits_path.read_bytes()).hexdigest()
            say(f"Wrote verified_by and verified_on to {limits_path}. Commit it with the report.")
        else:
            say("Limits file left unchanged.")
    path = _write_report(output_dir, report, stamp, dry_run)
    report["report_path"] = str(path)
    say(f"Report written: {path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the defaults point at the repository's limits and reports."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--operator", required=True, help="Name recorded in the report.")
    parser.add_argument("--gripper", choices=GRIPPER_TYPES, default="50mm",
                        help="The installed gripper; the other type is listed as unmeasured.")
    parser.add_argument("--topic", default="/joint_states",
                        help="Follower joint-state topic (read-only subscription).")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS,
                        help="Limits file to compare against and, on full pass, to stamp.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where the JSON report is written.")
    parser.add_argument("--tolerance-deg", type=float, default=1.0,
                        help="How far short of a configured bound a stop may be and still pass.")
    parser.add_argument("--gripper-tolerance-m", type=float, default=0.002,
                        help="Allowed difference between gripper readings and 0 or the stroke.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a fake follower that reaches every bound; never stamps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 when every stop passed, 1 otherwise."""
    args = build_parser().parse_args(argv)
    read_key: Callable[[], str] = default_read_key
    if args.dry_run and not sys.stdin.isatty():
        # An unattended dry run (CI) marks every stop so the whole flow is exercised.
        read_key = lambda: "m"  # noqa: E731
    report = run(
        limits_path=args.limits, output_dir=args.output_dir, operator=args.operator,
        gripper=args.gripper, topic=args.topic, tolerance_deg=args.tolerance_deg,
        gripper_tolerance_m=args.gripper_tolerance_m, dry_run=args.dry_run,
        read_key=read_key,
    )
    if report.get("aborted"):
        return 1
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

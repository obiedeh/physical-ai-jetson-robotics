"""Prove the Synria safety gates refuse unsafe input while the follower feed stands witness.

Four probes run through the real gate code, never through a publisher:

1. joint limit: a position beyond the URDF bound is refused by ``ArmSafetyGate``; a
   follower state beyond the operator limits is refused by the policy client; a policy
   target beyond the operator limits is clamped to the bound, never past it.
2. velocity limit: a joint speed above the speed-fraction bound is refused by
   ``ArmSafetyGate``.
3. stale observation: a follower sample older than ``max_observation_age_s`` makes the
   policy client hold instead of acting; a sample from the future does the same.
4. per-cycle step cap: a policy target far from the current pose is clamped to
   speed times command period on every joint; without opt-in no sink receives anything,
   and opt-in itself is refused while the policy safety config is unverified.

Before and after the probes the script reads the follower state and counts publishers on
the guarded command topics. The report records both and whether the follower moved.

The script refuses to run without ``--i-am-the-operator`` and a typed ``arm-clear``
confirmation. ``--dry-run`` replaces the follower subscriber with a fake and a fake clock;
its report is named ``gate_proof_dryrun_*`` and is never evidence.

Usage::

    python scripts/prove_safety_gates.py --i-am-the-operator --operator "Your Name"
    python scripts/prove_safety_gates.py --i-am-the-operator --operator "Your Name" --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from arm_control.safety import ArmSafetyGate, SafetyViolation  # noqa: E402
from scripts.verify_limits import LiveFollowerSource  # noqa: E402
from synria_lerobot.embodiment import SynriaEmbodiment, SynriaObservation  # noqa: E402
from synria_lerobot.physical_contract import (  # noqa: E402
    DEFAULT_COMMAND_TOPICS,
    DRIVER_JOINT_NAMES,
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.policy_client import (  # noqa: E402
    FakePolicy,
    GuardedCommandPath,
    PolicyClient,
    PolicySafetyConfig,
)
from synria_lerobot.quality_gates import PhysicalLimits, load_limits  # noqa: E402
from synria_lerobot.task_registry import DISPOSABLE_SMOKE, load_task_source  # noqa: E402

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
DEFAULT_POLICY = REPOSITORY / "config" / "synria_policy.json"
DEFAULT_TASKS = REPOSITORY / "config" / "synria_tasks.json"
DEFAULT_OUTPUT_DIR = REPOSITORY / "reports" / "safety"
CONFIRMATION = "arm-clear"
COVERAGE_NOTE = (
    "The per-cycle write cap is the policy client's step clamp: at most max joint speed "
    "times the command period per joint per cycle. The ROS adapter's refusal of a second "
    "offer within one command period needs an authorized motion session and is covered by "
    "unit tests only."
)


class FakeClock:
    """A clock that only moves when the fake sleep is called."""

    def __init__(self, start: float = 1000.0) -> None:
        """Start at ``start`` seconds."""
        self.now = start

    def __call__(self) -> float:
        """Current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time."""
        self.now += seconds


class FakeFollower:
    """A follower at rest inside the limits whose samples are stamped by the clock."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Stamp every sample with ``clock`` so freshness follows the fake time."""
        self.clock = clock
        self.closed = False

    def read(self) -> PhysicalState:
        """A fresh in-range state."""
        now = self.clock()
        return PhysicalState((0.1, 0.2, 0.3, 0.0, 0.1, 0.0), 0.01, now, now,
                             ros_arrival_stamp_s=now)

    def close(self) -> None:
        """Record closure."""
        self.closed = True


class RecordingSink:
    """A command sink that only counts; it is never reached without opt-in."""

    def __init__(self) -> None:
        """Start with nothing offered."""
        self.offers: list[tuple[float, ...]] = []
        self.holds = 0

    def offer(self, action: tuple[float, ...]) -> None:
        """Record an offered action."""
        self.offers.append(action)

    def hold(self) -> None:
        """Record a hold."""
        self.holds += 1


@dataclass
class Probe:
    """One gate exercise and what happened."""

    name: str
    gate: str
    utc: str
    input: dict[str, Any]
    expected: str
    observed: dict[str, Any]
    refused: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary for the JSON report."""
        return {
            "name": self.name, "gate": self.gate, "utc": self.utc, "input": self.input,
            "expected": self.expected, "observed": self.observed, "refused": self.refused,
            "passed": self.passed,
        }


def git_sha() -> str:
    """The repository commit the proof ran on, or an explicit unknown marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def command_publisher_counts(topics: tuple[str, ...]) -> dict[str, int] | None:
    """Publisher counts on the guarded topics from a private, short-lived node; None without ROS."""
    try:
        import rclpy  # type: ignore[import-not-found]
        from rclpy.context import Context  # type: ignore[import-not-found]
    except ImportError:
        return None
    context = Context()
    rclpy.init(context=context)
    try:
        node = rclpy.create_node("synria_gate_proof_witness", context=context)
        try:
            return {topic: int(node.count_publishers(topic)) for topic in topics}
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown(context=context)


def state_facts(state: PhysicalState) -> dict[str, Any]:
    """Joint angles in degrees, gripper in metres and the sample stamp."""
    return {
        "joint_positions_deg": [round(math.degrees(v), 3) for v in state.joint_positions_rad],
        "gripper_m": round(state.gripper_m, 5),
        "monotonic_timestamp_s": state.monotonic_timestamp_s,
    }


def max_change_deg(before: PhysicalState, after: PhysicalState) -> float:
    """The largest joint movement between two samples, in degrees."""
    return max(
        abs(math.degrees(b - a))
        for a, b in zip(before.joint_positions_rad, after.joint_positions_rad, strict=True)
    )


def observation(state: PhysicalState, stamp: float, task_text: str) -> SynriaObservation:
    """An observation around ``state`` with placeholder images stamped at ``stamp``."""
    image = ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), stamp)
    return SynriaObservation(state, image, image, task_text)


def build_client(
    limits: PhysicalLimits, config: PolicySafetyConfig, transport: Any, clock: Callable[[], float],
    *, gripper: str, tasks_path: Path,
) -> PolicyClient:
    """A policy client on the real limits and policy config with a disposable contract.

    The contract uses the disposable smoke purpose so no operator task timing is invented;
    the client's limit, staleness and clamp rules do not depend on the purpose.
    """
    task = load_task_source(tasks_path)["die_into_cup"]
    contract = PhysicalDatasetContract(
        gripper, ActionSource.NEXT_STATE, False, task_id=task.task_id, task_definition=task,
        recording_purpose=DISPOSABLE_SMOKE,
    )
    return PolicyClient(SynriaEmbodiment(contract), transport, limits, config, clock=clock)


def run(
    *,
    operator: str,
    dry_run: bool,
    topic: str,
    gripper: str,
    command_period_s: float,
    response_timeout_s: float,
    max_follower_change_deg: float,
    limits_path: Path = DEFAULT_LIMITS,
    policy_path: Path = DEFAULT_POLICY,
    tasks_path: Path = DEFAULT_TASKS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ask: Callable[[str], str] | None = None,
    say: Callable[[str], None] = print,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Confirm the arm is clear, run the four probes, write the report and return it."""
    ask = ask or input
    if not operator.strip():
        raise ValueError("an operator name is required; it is what the report attributes")
    answer = ask(
        f"Confirm the arm is clear, secured and the e-stop is within reach by typing "
        f"{CONFIRMATION}: "
    ).strip()
    if answer != CONFIRMATION:
        say("Not confirmed; nothing was run and no report was written.")
        return {"refused": True}
    confirmed_utc = now().isoformat(timespec="seconds")
    limits = load_limits(limits_path)
    config = PolicySafetyConfig.load(
        policy_path, command_period_s=command_period_s, response_timeout_s=response_timeout_s,
    )
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    source: Any
    if dry_run:
        fake_clock = FakeClock()
        clock, sleep = fake_clock, fake_clock.sleep
        source = FakeFollower(fake_clock)
        say("DRY RUN: fake follower and fake clock; no ROS, no hardware.")
    else:
        clock, sleep = time.monotonic, time.sleep
        say(f"Subscribing read-only to {topic}; no publisher is created.")
        source = LiveFollowerSource(topic)
    probes: list[Probe] = []
    task_text = "gate proof; no motion"
    bounds = (*limits.joint_limits_rad, limits.gripper_limits_m[gripper])
    try:
        publishers_before = None if dry_run else command_publisher_counts(DEFAULT_COMMAND_TOPICS)
        before = source.read()
        say(f"Follower before: {state_facts(before)['joint_positions_deg']} deg")

        # 1. Joint limit.
        gate = ArmSafetyGate()
        urdf_results = gate.check_joint_positions({"joint_2": 2.0})
        urdf_refused = any(r.violation is SafetyViolation.joint_limit for r in urdf_results)
        beyond = list(before.joint_positions_rad)
        beyond[0] = limits.joint_limits_rad[0][1] + 0.1
        outside = replace(before, joint_positions_rad=tuple(beyond))
        client = build_client(limits, config, FakePolicy((0.0,) * 7), clock, gripper=gripper,
                              tasks_path=tasks_path)
        stamp = clock()
        outside_decision = client.step(observation(
            replace(outside, monotonic_timestamp_s=stamp), stamp, task_text))
        far = build_client(limits, config, FakePolicy((100.0,) * 7), clock, gripper=gripper,
                           tasks_path=tasks_path)
        fresh = source.read()
        stamp = clock()
        far_decision = far.step(observation(
            replace(fresh, monotonic_timestamp_s=stamp), stamp, task_text))
        clamped_inside = far_decision.action is not None and all(
            lo <= value <= hi for value, (lo, hi) in zip(far_decision.action, bounds, strict=True)
        )
        probes.append(Probe(
            "joint_limit", "ArmSafetyGate.check_joint_positions; PolicyClient.step",
            now().isoformat(timespec="seconds"),
            {"urdf_command": {"joint_2": 2.0},
             "follower_state_joint1_deg": round(math.degrees(beyond[0]), 3),
             "policy_target_rad": 100.0},
            "URDF command refused; out-of-limit state holds; far target clamped to the bound",
            {"urdf_refused": urdf_refused,
             "urdf_message": urdf_results[0].message,
             "state_hold": outside_decision.hold, "state_reason": outside_decision.reason,
             "clamped_action_rad": far_decision.action,
             "operator_bounds_rad": [list(b) for b in bounds],
             "clamped_inside_bounds": clamped_inside},
            urdf_refused and outside_decision.hold,
            urdf_refused and outside_decision.hold
            and outside_decision.reason == "current state outside limits" and clamped_inside,
        ))

        # 2. Velocity limit.
        velocity_results = gate.check_joint_velocities({"joint_1": 10.0})
        velocity_refused = any(
            r.violation is SafetyViolation.velocity_limit for r in velocity_results
        )
        probes.append(Probe(
            "velocity_limit", "ArmSafetyGate.check_joint_velocities",
            now().isoformat(timespec="seconds"),
            {"velocity_rad_s": {"joint_1": 10.0}, "speed_fraction": gate.config.speed_fraction},
            "velocity refused",
            {"refused": velocity_refused, "message": velocity_results[0].message},
            velocity_refused, velocity_refused,
        ))

        # 3. Stale observation.
        sample = source.read()
        taken = clock()
        sleep(config.max_observation_age_s + 0.3)
        stale_decision = client.step(observation(
            replace(sample, monotonic_timestamp_s=taken), taken, task_text))
        age = clock() - taken
        future_stamp = clock() + 5.0
        future_decision = client.step(observation(
            replace(sample, monotonic_timestamp_s=future_stamp), future_stamp, task_text))
        stale_ok = stale_decision.hold and stale_decision.reason == "stale or future observation"
        future_ok = (future_decision.hold
                     and future_decision.reason == "stale or future observation")
        probes.append(Probe(
            "stale_observation", "PolicyClient.step",
            now().isoformat(timespec="seconds"),
            {"observation_age_s": round(age, 3),
             "max_observation_age_s": config.max_observation_age_s,
             "future_offset_s": 5.0},
            "stale sample holds; future sample holds",
            {"stale_hold": stale_decision.hold, "stale_reason": stale_decision.reason,
             "future_hold": future_decision.hold, "future_reason": future_decision.reason},
            stale_ok and future_ok, stale_ok and future_ok,
        ))

        # 4. Per-cycle step cap.
        current = (*fresh.joint_positions_rad, fresh.gripper_m)
        steps = None if far_decision.action is None else [
            abs(a - c) for a, c in zip(far_decision.action, current, strict=True)
        ]
        within_cap = steps is not None and all(
            step <= delta + 1e-12 for step, delta in zip(steps, config.max_delta, strict=True)
        )
        sink = RecordingSink()
        no_opt_in = GuardedCommandPath(far, lambda: sink, enable_motion=False)
        stamp = clock()
        no_opt_in.step(observation(replace(fresh, monotonic_timestamp_s=stamp), stamp, task_text))
        opt_in_error: str | None = None
        try:
            GuardedCommandPath(far, lambda: sink, enable_motion=True)
        except ValueError as error:
            opt_in_error = str(error)
        probes.append(Probe(
            "per_cycle_step_cap", "PolicyClient.step clamp; GuardedCommandPath opt-in",
            now().isoformat(timespec="seconds"),
            {"policy_target_rad": 100.0, "command_period_s": config.command_period_s,
             "max_joint_speed_rad_s": list(config.max_joint_speed_rad_s),
             "max_delta_per_cycle": list(config.max_delta)},
            "every joint moves at most speed times period; no sink without opt-in; "
            "opt-in refused (disposable contract; policy config unverified)",
            {"step_per_joint": steps, "within_cap": within_cap,
             "sink_offers": len(sink.offers), "sink_holds": sink.holds,
             "opt_in_refusal": opt_in_error,
             "policy_config_verified": config.verified,
             "limits_verified": limits.verified},
            within_cap and not sink.offers and opt_in_error is not None,
            within_cap and not sink.offers and opt_in_error is not None,
        ))

        after = source.read()
        publishers_after = None if dry_run else command_publisher_counts(DEFAULT_COMMAND_TOPICS)
    finally:
        source.close()
    change = max_change_deg(before, after)
    unchanged = change <= max_follower_change_deg
    all_passed = all(p.passed for p in probes) and unchanged
    finished = now()
    report: dict[str, Any] = {
        "kind": "synria_safety_gate_proof",
        "dry_run": dry_run,
        "evidence_class": ("dry run, not evidence" if dry_run
                           else "operator measurement on hardware"),
        "started_utc": confirmed_utc,
        "finished_utc": finished.isoformat(timespec="seconds"),
        "operator": operator,
        "arm_clear_confirmed_utc": confirmed_utc,
        "git_sha": git_sha(),
        "follower_topic": None if dry_run else topic,
        "limits_path": str(limits_path),
        "limits_sha256": hashlib.sha256(limits_path.read_bytes()).hexdigest(),
        "policy_config_path": str(policy_path),
        "policy_config_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "command_period_s": command_period_s,
        "response_timeout_s": response_timeout_s,
        "joint_names": list(DRIVER_JOINT_NAMES),
        "follower_before": state_facts(before),
        "follower_after": state_facts(after),
        "max_follower_change_deg": round(change, 4),
        "follower_change_allowance_deg": max_follower_change_deg,
        "follower_unchanged": unchanged,
        "command_publishers_before": publishers_before,
        "command_publishers_after": publishers_after,
        "probes": [p.as_dict() for p in probes],
        "all_probes_refused": all(p.refused for p in probes),
        "all_passed": all_passed,
        "coverage_note": COVERAGE_NOTE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp_text = finished.strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"gate_proof_{'dryrun_' if dry_run else ''}{stamp_text}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    for probe in probes:
        say(f"{'PASS' if probe.passed else 'FAIL'}  {probe.name}: {probe.expected}")
    say(f"{'PASS' if unchanged else 'FAIL'}  follower unchanged: max change {change:.4f} deg")
    say(f"Report written: {path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-am-the-operator", action="store_true",
                        help="Required. States that the person running this owns the bench.")
    parser.add_argument("--operator", default="", help="Name recorded in the report.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fake follower and clock; the report is never evidence.")
    parser.add_argument("--topic", default="/joint_states",
                        help="Follower joint-state topic (read-only subscription).")
    parser.add_argument("--gripper", choices=("50mm", "100mm"), default="50mm")
    parser.add_argument("--command-period-s", type=float, default=0.4,
                        help="Command period used to derive the per-cycle step cap.")
    parser.add_argument("--response-timeout-s", type=float, default=0.1,
                        help="Policy response timeout for the fake policy.")
    parser.add_argument("--max-follower-change-deg", type=float, default=0.5,
                        help="Largest joint movement between the before and after reads.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; 2 without the operator flag, 1 on refusal or failure, 0 on full pass."""
    args = build_parser().parse_args(argv)
    if not args.i_am_the_operator or not args.operator.strip():
        print("Refusing: pass --i-am-the-operator and --operator \"Your Name\".", file=sys.stderr)
        return 2
    report = run(
        operator=args.operator, dry_run=args.dry_run, topic=args.topic, gripper=args.gripper,
        command_period_s=args.command_period_s, response_timeout_s=args.response_timeout_s,
        max_follower_change_deg=args.max_follower_change_deg, output_dir=args.output_dir,
    )
    if report.get("refused"):
        return 1
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

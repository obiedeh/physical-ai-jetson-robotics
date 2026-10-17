"""Report bounded, read-only connection snapshots without certifying physical safety.

The console reuses the recorder builder in a disposable context. No episode is
started or saved, and no connection check substitutes for actual session preflight.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synria_lerobot.quality_gates import (
    GateConfig,
    PhysicalLimits,
    gate_config_for,
    source_timestamp_skew,
    within_recording_limits,
)
from synria_lerobot.recorder import PhysicalEpisodeRecorder

ARM_CHECK_STATEMENT = ("I have physically checked the arm: the area around it is clear and the "
                       "leader arm drives the sim arm.")
ARM_CHECK_MISSING = "Tick the box to confirm you have checked the arm"
SIMULATION_CONFIRMATIONS = {"arm_checked": ARM_CHECK_STATEMENT}
ISAAC_RECORD_HELP = ("Isaac record mode is not running. Press Start Isaac record mode above and "
                     "wait for Running; the checks then run by themselves.")
# The recorded sim views match the physical rig: the wrist camera and the overhead camera
# (the recorder's "front" slot). The sim's side-on front view is published for the operator's
# page only and is never recorded.
SIM_TOPICS = {"state": "/isaac_joint_states", "wrist": "/isaac/wrist_camera/image_raw",
              "front": "/isaac/overhead_camera/image_raw", "action": "/joint_states",
              "scene": "/isaac/front_camera/image_raw"}
CONFIRMATIONS = {
    "secured": "Arm secured, workspace clear and emergency stop within reach",
    "read_only": "Existing state source has writes disabled and startup leaves torque unchanged",
    "leader_link": "Leader/follower sync cable and existing teleoperation verified by operator",
    "camera_roles": "Two distinct camera identities mapped to wrist and fixed front views",
    "procedure": "Runbook safety preflight accepted; power and gripper identity checked",
}


def runtime_dependencies(*, demo: bool) -> dict[str, bool]:
    """Inspect installation availability without importing a device runtime or executing tools."""
    modules = ["lerobot", "cv2"] + ([] if demo else ["rclpy", "sensor_msgs"])
    return {**{name: importlib.util.find_spec(name) is not None for name in modules},
            "ffmpeg": shutil.which("ffmpeg") is not None}


class ReadinessReport:
    """Keep auditable check results with conservative readiness wording and automatic expiry."""

    def __init__(self, clock: Callable[[], float], *, demo: bool, simulation: bool = False) -> None:
        """Start unverified; demo and simulation results never become a physical readiness claim."""
        self.clock = clock
        self.demo = demo
        self.simulation = simulation
        self.running = False
        self.completed_s: float | None = None
        self.checks: dict[str, dict[str, Any]] = {}
        self.checked_utc: str | None = None
        self.form_settings: dict[str, Any] = {}
        # Latest camera sample per role from the last check, for a thumbnail on the setup screen.
        self.thumbnails: dict[str, Any] = {}

    def begin(self) -> None:
        """Invalidate previous evidence before a fresh explicit diagnostic request."""
        self.running = True
        self.completed_s = None
        self.thumbnails = {}
        self.checked_utc = datetime.now(timezone.utc).isoformat()
        self.checks = {name: {"passed": False, "status": "not_checked", "message": "Not run"}
                       for name in (
            "operator_safety", "form_configuration", "recording_metadata",
            "qualifying_configuration",
            "limits_verification", "disk_space", "camera_mapping", "runtime_dependencies",
            "source_startup", "follower_sample", "action_sample", "wrist_sample", "front_sample",
            "source_alignment", "distinct_views", "cleanup",
        ) + (("isaac_record_mode",) if self.simulation else ())}

    def not_applicable(self, name: str, message: str) -> None:
        """Record openly that a real-arm fact does not apply to this source, never as a pass."""
        self.checks = {**self.checks, name: {
            "passed": True, "status": "not_applicable", "message": message,
        }}

    def check(self, name: str, passed: bool, message: str, **facts: Any) -> None:
        """Publish one immutable result so polling never iterates a changing dictionary."""
        self.checks = {**self.checks, name: {
            "passed": passed, "status": "ready" if passed else "not_ready",
            "message": message, **facts,
        }}

    def launching(self, name: str, message: str) -> None:
        """Identify only the path currently being checked, never an unstarted device service."""
        self.checks = {**self.checks, name: {
            "passed": False, "status": "launching", "message": message,
        }}

    def observe(self, name: str, result: dict[str, Any]) -> None:
        """Retain the shared builder's actual checks without guessing unrun successes."""
        if name == "state_source" and result.get("passed") is True:
            self.check("state_subscription", True,
                       "Subscription created; receipt of follower data is checked separately",
                       status="configured", details=copy.deepcopy(result))
            return
        self.check(name, result.get("passed") is True,
                   result.get("message", "Shared recorder check passed"),
                   details=copy.deepcopy(result))

    def finish(self) -> None:
        """Timestamp the completed snapshot only after owned resources have closed."""
        for name, result in self.checks.items():
            if result.get("status") == "launching":
                self.check(name, False, "Check interrupted; see diagnostic error")
        self.running = False
        self.completed_s = self.clock()

    def snapshot(self) -> dict[str, Any]:
        """Derive a readiness meter that never promotes partial or expired checks to ready."""
        checks = copy.deepcopy(self.checks)
        passed = sum(row["passed"] is True for row in checks.values())
        connected = all(checks.get(name, {}).get("passed") is True for name in (
            "follower_sample", "wrist_sample", "front_sample", "action_sample",
        ))
        expired = self.completed_s is not None and self.clock() - self.completed_s >= 60
        complete = bool(checks) and passed == len(checks) and self.completed_s is not None
        if self.running:
            status, label, percent = "checking", "Checking connections", min(90, passed * 4)
        elif expired:
            status, label, percent = "expired", "Check expired · run again", 0
        elif self.completed_s is None:
            status, label, percent = "unchecked", "Not checked", 0
        elif complete and connected:
            status, label, percent = "ready", "Ready for session preflight", 100
        elif connected:
            status, label, percent = "attention", "Connected · setup required", 70
        else:
            status, label, percent = "blocked", "Connection checks incomplete", 25 if passed else 0
        if self.demo:
            label = "SYNTHETIC DEMO · " + label
        elif self.simulation:
            label = "SIMULATION · " + label
        return {
            "status": status, "label": label, "percent": percent, "checks": checks,
            "passed_count": passed, "check_count": len(checks), "checked_utc": self.checked_utc,
            "expires_after_s": 60, "demo": self.demo, "simulation": self.simulation,
            "thumbnails": sorted(self.thumbnails),
            "form_settings": copy.deepcopy(self.form_settings),
            "scope": "Read-only snapshot, not motion authorization. Sources close after checks. "
                     "Actual session contract, lock/recovery and every episode guard run again. "
                     "Leader sync, torque, emergency stop and view framing need operator checks. "
                     "Frozen-video and episode quality gates still run on recorded episodes.",
        }


def require_distinct_cameras(wrist: str, front: str) -> None:
    """Reject the same path, aliases or two video interfaces advertising one USB identity."""
    paths = (Path(wrist), Path(front))
    identities = tuple(path.name.rsplit("-video-index", 1)[0] for path in paths)
    if paths[0].resolve() == paths[1].resolve() or identities[0] == identities[1]:
        raise ValueError("Choose two distinct cameras; video-index0 and video-index1 "
                         "of one USB identity are not a verified wrist/front pair.")


JOINT_WORDS = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6", "Gripper")


def limit_violations(values: tuple[float, ...], limits: PhysicalLimits,
                     gripper_type: str) -> list[str]:
    """Name every value outside the recording bounds, in operator units.

    Joints are reported in degrees against the operator bound plus the recording
    tolerance; the gripper in millimetres against its exact bound. An unknown gripper
    type or a malformed vector is reported as such rather than silently passing.
    """
    if len(values) != 7:
        return [f"expected 6 joints and a gripper, got {len(values)} values"]
    if gripper_type not in limits.gripper_limits_m:
        return [f"gripper type {gripper_type!r} has no limits"]
    out: list[str] = []
    bounds = limits.recording_bounds(gripper_type)
    for name, value, (lower, upper) in zip(JOINT_WORDS, values, bounds, strict=True):
        if math.isfinite(value) and lower <= value <= upper:
            continue
        if name == "Gripper":
            out.append(f"{name} at {value * 1000:.1f} mm is outside {lower * 1000:.0f} to "
                       f"{upper * 1000:.0f} mm")
        else:
            side, bound = (("below", lower) if math.isfinite(value) and value < lower
                           else ("above", upper))
            out.append(f"{name} at {math.degrees(value):.1f} deg is {side} its "
                       f"{math.degrees(bound):.0f} deg recording bound")
    return out


def limits_pose_message(values: tuple[float, ...], limits: PhysicalLimits,
                        gripper_type: str) -> str:
    """The refusal shown when a sample is outside the operator joint or gripper limits."""
    named = "; ".join(limit_violations(values, limits, gripper_type)) or "unknown value"
    return (f"Arm is beyond a recording limit: {named}. The arm is probably at its folded rest "
            "position. Move the arm to an upright position, away from the rest pose, and press "
            "Check again. (Outside the operator joint or gripper limits; recording tolerance "
            "applied to joints only.)")


def inspect_samples(
    recorder: PhysicalEpisodeRecorder, report: ReadinessReport, limits: PhysicalLimits, *,
    sleep: Callable[[float], None] = time.sleep, settle_s: float = 0.5, window_s: float = 1.0,
) -> None:
    """Inspect latest source values only; never start, capture, label or save an episode.

    First samples prove each source delivers data. Alignment is then judged only on
    steady-state reads taken after ``settle_s``, back to back the way an episode frame
    is captured, for ``window_s`` at the configured rate; the worst skew in that window
    is compared with the unchanged episode gate. A cold-start comparison is never
    reported as measured skew.
    """
    import numpy as np

    config = recorder.config
    gates = gate_config_for(config.contract.recording_purpose,
                            config.min_episode_s, config.max_episode_s)
    timestamps: dict[str, float] = {}
    images: list[Any] = []

    def fresh(timestamp: float) -> None:
        """Apply the episode gate's source-age threshold without relaxing it for startup."""
        age = recorder.clock() - timestamp
        if not math.isfinite(age) or not 0 <= age <= gates.max_source_age_s:
            raise ValueError(f"Source sample age {age:.3f} s exceeds freshness requirements")

    def header(stamp: float | None, arrival: float | None) -> None:
        """Compare ROS timestamps only in their own clock domain."""
        if stamp is None or arrival is None or not (
            -gates.max_header_future_s <= arrival - stamp <= gates.max_header_delay_s
        ):
            raise ValueError("ROS header and arrival stamps fail the source-freshness gate")

    def vector(values: tuple[float, ...]) -> None:
        """Check six joints and the gripper with the shared recording-limit helper.

        The recording tolerance applies to the joints only; commands never use it. The
        refusal names each value that is out of bounds and tells the operator to move
        the arm, since the usual cause is an arm parked at its folded rest pose.
        """
        if len(values) != 7 or not within_recording_limits(
            values, limits, config.contract.gripper_type,
        ):
            raise ValueError(limits_pose_message(values, limits, config.contract.gripper_type))

    report.launching("follower_sample", "Checking latest follower state")
    try:
        follower = config.contract.prepare_state(recorder.state_source.read())
        fresh(follower.monotonic_timestamp_s)
        header(follower.ros_header_stamp_s, follower.ros_arrival_stamp_s)
        vector((*follower.joint_positions_rad, follower.gripper_m))
        timestamps["state_monotonic_s"] = follower.monotonic_timestamp_s
        report.check("follower_sample", True, "Fresh follower state received; not a motion test")
        report.launching("action_sample", "Checking configured action source")
        try:
            action = recorder.action_source.read(follower)
            if action is not None:
                fresh(action.monotonic_timestamp_s)
                header(action.ros_header_stamp_s, action.ros_arrival_stamp_s)
                vector(action.values)
                timestamps["action_monotonic_s"] = action.monotonic_timestamp_s
            report.check("action_sample", True, "USB leader sample received" if action else
                         "Actions derive from follower state; leader cable is operator-confirmed")
        except Exception as error:
            report.check("action_sample", False, str(error))
    except Exception as error:
        report.check("follower_sample", False, str(error))
        report.check("action_sample", False, "Follower state unavailable; action check not run",
                     status="not_checked")
    for name, source in (("wrist", recorder.wrist_source), ("front", recorder.front_source)):
        report.launching(f"{name}_sample", f"Checking latest {name} camera frame")
        try:
            frame = source.read()
            fresh(frame.monotonic_timestamp_s)
            pixels = np.asarray(frame.data)
            if pixels.dtype != np.uint8 or pixels.shape != (
                config.image_height, config.image_width, 3,
            ):
                raise ValueError("Camera must provide uint8 RGB at the configured stored size")
            report.thumbnails[name] = pixels.copy()
            if float(pixels.mean()) <= gates.black_mean_threshold:
                raise ValueError("Camera sample is black; check cover, lighting and exposure")
            images.append(pixels)
            timestamps[f"{name}_monotonic_s"] = frame.monotonic_timestamp_s
            report.check(f"{name}_sample", True, "Fresh RGB frame received; framing needs review",
                         native_resolution=frame.native_resolution,
                         stored_resolution=[config.image_width, config.image_height])
        except Exception as error:
            report.check(f"{name}_sample", False, str(error))
    required_checks = ["follower_sample", "wrist_sample", "front_sample"]
    if config.contract.action_source.value == "leader":
        required_checks.append("action_sample")
    missing = [
        name for name in required_checks
        if report.checks.get(name, {}).get("passed") is not True
    ]
    if missing:
        report.check(
            "source_alignment", False,
            "Not evaluated: first obtain fresh samples from all required sources; "
            "then their timestamps are compared with the existing skew gate.",
            status="not_checked", waiting_for=missing,
            max_timestamp_skew_s=gates.max_timestamp_skew_s,
        )
    else:
        aligned_sources = ["state_monotonic_s", "wrist_monotonic_s", "front_monotonic_s"]
        if config.contract.action_source.value == "leader":
            aligned_sources.append("action_monotonic_s")
        if not all(name in timestamps for name in aligned_sources):
            report.check(
                "source_alignment", False,
                "Not evaluated: a required source timestamp is unavailable; rerun the "
                "read-only check after all source samples are fresh.",
                status="not_checked", expected_sources=len(required_checks),
                received_timestamps=len(timestamps),
                max_timestamp_skew_s=gates.max_timestamp_skew_s,
            )
        else:
            _measure_steady_alignment(
                recorder, report, gates, first_sample_timestamps=timestamps,
                sleep=sleep, settle_s=settle_s, window_s=window_s,
            )
    report.check("distinct_views", len(images) == 2 and not np.array_equal(*images),
                 "Wrist/front samples must differ; operator must still verify their roles")


def _measure_steady_alignment(
    recorder: PhysicalEpisodeRecorder, report: ReadinessReport, gates: GateConfig, *,
    first_sample_timestamps: dict[str, float], sleep: Callable[[float], None],
    settle_s: float, window_s: float,
) -> None:
    """Judge alignment on settled back-to-back reads, as the per-frame episode gate will.

    The first samples may include a camera's first frame that arrived long after the
    follower was read; that span is retained for audit but never reported as skew.
    """
    config = recorder.config
    action_source = config.contract.action_source.value
    sample_count = max(1, round(window_s * config.fps))
    period_s = 1.0 / config.fps
    worst_skew: float | None = None
    worst_stamps: dict[str, float] = {}
    worst_age = 0.0
    try:
        sleep(settle_s)
        for index in range(sample_count):
            if index:
                sleep(period_s)
            follower = config.contract.prepare_state(recorder.state_source.read())
            action = recorder.action_source.read(follower)
            wrist = recorder.wrist_source.read()
            front = recorder.front_source.read()
            sampled = recorder.clock()
            stamps = {
                "state_monotonic_s": follower.monotonic_timestamp_s,
                "wrist_monotonic_s": wrist.monotonic_timestamp_s,
                "front_monotonic_s": front.monotonic_timestamp_s,
            }
            if action is not None:
                stamps["action_monotonic_s"] = action.monotonic_timestamp_s
            for name, stamp in stamps.items():
                if name == "action_monotonic_s" and action_source != "leader":
                    continue
                age = sampled - stamp
                if not math.isfinite(age) or not 0 <= age <= gates.max_source_age_s:
                    raise ValueError(
                        f"{name.split('_', 1)[0]} sample age {age:.3f} s exceeds freshness "
                        "requirements during the alignment window"
                    )
                worst_age = max(worst_age, age)
            _, skew = source_timestamp_skew(stamps, action_source, gates)
            if skew is None:
                raise ValueError("a required source timestamp is unavailable")
            if worst_skew is None or skew > worst_skew:
                worst_skew, worst_stamps = skew, stamps
    except Exception as error:
        report.check(
            "source_alignment", False,
            f"Steady-state alignment could not be measured: {error}",
            max_timestamp_skew_s=gates.max_timestamp_skew_s,
            first_sample_timestamps_s=first_sample_timestamps,
            settle_s=settle_s, window_s=window_s,
        )
        return
    assert worst_skew is not None
    synchronized, _ = source_timestamp_skew(worst_stamps, action_source, gates)
    report.check(
        "source_alignment", synchronized,
        f"Steady-state skew over {sample_count} back-to-back reads in {window_s:.1f} s: "
        f"worst {worst_skew:.3f} s {'is within' if synchronized else 'exceeds'} the existing "
        f"{gates.max_timestamp_skew_s:.3f} s gate.",
        measured_skew_s=worst_skew, max_timestamp_skew_s=gates.max_timestamp_skew_s,
        source_timestamps_s=worst_stamps, worst_source_age_s=worst_age,
        sample_count=sample_count, settle_s=settle_s, window_s=window_s,
        first_sample_timestamps_s=first_sample_timestamps,
    )

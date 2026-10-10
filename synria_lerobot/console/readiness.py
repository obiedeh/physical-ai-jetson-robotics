"""Report bounded, read-only connection snapshots without certifying physical safety.

The console reuses the recorder builder in a disposable context. No episode is
started or saved, and no connection check substitutes for actual session preflight.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synria_lerobot.quality_gates import GateConfig, PhysicalLimits
from synria_lerobot.recorder import PhysicalEpisodeRecorder

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

    def __init__(self, clock: Callable[[], float], *, demo: bool) -> None:
        """Start unverified; demo results can never become a physical readiness claim."""
        self.clock = clock
        self.demo = demo
        self.running = False
        self.completed_s: float | None = None
        self.checks: dict[str, dict[str, Any]] = {}
        self.checked_utc: str | None = None
        self.form_settings: dict[str, Any] = {}

    def begin(self) -> None:
        """Invalidate previous evidence before a fresh explicit diagnostic request."""
        self.running = True
        self.completed_s = None
        self.checked_utc = datetime.now(timezone.utc).isoformat()
        self.checks = {name: {"passed": False, "status": "not_checked", "message": "Not run"}
                       for name in (
            "operator_safety", "form_configuration", "recording_metadata",
            "qualifying_configuration",
            "limits_verification", "disk_space", "camera_mapping", "runtime_dependencies",
            "source_startup", "follower_sample", "action_sample", "wrist_sample", "front_sample",
            "source_alignment", "distinct_views", "cleanup",
        )}

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
        return {
            "status": status, "label": label, "percent": percent, "checks": checks,
            "passed_count": passed, "check_count": len(checks), "checked_utc": self.checked_utc,
            "expires_after_s": 60, "demo": self.demo,
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


def inspect_samples(
    recorder: PhysicalEpisodeRecorder, report: ReadinessReport, limits: PhysicalLimits,
) -> None:
    """Inspect latest source values only; never start, capture, label or save an episode."""
    import numpy as np

    config = recorder.config
    gates = GateConfig(config.min_episode_s, config.max_episode_s)
    times: list[float] = []
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
        """Check six joint positions and the configured gripper against the existing limits."""
        bounds = (*limits.joint_limits_rad, limits.gripper_limits_m[config.contract.gripper_type])
        if len(values) != 7 or any(
            not math.isfinite(value) or not low <= value <= high
            for value, (low, high) in zip(values, bounds, strict=True)
        ):
            raise ValueError("State/action is outside the configured joint or gripper limits")

    report.launching("follower_sample", "Checking latest follower state")
    try:
        follower = config.contract.prepare_state(recorder.state_source.read())
        fresh(follower.monotonic_timestamp_s)
        header(follower.ros_header_stamp_s, follower.ros_arrival_stamp_s)
        vector((*follower.joint_positions_rad, follower.gripper_m))
        times.append(follower.monotonic_timestamp_s)
        report.check("follower_sample", True, "Fresh follower state received; not a motion test")
        report.launching("action_sample", "Checking configured action source")
        try:
            action = recorder.action_source.read(follower)
            if action is not None:
                fresh(action.monotonic_timestamp_s)
                header(action.ros_header_stamp_s, action.ros_arrival_stamp_s)
                vector(action.values)
                times.append(action.monotonic_timestamp_s)
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
            if float(pixels.mean()) <= gates.black_mean_threshold:
                raise ValueError("Camera sample is black; check cover, lighting and exposure")
            images.append(pixels)
            times.append(frame.monotonic_timestamp_s)
            report.check(f"{name}_sample", True, "Fresh RGB frame received; framing needs review",
                         native_resolution=frame.native_resolution,
                         stored_resolution=[config.image_width, config.image_height])
        except Exception as error:
            report.check(f"{name}_sample", False, str(error))
    expected = 4 if config.contract.action_source.value == "leader" else 3
    synchronized = len(times) == expected and max(times) - min(times) <= gates.max_timestamp_skew_s
    report.check("source_alignment", synchronized,
                 "Latest source timestamps must satisfy the existing skew gate")
    report.check("distinct_views", len(images) == 2 and not np.array_equal(*images),
                 "Wrist/front samples must differ; operator must still verify their roles")

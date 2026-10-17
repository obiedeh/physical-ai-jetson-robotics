"""Exercise connection readiness with injected sources, clocks and device-name lists only."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_console_service import ConsoleFixture, values
from test_physical_recorder import FakeClock, FakeStateSource, FakeWriter
from test_task_registry import synthetic_registry

from synria_lerobot.console import app
from synria_lerobot.console.readiness import (
    CONFIRMATIONS,
    ReadinessReport,
    inspect_samples,
    require_distinct_cameras,
    runtime_dependencies,
)
from synria_lerobot.physical_contract import ImageFrame, PhysicalState
from synria_lerobot.quality_gates import GateConfig, load_limits, source_timestamp_skew
from synria_lerobot.recorder import (
    NextStateActionSource,
    PhysicalEpisodeRecorder,
    RecorderState,
    recording_config_from_args,
)


def request_values() -> dict[str, Any]:
    """Declare fake physical identities without consulting this machine's device directory."""
    return {"settings": {**values(), "state_source": "standalone_driver",
                         "wrist_camera": "fake-wrist", "front_camera": "fake-front",
                         "follower_serial": "fake", "power_state_start": "operator-confirmed",
                         "image_width": 32, "image_height": 24},
            "confirmations": dict.fromkeys(CONFIRMATIONS, True)}


@pytest.mark.parametrize("available", [True, False])
def test_software_paths_report_availability_without_opening_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available: bool,
) -> None:
    """Discovery distinguishes each missing library and never constructs source runtimes."""
    monkeypatch.setattr(app, "runtime_dependencies", lambda **kwargs: {
        "lerobot": True, "cv2": True, "rclpy": available, "sensor_msgs": available,
        "ffmpeg": True,
    })
    with ConsoleFixture(tmp_path, builder=lambda *a, **k: pytest.fail("opened sources")) as service:
        result = service.read(["software"], {})
        assert result["status"] == ("ready" if available else "not_ready")
        assert result["dependencies"]["cv2"]
        assert result["dependencies"]["rclpy"] is available
        if not available:
            assert "rclpy, sensor_msgs" in result["message"]
        assert service.catalog.events() == []


def test_path_status_does_not_leave_interrupted_checks_launching() -> None:
    """An interrupted path becomes not ready while untouched paths remain not checked."""
    report = ReadinessReport(FakeClock(), demo=False)
    report.begin()
    report.launching("follower_sample", "Checking latest state")
    assert report.snapshot()["checks"]["follower_sample"]["status"] == "launching"
    report.finish()
    assert report.snapshot()["checks"]["follower_sample"]["status"] == "not_ready"
    assert report.snapshot()["checks"]["wrist_sample"]["status"] == "not_checked"


@pytest.fixture
def fake_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make inventory and verification evidence explicitly fake for every physical-mode test."""
    monkeypatch.setattr(app, "list_cameras", lambda: {
        "cameras": ["fake-wrist", "fake-front"], "message": "fake inventory",
    })
    monkeypatch.setattr(app, "runtime_dependencies", lambda **kwargs: {"fake": True})
    candidate = load_limits(app.REPOSITORY / "config/synria_limits.yaml")
    monkeypatch.setattr("synria_lerobot.quality_gates.load_limits", lambda path: replace(
        candidate, verified_by="fake operator", verified_on="2026-10-10",
    ))


def fake_recorder(args: argparse.Namespace, clock: FakeClock) -> PhysicalEpisodeRecorder:
    """Build one idle recorder with synthetic arrays and an in-memory writer."""
    config = recording_config_from_args(args)
    state = PhysicalState((0.0,) * 6, 0.01, clock(), clock(), ros_arrival_stamp_s=clock())
    sources = [SimpleNamespace(
        read=lambda color=color: ImageFrame(
            np.full((24, 32, 3), color, dtype=np.uint8), clock(), (640, 480),
        ), close=lambda: None,
    ) for color in (50, 100)]
    # Readiness re-reads every source for its steady-state window, so serve the same
    # construction-time state repeatedly instead of a single sample.
    return PhysicalEpisodeRecorder(
        config=config, state_source=FakeStateSource([state] * 256),
        action_source=NextStateActionSource(), wrist_source=sources[0], front_source=sources[1],
        writer=FakeWriter(args.dataset_path), clock=clock, command_publisher_guard=lambda _: None,
    )


def test_shared_probe_never_starts_or_saves_and_expires(
    tmp_path: Path, fake_connections: None,
) -> None:
    """A healthy fake probe closes its context, audits both boundaries and saves no episode."""
    clock = FakeClock()
    owned: list[PhysicalEpisodeRecorder] = []

    @contextmanager
    def builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[PhysicalEpisodeRecorder]:
        """Expose the real recorder lifecycle over inert samples and track its cleanup."""
        assert args.smoke is True
        session = fake_recorder(args, clock)
        owned.append(session)
        kwargs["preflight"]("command_publishers", {"passed": True})
        try:
            yield session
            assert session.state is RecorderState.IDLE
            assert session.pending_frame_count == 0
            assert session.writer.episodes == []  # type: ignore[attr-defined]
        finally:
            session.close()

    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    with ConsoleFixture(tmp_path / "workspace", clock=clock, registry=registry,
                        builder=builder) as service:
        assert service.read(["readiness"], {})["status"] == "unchecked"
        assert owned == []
        result = service.mutate(["readiness"], request_values())
        assert result["status"] == "ready" and result["percent"] == 100
        assert result["checks"]["cleanup"]["passed"]
        assert result["thumbnails"] == ["front", "wrist"]
        for camera in ("wrist", "front"):
            assert service.media(["readiness", camera], {})[:2] == b"\xff\xd8"
        with pytest.raises(KeyError):
            service.media(["readiness", "other"], {})
        assert owned[0].state_source.closed  # type: ignore[attr-defined]
        assert service.controller is None
        assert service.catalog.sessions() == []
        assert not list(service.workspace.glob(".readiness-*"))
        assert [row["kind"] for row in service.catalog.events()] == [
            "readiness_started", "readiness_completed",
        ]
        clock.now = 60
        assert service.read(["readiness"], {})["status"] == "expired"
        assert service.read(["readiness"], {})["percent"] == 0


@pytest.mark.parametrize("fault", [
    "confirmation", "same_camera", "missing_camera", "disk", "dependencies",
])
def test_setup_refusals_never_open_sources(
    tmp_path: Path, fake_connections: None, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Missing consent, storage or distinct camera mapping fail before the shared builder."""
    request = request_values()
    if fault == "confirmation":
        request["confirmations"]["secured"] = False
    elif fault == "same_camera":
        request["settings"]["front_camera"] = "fake-wrist"
    elif fault == "missing_camera":
        request["settings"]["front_camera"] = "absent"
    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    with ConsoleFixture(tmp_path / "workspace", registry=registry,
                        builder=lambda *a, **k: pytest.fail("opened sources")) as service:
        if fault == "disk":
            monkeypatch.setattr(app.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        if fault == "dependencies":
            monkeypatch.setattr(app, "runtime_dependencies", lambda **kwargs: {"missing": False})
        result = service.mutate(["readiness"], request)
        assert result["status"] == "blocked"
        assert result["checks"]["diagnostic"]["passed"] is False
        assert not service.readiness.running


@pytest.mark.parametrize("missing", [
    ("follower_serial",), ("power_state_start",), ("follower_serial", "power_state_start"),
])
def test_missing_recording_metadata_does_not_block_connection_diagnostics(
    tmp_path: Path, fake_connections: None, missing: tuple[str, ...],
) -> None:
    """Unknown recording facts stay explicit while fake connectivity is tested without capture."""
    request = request_values()
    for name in missing:
        request["settings"][name] = ""
    owned: list[PhysicalEpisodeRecorder] = []

    @contextmanager
    def builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[PhysicalEpisodeRecorder]:
        """Exercise real recorder sample checks with no source devices or saved episode."""
        session = fake_recorder(args, FakeClock())
        owned.append(session)
        try:
            yield session
            assert session.pending_frame_count == 0
            assert session.writer.episodes == []  # type: ignore[attr-defined]
        finally:
            session.close()

    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    with ConsoleFixture(tmp_path / "workspace", registry=registry, builder=builder) as service:
        result = service.mutate(["readiness"], request)
        assert len(owned) == 1
        assert result["status"] == "attention"
        assert result["checks"]["form_configuration"]["passed"]
        assert result["checks"]["follower_sample"]["passed"]
        assert result["checks"]["wrist_sample"]["passed"]
        assert result["checks"]["front_sample"]["passed"]
        assert result["checks"]["cleanup"]["passed"]
        metadata = result["checks"]["recording_metadata"]
        assert not metadata["passed"]
        for name in missing:
            assert name in metadata["message"]
            assert result["form_settings"][name] == ""
        for route in (["sessions"], ["smoke"]):
            with pytest.raises(ValueError, match="Required fields are missing"):
                service.mutate(route, request["settings"])
        with pytest.raises(ValueError, match="unknown session settings"):
            service.create_session({**request["settings"], "diagnostic": True})
        assert service.catalog.sessions() == []
        assert service.controller is None
        assert len(owned) == 1


def test_incomplete_recording_metadata_keeps_safety_confirmation_required(
    tmp_path: Path, fake_connections: None,
) -> None:
    """Omitting power text cannot waive the actual operator power and safety confirmation."""
    request = request_values()
    request["settings"].update(follower_serial="", power_state_start="")
    request["confirmations"]["procedure"] = False
    with ConsoleFixture(tmp_path, builder=lambda *a, **k: pytest.fail("opened sources")) as service:
        result = service.mutate(["readiness"], request)
        assert result["status"] == "blocked"
        assert "physical safety checklist" in result["checks"]["diagnostic"]["message"]


def test_invalid_source_form_is_reported_as_configuration_not_connection_failure(
    tmp_path: Path, fake_connections: None,
) -> None:
    """Name the blocking input and leave unattempted source checks explicitly not run."""
    request = request_values()
    request["settings"]["fps"] = ""
    with ConsoleFixture(tmp_path, builder=lambda *a, **k: pytest.fail("opened sources")) as service:
        result = service.mutate(["readiness"], request)
        assert not result["checks"]["form_configuration"]["passed"]
        assert "fps" in result["checks"]["form_configuration"]["message"]
        assert result["checks"]["follower_sample"]["message"] == "Not run"


def test_two_video_interfaces_are_not_two_cameras(tmp_path: Path) -> None:
    """A second index and a filesystem alias must not masquerade as another view."""
    with pytest.raises(ValueError, match="two distinct"):
        require_distinct_cameras("usb-camera-video-index0", "usb-camera-video-index1")
    with pytest.raises(ValueError, match="two distinct"):
        require_distinct_cameras(str(tmp_path / "one"), str(tmp_path / "sub/../one"))
    require_distinct_cameras("usb-wrist-video-index0", "usb-front-video-index0")


@pytest.mark.parametrize("fault", ["publisher exists", "no state sample", "camera busy"])
def test_builder_refusals_are_not_promoted_to_readiness(
    tmp_path: Path, fake_connections: None, fault: str,
) -> None:
    """Surface the shared guard's actual message and leave the console idle on refusal."""
    @contextmanager
    def refused(args: argparse.Namespace, **kwargs: Any) -> Iterator[Any]:
        """Stand in for a bounded source startup failure, without touching a runtime."""
        kwargs["preflight"]("source", {"passed": False, "message": fault})
        raise RuntimeError(fault)
        yield

    with ConsoleFixture(tmp_path / "workspace", builder=refused) as service:
        result = service.mutate(["readiness"], request_values())
        assert result["status"] == "blocked"
        assert result["checks"]["source"]["message"] == fault
        assert service.controller is None


@pytest.mark.parametrize("fault", ["stale", "header", "black", "duplicate", "shape", "limits"])
def test_latest_sample_faults_never_report_ready(
    tmp_path: Path, fake_connections: None, fault: str,
) -> None:
    """Independent live-sample defects remain failed checks despite successful construction."""
    clock = FakeClock()
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        if fault in {"stale", "header", "limits"}:
            session.state_source = FakeStateSource([PhysicalState(
                ((4.0,) * 6 if fault == "limits" else (0.0,) * 6), 0.01,
                -1.0 if fault == "stale" else 0.0,
                -1.0 if fault == "header" else 0.0, ros_arrival_stamp_s=0.0,
            )])
        else:
            color = 0 if fault == "black" else 50
            shape = (2, 2, 3) if fault == "shape" else (24, 32, 3)
            session.front_source = SimpleNamespace(
                read=lambda: ImageFrame(np.full(shape, color, dtype=np.uint8), 0.0),
                close=lambda: None,
            )
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"))
        report.finish()
        assert report.snapshot()["status"] != "ready"
        assert any(not row["passed"] for row in report.checks.values())
        assert session.pending_frame_count == 0
        session.close()


@pytest.mark.parametrize(
    ("mode", "expected_status", "message_fragment"),
    [
        ("aligned", "ready", "0.000 s is within the existing 0.050 s gate"),
        ("missing_front", "not_checked", "first obtain fresh samples"),
        ("skewed", "not_ready", "0.100 s exceeds the existing 0.050 s gate"),
    ],
)
def test_source_alignment_distinguishes_pending_from_measured_skew(
    tmp_path: Path, fake_connections: None, mode: str,
    expected_status: str, message_fragment: str,
) -> None:
    """Readiness reports alignment only after all fresh required source timestamps exist."""
    clock = FakeClock()
    clock.now = 1.0
    with ConsoleFixture(tmp_path / mode) as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        if mode == "missing_front":
            session.front_source = SimpleNamespace(
                read=lambda: (_ for _ in ()).throw(RuntimeError("front camera unavailable")),
                close=lambda: None,
            )
        elif mode == "skewed":
            clock.now = 1.1
            session.front_source = SimpleNamespace(
                read=lambda: ImageFrame(
                    np.full((24, 32, 3), 100, dtype=np.uint8), 1.1, (640, 480),
                ),
                close=lambda: None,
            )
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"))
        report.finish()
        alignment = report.checks["source_alignment"]
        assert alignment["status"] == expected_status
        assert message_fragment in alignment["message"]
        if mode == "missing_front":
            assert alignment["waiting_for"] == ["front_sample"]
        elif mode == "skewed":
            assert alignment["measured_skew_s"] == pytest.approx(0.1)
            assert alignment["source_timestamps_s"] == {
                "state_monotonic_s": 1.0,
                "wrist_monotonic_s": 1.1,
                "front_monotonic_s": 1.1,
            }
        session.close()


def advancing_sleep(clock: FakeClock) -> Any:
    """Advance the fake clock instead of waiting, and count the calls made."""
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        """Move fake time forward by the requested wait."""
        calls.append(seconds)
        clock.now += seconds

    return sleep, calls


def live_state_source(clock: FakeClock) -> Any:
    """Serve a follower state stamped at read time, like a streaming subscriber cache."""
    return SimpleNamespace(
        read=lambda: PhysicalState((0.0,) * 6, 0.01, clock(), clock(), ros_arrival_stamp_s=clock()),
        close=lambda: None,
    )


def test_camera_warm_up_delay_is_not_reported_as_measured_skew(
    tmp_path: Path, fake_connections: None,
) -> None:
    """A first frame arriving long after the follower read must not fail steady alignment."""
    clock = FakeClock()
    clock.now = 100.0
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        session.state_source = live_state_source(clock)
        reads = {"front": 0}

        def cold_front() -> ImageFrame:
            """Block for a 0.6 s warm-up on the first read only, then stream fresh frames."""
            reads["front"] += 1
            if reads["front"] == 1:
                clock.now += 0.6
            return ImageFrame(np.full((24, 32, 3), 100, dtype=np.uint8), clock(), (640, 480))

        session.front_source = SimpleNamespace(read=cold_front, close=lambda: None)
        sleep, calls = advancing_sleep(clock)
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"),
                        sleep=sleep)
        report.finish()
        alignment = report.checks["source_alignment"]
        assert alignment["status"] == "ready"
        assert alignment["measured_skew_s"] == pytest.approx(0.0)
        assert "0.600" not in alignment["message"]
        first = alignment["first_sample_timestamps_s"]
        assert first["front_monotonic_s"] - first["state_monotonic_s"] == pytest.approx(0.6)
        assert alignment["sample_count"] == round(settings["fps"])
        assert calls[0] == pytest.approx(0.5)
        assert len(calls) == alignment["sample_count"]
        session.close()


def test_steady_lag_still_fails_alignment(tmp_path: Path, fake_connections: None) -> None:
    """A source that stays 0.1 s behind the others fails every read in the window."""
    clock = FakeClock()
    clock.now = 100.0
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        session.state_source = live_state_source(clock)
        session.front_source = SimpleNamespace(
            read=lambda: ImageFrame(
                np.full((24, 32, 3), 100, dtype=np.uint8), clock() - 0.1, (640, 480),
            ),
            close=lambda: None,
        )
        sleep, _ = advancing_sleep(clock)
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"),
                        sleep=sleep)
        report.finish()
        alignment = report.checks["source_alignment"]
        assert alignment["status"] == "not_ready"
        assert alignment["measured_skew_s"] == pytest.approx(0.1)
        assert "worst 0.100 s exceeds the existing 0.050 s gate" in alignment["message"]
        assert report.snapshot()["status"] != "ready"
        session.close()


def test_missing_source_skips_alignment_window(tmp_path: Path, fake_connections: None) -> None:
    """Without a required first sample, alignment stays Not checked and no window is sampled."""
    clock = FakeClock()
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        session.wrist_source = SimpleNamespace(
            read=lambda: (_ for _ in ()).throw(RuntimeError("wrist camera unavailable")),
            close=lambda: None,
        )
        sleep, calls = advancing_sleep(clock)
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"),
                        sleep=sleep)
        report.finish()
        alignment = report.checks["source_alignment"]
        assert alignment["status"] == "not_checked"
        assert alignment["waiting_for"] == ["wrist_sample"]
        assert "measured_skew_s" not in alignment
        assert calls == []
        session.close()


def test_source_fault_during_alignment_window_is_not_ready(
    tmp_path: Path, fake_connections: None,
) -> None:
    """A source that goes stale after its first sample cannot leave alignment green."""
    clock = FakeClock()
    clock.now = 100.0
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        sleep, _ = advancing_sleep(clock)
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"),
                        sleep=sleep)
        report.finish()
        alignment = report.checks["source_alignment"]
        assert alignment["status"] == "not_ready"
        assert "could not be measured" in alignment["message"]
        assert "state sample age" in alignment["message"]
        assert "measured_skew_s" not in alignment
        session.close()


@pytest.mark.parametrize(("joint3_deg", "expected"), [(-34.37, True), (-35.5, False)])
def test_follower_sample_uses_recording_tolerance_on_joints(
    tmp_path: Path, fake_connections: None, joint3_deg: float, expected: bool,
) -> None:
    """The readiness probe judges the follower sample with the episode gate's tolerance."""
    import math

    clock = FakeClock()
    with ConsoleFixture(tmp_path / "workspace") as service:
        settings = service._settings(request_values()["settings"], smoke=True)
        args = service._args(settings, "local/fake", tmp_path / "dataset", True)
        session = fake_recorder(args, clock)
        state = PhysicalState(
            (0.0, 0.0, math.radians(joint3_deg), 0.0, 0.0, 0.0), 0.01, clock(), clock(),
            ros_arrival_stamp_s=clock(),
        )
        session.state_source = FakeStateSource([state] * 256)
        report = ReadinessReport(clock, demo=False)
        report.begin()
        inspect_samples(session, report, load_limits(app.REPOSITORY / "config/synria_limits.yaml"),
                        sleep=lambda seconds: None)
        report.finish()
        assert report.checks["follower_sample"]["passed"] is expected
        if not expected:
            message = report.checks["follower_sample"]["message"]
            assert message.startswith("Arm is beyond a recording limit: Joint3 at -35.5 deg is "
                                      "below its -35 deg recording bound.")
            assert "Move the arm to an upright position" in message
            assert "operator joint or gripper limits" in message
        session.close()


def test_limit_violations_name_each_joint_and_the_gripper() -> None:
    """The folded rest pose names Joint2 and the bound; a negative gripper is named in mm."""
    import math

    from synria_lerobot.console.readiness import limit_violations, limits_pose_message

    limits = load_limits(app.REPOSITORY / "config/synria_limits.yaml")
    parked = (0.0, math.radians(101.3), math.radians(-33.7), 0.0, 0.0, 0.0, 0.0001)
    assert limit_violations(parked, limits, "50mm") == [
        "Joint2 at 101.3 deg is above its 97 deg recording bound"]
    assert limit_violations((0.0,) * 6 + (-0.0005,), limits, "50mm") == [
        "Gripper at -0.5 mm is outside 0 to 25 mm"]
    assert limit_violations((0.0,) * 7, limits, "50mm") == []
    assert limit_violations((0.0,) * 6, limits, "50mm") == [
        "expected 6 joints and a gripper, got 6 values"]
    assert limits_pose_message(parked, limits, "50mm").startswith(
        "Arm is beyond a recording limit: Joint2 at 101.3 deg is above its 97 deg recording "
        "bound. The arm is probably at its folded rest position.")


def test_readiness_uses_episode_skew_gate_and_action_source_semantics() -> None:
    """The readiness snapshot shares the exact episode threshold and contemporaneous inputs."""
    gate = GateConfig(1.0, 2.0)
    samples = {
        "state_monotonic_s": 10.0,
        "wrist_monotonic_s": 9.95,
        "front_monotonic_s": 10.0,
        # Next-state action is deliberately shifted, and must not enter source alignment.
        "action_monotonic_s": 11.0,
    }
    passed, skew = source_timestamp_skew(samples, "next_state", gate)
    assert passed
    assert skew == pytest.approx(0.05)
    # Leader actions are contemporaneous inputs and must satisfy the same bound.
    samples["action_monotonic_s"] = 9.94
    passed, skew = source_timestamp_skew(samples, "leader", gate)
    assert not passed
    assert skew == pytest.approx(0.06)


@pytest.mark.parametrize("blocker", ["window", "limits"])
def test_connected_sources_do_not_override_qualifying_requirements(
    tmp_path: Path, fake_connections: None, monkeypatch: pytest.MonkeyPatch, blocker: str,
) -> None:
    """Disposable connection diagnostics cannot make an untimed task qualify."""
    @contextmanager
    def builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[PhysicalEpisodeRecorder]:
        """Supply complete fake connections while retaining the real unset task snapshot."""
        session = fake_recorder(args, FakeClock())
        try:
            yield session
        finally:
            session.close()

    options: dict[str, Any] = {}
    if blocker == "limits":
        options["registry"] = synthetic_registry(tmp_path / "tasks.json", 1, 30)
        # The repository file carries the operator's signature; blank it for this blocker.
        monkeypatch.setattr("synria_lerobot.quality_gates.load_limits", lambda path: replace(
            load_limits(path), verified_by="", verified_on="",
        ))
    with ConsoleFixture(tmp_path / "workspace", builder=builder, **options) as service:
        result = service.mutate(["readiness"], request_values())
        assert result["status"] == "attention"
        check = "qualifying_configuration" if blocker == "window" else "limits_verification"
        assert not result["checks"][check]["passed"]
        assert blocker in result["checks"][check]["message"]
        assert json.dumps(result, allow_nan=False)


def test_running_session_blocks_new_diagnostic_without_touching_it(tmp_path: Path) -> None:
    """Never open a second camera/source pair while the recorder owns the current session."""
    with ConsoleFixture(tmp_path / "workspace") as service:
        service.controller = object()  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="Close the current sources"):
                service.mutate(["readiness"], request_values())
            assert service.readiness.snapshot()["status"] == "unchecked"
        finally:
            service.controller = None


def test_readiness_browser_invalidation_and_explicit_request() -> None:
    """Form edits and withdrawn confirmations must not leave a green browser meter."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    subprocess.run(
        [executable, str(Path(__file__).with_name("console_readiness_ui.cjs")),
         str(app.REPOSITORY / "synria_lerobot/console/static/readiness.js")],
        check=True, capture_output=True, text=True, timeout=15,
    )


def test_dependency_inventory_never_imports_ros_in_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demo dependency discovery must remain independent of ROS or device APIs."""
    seen: list[str] = []
    monkeypatch.setattr("importlib.util.find_spec", lambda name: seen.append(name) or object())
    monkeypatch.setattr("shutil.which", lambda name: "/fake/ffmpeg")
    assert all(runtime_dependencies(demo=True).values())
    assert seen == ["lerobot", "cv2"]


def test_cleanup_failure_cannot_leave_green_readiness(
    tmp_path: Path, fake_connections: None,
) -> None:
    """Successful live samples are insufficient if releasing owned resources failed."""
    @contextmanager
    def builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[PhysicalEpisodeRecorder]:
        """Simulate cleanup failure after closing all fake-owned source objects."""
        session = fake_recorder(args, FakeClock())
        try:
            yield session
        finally:
            session.close()
            raise RuntimeError("source cleanup failed")

    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    with ConsoleFixture(tmp_path / "workspace", registry=registry, builder=builder) as service:
        result = service.mutate(["readiness"], request_values())
        assert result["status"] == "attention"
        assert result["checks"]["diagnostic"]["message"] == "source cleanup failed"
        assert result["checks"]["cleanup"]["message"] == "Not run"


class SteppingClock:
    """A clock that moves a little on every read, so back-to-back reads straddle device ticks."""

    def __init__(self, start: float, step_s: float) -> None:
        """Start at an arbitrary phase and advance by ``step_s`` each time it is read."""
        self.now = start
        self.step_s = step_s

    def __call__(self) -> float:
        """Return the current time, then move on so the next read lands later."""
        value = self.now
        self.now += self.step_s
        return value

    def sleep(self, seconds: float) -> None:
        """Advance without a real wait."""
        self.now += seconds


@pytest.mark.parametrize("start", [0.0, 0.0301, 0.0499, 0.0666, 0.1234, 0.9999])
@pytest.mark.parametrize("step_s", [0.001, 0.004])
def test_synthetic_alignment_verdict_is_stable_across_phases(
    tmp_path: Path, start: float, step_s: float,
) -> None:
    """An on-entry check and a manual rerun must agree: reads straddling a device tick stay aligned.

    The synthetic cameras tick at 30 Hz and the arm feed at 50 Hz, independent of the
    15 Hz recording rate, so the worst straddle stays under the unchanged 0.050 s gate.
    """
    from synria_lerobot import recorder as recorder_module
    from synria_lerobot.task_registry import SYNTHETIC_DEMO

    args = argparse.Namespace(
        task_id="die_into_cup", task_registry=synthetic_registry(tmp_path / "tasks.json", 1, 30),
        dataset_path=tmp_path / "dataset", repo_id="local/readiness-test", gripper_type="50mm",
        action_source="next_state", state_has_velocity=False, smoke=False, fps=15,
        image_width=32, image_height=24, action_lookahead_steps=1,
        follower_topic="/joint_states", leader_topic="/leader/joint_states",
        state_startup_timeout_s=10.0, state_source=SYNTHETIC_DEMO,
        guard_command_topic=[], wrist_camera="synthetic:wrist", front_camera="synthetic:front",
    )
    clock = SteppingClock(start, step_s)
    limits = load_limits(app.REPOSITORY / "config/synria_limits.yaml")
    verdicts = []
    for _ in range(2):
        report = ReadinessReport(clock, demo=True)
        report.begin()
        with recorder_module.build_recording_session(
            args, synthetic_demo=True, clock=clock,
        ) as session:
            inspect_samples(session, report, limits, sleep=clock.sleep)
        alignment = report.checks["source_alignment"]
        assert alignment["passed"] is True, alignment["message"]
        assert alignment["measured_skew_s"] <= 1 / 30 + 4 * step_s
        verdicts.append(alignment["passed"])
    assert verdicts == [True, True], "the first run and the rerun must give the same verdict"

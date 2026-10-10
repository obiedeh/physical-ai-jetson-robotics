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
from synria_lerobot.quality_gates import load_limits
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
    return PhysicalEpisodeRecorder(
        config=config, state_source=FakeStateSource([state]),
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
        session.close()


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
        monkeypatch.setattr("synria_lerobot.quality_gates.load_limits", load_limits)
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

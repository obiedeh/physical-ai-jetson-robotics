"""Verify facade state and source construction using fake writers and synthetic-only inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_physical_recorder import FakeClock, FakeFrameSource, FakeStateSource, FakeWriter
from test_task_registry import synthetic_registry

from synria_lerobot import recorder
from synria_lerobot.console.app import (
    ConsoleService,
    WorkspaceLock,
    list_cameras,
    list_serial_connections,
    main,
)
from synria_lerobot.physical_contract import PhysicalState


def values() -> dict[str, Any]:
    """Provide a complete explicit synthetic session form without physical source claims."""
    return {"name": "Test collection", "task_id": "die_into_cup", "gripper_type": "50mm",
            "fps": 15, "operator": "test", "scene": "synthetic scene"}


def test_workspace_lock_refuses_second_instance_and_releases_owned_marker(tmp_path: Path) -> None:
    """An existing console marker cannot be overwritten or claimed by another launch."""
    with ConsoleFixture(tmp_path) as service:
        with pytest.raises(RuntimeError, match="another console"):
            ConsoleService(tmp_path)
        assert service.workspace == tmp_path.resolve()
    assert not (tmp_path / ".console.lock").exists()


@contextmanager
def ConsoleFixture(tmp_path: Path, **kwargs: Any) -> Iterator[ConsoleService]:
    """Own a temporary facade and always close it without constructing real sources."""
    service = ConsoleService(tmp_path, min_free_bytes=1, **kwargs)
    try:
        yield service
    finally:
        service.close(force=True, reason="fake test cleanup")


def test_form_cannot_supply_paths_windows_or_physical_demo_source(tmp_path: Path) -> None:
    """User form data cannot turn a demo into qualifying data or override registry timing."""
    with ConsoleFixture(tmp_path, demo=True) as service:
        for forbidden in ("dataset_path", "task_registry", "min_episode_s", "recording_purpose"):
            with pytest.raises(ValueError, match="server-owned"):
                service.create_session({**values(), forbidden: "override"})
        session = service.create_session(values())
        assert session["settings"]["state_source"] == "synthetic_demo"
        assert session["recording_purpose"] == "synthetic_demo"
        assert session["repo_id"].startswith("local/")
        assert not Path(session["dataset_path"]).exists()


def test_unset_physical_window_refuses_before_sources(tmp_path: Path) -> None:
    """An untimed task cannot create a qualifying physical session through the form."""
    with ConsoleFixture(tmp_path) as service:
        with pytest.raises(ValueError, match="window|timing"):
            service.create_session({**values(), "state_source": "standalone_driver",
                                    "wrist_camera": "fake-wrist", "front_camera": "fake-front",
                                    "follower_serial": "fake", "power_state_start": "fake"})
        assert service.catalog.sessions() == []


def test_preflight_refusal_preserves_not_run_checks_and_capture_audit(tmp_path: Path) -> None:
    """A builder refusal is visible, audited and cannot be dismissed into an active recorder."""
    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)

    @contextmanager
    def refused(args: argparse.Namespace, **kwargs: Any) -> Iterator[Any]:
        """Fail at the graph guard before a camera or writer could be opened."""
        kwargs["preflight"]("command_publishers", {"passed": False, "message": "publisher exists"})
        raise RuntimeError("publisher exists")
        yield

    with ConsoleFixture(tmp_path / "workspace", registry=registry, builder=refused) as service:
        session = service.create_session({**values(), "state_source": "standalone_driver",
                                          "wrist_camera": "fake-wrist",
                                          "front_camera": "fake-front",
                                          "follower_serial": "fake", "power_state_start": "fake"})
        with pytest.raises(RuntimeError, match="publisher exists"):
            service.open_session(session["id"])
        state = service.state()
        assert state["active_session"] is None
        assert state["preflight"]["checks"]["wrist_camera"] == {"status": "not run"}
        assert not state["preflight"]["checks"]["command_publishers"]["passed"]
        assert service.catalog.connection.execute("SELECT count(*) FROM capture_runs").fetchone()[0]
        assert any(event["kind"] == "preflight_refused"
                   for event in service.catalog.events(session["id"]))


def test_reload_rename_discard_and_pending_quit_are_server_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live fake session survives page reload and retains pending frames across quit refusal."""
    pytest.importorskip("cv2")
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", lambda config: FakeWriter(tmp_path))
    with ConsoleFixture(tmp_path / "workspace", demo=True) as service:
        session = service.create_session(values())
        identity = session["id"]
        service.open_session(identity)
        service.mutate(["session", identity, "rename"], {"name": "Renamed collection"})
        assert service.read(["state"], {})["active_session"]["name"] == "Renamed collection"
        service.mutate(["record", "start"], {})
        with pytest.raises(RuntimeError, match="pending"):
            service.mutate(["shutdown"], {})
        assert not service.quit_requested.is_set()
        assert service.state()["capture"]["state"] == "recording"
        service.mutate(["record", "discard"], {"reason": "synthetic capture interruption"})
        assert service.state()["active_session"]["counts"]["discarded"] == 1
        service.close_session()
        service.open_session(identity)
        assert service.state()["active_session"]["counts"]["discarded"] == 1


def test_registry_drift_refuses_resume_without_opening_sources(tmp_path: Path) -> None:
    """Stored task hashes remain binding even when a named session has no saved dataset yet."""
    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    with ConsoleFixture(tmp_path / "workspace", demo=True, registry=registry) as service:
        session = service.create_session(values())
        payload = json.loads(registry.read_text(encoding="utf-8"))
        payload["tasks"][0]["task_text"] += " changed"
        registry.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="task definition differs"):
            service.open_session(session["id"])


def test_camera_listing_only_lists_names_and_missing_directory(tmp_path: Path) -> None:
    """Stable camera inventory never opens files and gracefully handles other platforms."""
    missing = list_cameras(tmp_path / "missing")
    assert missing["cameras"] == [] and missing["message"]
    (tmp_path / "fake-camera").write_text("not a device", encoding="utf-8")
    assert list_cameras(tmp_path)["cameras"] == [str(tmp_path / "fake-camera")]


def test_serial_inventory_reads_names_without_opening_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection inventory is safe for disconnected hosts and never reads device contents."""
    candidate = tmp_path / "usb-fake-adapter"
    candidate.write_text("not a device", encoding="utf-8")
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("device opened"))
    assert list_serial_connections(tmp_path)["connections"] == [str(candidate)]
    assert "not manufacturer" in list_serial_connections(tmp_path)["message"]
    assert list_serial_connections(tmp_path / "missing")["connections"] == []


def test_usb_hint_is_separate_from_manufacturer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected connection is retained as a hint and never fills a missing arm serial."""
    candidate = "usb-fake-adapter"
    monkeypatch.setattr("synria_lerobot.console.app.list_serial_connections", lambda: {
        "connections": [candidate], "message": "unverified",
    })
    registry = synthetic_registry(tmp_path / "tasks.json", 1, 30)
    payload = {**values(), "state_source": "ros2_control", "wrist_camera": "fake-wrist",
               "front_camera": "fake-front", "power_state_start": "observed fake state",
               "follower_usb_id": candidate}
    with ConsoleFixture(tmp_path / "workspace", registry=registry) as service:
        with pytest.raises(ValueError, match="follower_serial.*does not replace"):
            service.create_session(payload)
        session = service.create_session({**payload, "follower_serial": "fake-arm-serial"})
        assert session["settings"]["follower_usb_id"] == candidate
        assert session["follower_serial"] == "fake-arm-serial"
        assert service._args(session["settings"], session["repo_id"],
                             Path(session["dataset_path"]), False).follower_topic == "/joint_states"
        with pytest.raises(ValueError, match="unavailable"):
            service._settings({**payload, "follower_serial": "fake-arm-serial",
                               "follower_usb_id": "unknown-port"})


def test_demo_serial_inventory_does_not_inspect_host_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic setup never discovers or preserves physical USB association hints."""
    monkeypatch.setattr("synria_lerobot.console.app.list_serial_connections",
                        lambda: pytest.fail("physical inventory in demo"))
    with ConsoleFixture(tmp_path, demo=True) as service:
        assert service.read(["serial-connections"], {})["connections"] == []
        result = service.create_session({**values(), "follower_usb_id": "not-a-real-device"})
        assert "follower_usb_id" not in result["settings"]


def test_setup_suggestions_restore_declarations_not_safety_facts(tmp_path: Path) -> None:
    """Previous user inputs remain distinguishable from current host facts and recipe candidates."""
    with ConsoleFixture(tmp_path) as service:
        service.catalog.record_event(None, "readiness_completed", {"demo": False, "form_settings": {
            "operator": "prior operator", "state_source": "ros2_control", "gripper_type": "50mm",
            "follower_serial": "old serial", "power_state_start": "old power state", "fps": 30,
        }})
        service.catalog.record_event(None, "readiness_completed", {"demo": True, "form_settings": {
            "operator": "synthetic operator",
        }})
        result = service.read(["setup-suggestions"], {})
        assert result["previous_operator_entries"] == {
            "operator": "prior operator", "state_source": "ros2_control", "gripper_type": "50mm",
        }
        assert result["recommended"]["fps"] == 15
        assert result["host"] and result["account"] and result["free_disk_bytes"] > 0
        assert "follower_serial" not in result["recommended"]
        assert "power_state_start" not in result["recommended"]
        assert service.controller is None


def test_changed_workspace_lock_is_not_deleted(tmp_path: Path) -> None:
    """Cleanup preserves a substituted lock as evidence rather than guessing ownership."""
    lock = WorkspaceLock(tmp_path)
    lock.path.write_text("not owned", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        lock.close()
    assert lock.path.exists()


def test_startup_disk_failure_closes_context_and_preserves_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk error after shared construction still finalizes all newly owned sources."""
    pytest.importorskip("cv2")
    writer = FakeWriter(tmp_path)
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", lambda config: writer)
    service = ConsoleService(tmp_path / "workspace", demo=True, min_free_bytes=1)
    session = service.create_session(values())

    def fail_disk(path: Path) -> Any:
        """Inject failure between source construction and controller ownership."""
        raise OSError("synthetic disk query failure")

    with monkeypatch.context() as patch:
        patch.setattr("synria_lerobot.console.app.shutil.disk_usage", fail_disk)
        with pytest.raises(OSError, match="disk query"):
            service.open_session(session["id"])
    assert service.controller is None and service._context is None
    assert service.catalog.get_session(session["id"])["status"] == "closed"
    service.close()


def test_source_cleanup_failure_still_ends_run_and_releases_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup attempts independent resources even when one recorder-owned close reports failure."""
    pytest.importorskip("cv2")
    monkeypatch.setattr(recorder, "LeRobotDatasetWriter", lambda config: FakeWriter(tmp_path))
    service = ConsoleService(tmp_path / "workspace", demo=True, min_free_bytes=1)
    session = service.create_session(values())
    service.open_session(session["id"])
    controller = service.controller
    assert controller is not None
    original = controller.shutdown

    def failed_shutdown(reason: str, *, force: bool = False) -> None:
        """Finish the real synthetic worker, then inject one reported close failure."""
        original(reason, force=force)
        raise OSError("synthetic close failure")

    monkeypatch.setattr(controller, "shutdown", failed_shutdown)
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        service.close()
    assert service.controller is None and service._run_id is None
    assert not (service.workspace / ".console.lock").exists()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_disposable_smoke_starts_after_shared_builder_and_stays_transient(
    tmp_path: Path, cleanup_failure: bool,
) -> None:
    """One smoke request starts capture while its review context stays temporary and uncounted."""
    pytest.importorskip("cv2")
    clock = FakeClock()

    @contextmanager
    def fake_builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[Any]:
        """Inject synthetic caches only after validating the actual disposable contract."""
        assert args.smoke and not kwargs["synthetic_demo"]
        config = replace(recorder.recording_config_from_args(
            args, dataset_path=tmp_path / "temporary-smoke" / "dataset",
        ), state_source_provenance=None)
        assert config.min_episode_s == config.max_episode_s == 20
        capture = recorder.PhysicalEpisodeRecorder(
            config=config, state_source=FakeStateSource([
                PhysicalState((0.0,) * 6, 0.01, 0.0, 0.0, ros_arrival_stamp_s=0.0)
            ] * 100),
            action_source=recorder.NextStateActionSource(),
            wrist_source=FakeFrameSource("wrist", clock),
            front_source=FakeFrameSource("front", clock),
            writer=FakeWriter(tmp_path), clock=clock,
        )
        try:
            yield capture
        finally:
            capture.close()
            if cleanup_failure:
                raise OSError("synthetic smoke cleanup failure")

    with ConsoleFixture(tmp_path / "workspace", builder=fake_builder, clock=clock) as service:
        state = service.mutate(["smoke"], {
            **values(), "state_source": "standalone_driver", "wrist_camera": "fake-wrist",
            "front_camera": "fake-front", "follower_serial": "fake", "power_state_start": "fake",
        })
        assert state["capture"]["state"] == "recording"
        assert state["active_session"]["id"] == "smoke"
        assert service.catalog.sessions() == []
        service.mutate(["record", "discard"], {"reason": "synthetic smoke test complete"})
        if cleanup_failure:
            with pytest.raises(RuntimeError, match="synthetic smoke cleanup failure"):
                service.close_session()
        else:
            service.close_session()
        assert service.state()["active_session"] is None
        events = service.catalog.events()
        assert sum(event["kind"] == "capture_run_started" for event in events) == 1
        ended = [event for event in events if event["kind"] == "capture_run_ended"]
        assert len(ended) == 1 and ended[0]["session"] is None
        detail = ended[0]["detail"]
        assert detail["recording_purpose"] == "disposable_smoke"
        assert detail["exit_reason"] == "operator closed"
        assert detail["disposal_succeeded"]
        assert bool(detail["cleanup_errors"]) is cleanup_failure


def test_failed_browser_launch_releases_workspace_and_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A desktop browser failure cannot leave a console instance lock behind."""
    def fail_browser(url: str) -> None:
        """Replace the desktop launch with an explicit fake exception before any source opens."""
        raise OSError("synthetic browser failure")

    monkeypatch.setattr("synria_lerobot.console.app.webbrowser.open", fail_browser)
    with pytest.raises(OSError, match="browser failure"):
        main(["--workspace", str(tmp_path / "workspace"), "--port", "0"])
    assert not (tmp_path / "workspace" / ".console.lock").exists()

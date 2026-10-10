"""Exercise every console capture transition with fake clocks and memory-only sources."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_console_recording_builder import settings
from test_physical_recorder import FakeClock, FakeWriter

from synria_lerobot.console.controller import RecordingController
from synria_lerobot.console.demo import SyntheticStateSource
from synria_lerobot.physical_contract import ImageFrame
from synria_lerobot.recorder import (
    NextStateActionSource,
    PhysicalEpisodeRecorder,
    RecorderState,
    recording_config_from_args,
)


def setup_controller(
    tmp_path: Path,
) -> tuple[RecordingController, FakeClock, FakeWriter, list[Any]]:
    """Build a real recorder around cached synthetic values and a memory-only writer."""
    config = recording_config_from_args(settings(tmp_path, demo=True), synthetic_demo=True)
    clock = FakeClock()
    writer = FakeWriter(tmp_path)
    events: list[Any] = []
    pixels = np.full((24, 32, 3), 100, dtype=np.uint8)

    def frame() -> ImageFrame:
        """Return an immutable-style latest sample without acquiring any device."""
        return ImageFrame(pixels, clock(), native_resolution=(32, 24), source_id="fake")

    source = SimpleNamespace(read=frame, close=lambda: None)
    recorder = PhysicalEpisodeRecorder(
        config=config,
        state_source=SyntheticStateSource(config, clock=clock),
        action_source=NextStateActionSource(),
        wrist_source=source,
        front_source=source,
        writer=writer,
        clock=clock,
    )
    controller = RecordingController(
        recorder,
        emit_event=lambda name, detail: events.append((name, detail)),
        on_saved=lambda episode: {"gate_passed": True, "failed_gates": []},
        min_free_bytes=100,
        disk_free=lambda: 1000,
    )
    return controller, clock, writer, events


@pytest.mark.parametrize("label", ["success", "failure"])
def test_complete_capture_transitions_and_reload_snapshot(tmp_path: Path, label: str) -> None:
    """Saved episodes advance exactly once and page reloads recover all server state."""
    controller, clock, writer, events = setup_controller(tmp_path)
    assert controller.command("start")["state"] == "recording"
    controller.tick()
    clock.now = 1
    controller.tick()
    stopped = controller.command("stop")
    assert stopped["pending_frame_count"] == 2
    assert stopped["controls"]["success"]
    saved = controller.command(label)
    assert saved["state"] == "idle" and saved["episode_index"] == 1
    assert saved["last_episode"]["gate_passed"]
    assert len(writer.episodes) == 1
    assert controller.snapshot() == controller.snapshot()
    assert {name for name, _ in events} >= {"episode started", "episode stopped", "episode saved"}
    controller.shutdown("operator closed")
    assert writer.finalized == 1


def test_hard_cap_runs_without_any_browser_or_command(tmp_path: Path) -> None:
    """The owner timing rule stops at the hard cap even when no client is attached."""
    controller, clock, _, events = setup_controller(tmp_path)
    controller.command("start")
    controller.tick()
    clock.now = 30
    controller.tick()
    state = controller.snapshot()
    assert state["state"] == "stopped" and state["elapsed_s"] == 30
    assert state["pending_frame_count"] == 1
    assert ("episode stopped", {"reason": "hard cap", "episode_index": 0}) in events
    controller.shutdown("test finished", force=True)


def test_empty_stopped_capture_only_offers_discard(tmp_path: Path) -> None:
    """A stop before the first accepted sample cannot advertise an invalid save."""
    controller, _, _, _ = setup_controller(tmp_path)
    controller.command("start")
    state = controller.command("stop")
    assert state["pending_frame_count"] == 0
    assert state["controls"]["discard"]
    assert not any(value for name, value in state["controls"].items() if name != "discard")
    controller.command("discard", reason="operator interruption")
    controller.shutdown("done")


@pytest.mark.parametrize("pending", [False, True])
def test_closed_controller_disables_every_capture_control(tmp_path: Path, pending: bool) -> None:
    """A restored page cannot offer capture actions after idle or forced shutdown."""
    controller, _, _, _ = setup_controller(tmp_path)
    if pending:
        controller.command("start")
        controller.tick()
    controller.shutdown("done", force=pending)
    state = controller.snapshot()
    assert state["state"] == "closed"
    assert not any(state["controls"].values())


def test_save_failure_retains_frames_then_retry_preserves_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed transaction remains retryable without permitting an accidental relabel."""
    controller, _, writer, events = setup_controller(tmp_path)
    original = writer.write_episode
    calls = 0

    def write(episode: Any) -> None:
        """Fail the first transaction before delegating the second unchanged episode."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic disk fault")
        original(episode)

    monkeypatch.setattr(writer, "write_episode", write)
    controller.command("start")
    controller.tick()
    controller.command("stop")
    with pytest.raises(OSError, match="disk fault"):
        controller.command("failure")
    state = controller.snapshot()
    assert state["pending_frame_count"] == 1 and state["controls"]["retry"]
    assert not state["controls"]["success"]
    with pytest.raises(RuntimeError, match="already has a label"):
        controller.command("success")
    controller.command("retry")
    assert writer.episodes[0].operator_label.value == "failure"
    assert {name for name, _ in events} >= {"episode save failed", "episode retried"}
    controller.shutdown("done")


def test_catalog_failure_after_save_cannot_duplicate_dataset(tmp_path: Path) -> None:
    """A crash between dataset and catalog writes requires reindex, not dataset retry."""
    controller, _, writer, _ = setup_controller(tmp_path)

    def failed_catalog(episode: Any) -> dict[str, Any]:
        """Represent unavailable catalog storage after the dataset transaction commits."""
        raise OSError("catalog unavailable")

    controller._on_saved = failed_catalog
    controller.command("start")
    controller.tick()
    controller.command("stop")
    result = controller.command("success")
    assert result["state"] == "idle" and result["episode_index"] == 1
    assert "was saved" in result["catalog_error"]
    assert not result["controls"]["start"]
    for command in ("start", "countdown"):
        with pytest.raises(RuntimeError, match="close, reindex and reopen"):
            controller.command(command)
    with pytest.raises(RuntimeError, match="no failed labeled episode"):
        controller.command("retry")
    assert len(writer.episodes) == 1
    controller.shutdown("done")


def test_audit_failure_preserves_pending_resolution_but_blocks_next_capture(tmp_path: Path) -> None:
    """An unavailable audit log must not accumulate more unaudited demonstrations."""
    controller, _, _, _ = setup_controller(tmp_path)

    def unavailable(kind: str, detail: dict[str, Any]) -> None:
        """Represent catalog storage failure without damaging the recorder transaction."""
        raise OSError("catalog unavailable")

    controller._emit_event = unavailable
    controller.command("start")
    controller.tick()
    controller.command("stop")
    saved = controller.command("failure")
    assert saved["episode_index"] == 1 and saved["catalog_error"]
    assert not saved["controls"]["start"]
    with pytest.raises(RuntimeError, match="reconciliation"):
        controller.command("start")
    controller.shutdown("done")


@pytest.mark.parametrize("refusal", ["disk", "guard", "recovery"])
def test_preflight_refusal_preserves_idle_state_and_episode_index(
    tmp_path: Path,
    refusal: str,
) -> None:
    """The console cannot bypass disk, graph or unresolved-recovery checks."""
    controller, _, writer, events = setup_controller(tmp_path)
    if refusal == "disk":
        controller._disk_free = lambda: 99
    elif refusal == "recovery":
        writer.recovery_blocked = True
    else:

        def refuse(topics: tuple[str, ...]) -> None:
            """Model an existing publisher found by the source's real guard interface."""
            raise RuntimeError("command publisher exists")

        controller.recorder._command_publisher_guard = refuse
    with pytest.raises(RuntimeError):
        controller.command("start")
    assert controller.recorder.state is RecorderState.IDLE
    assert controller.recorder.next_episode_index == 0
    assert controller.recorder.pending_frame_count == 0
    assert any(name == "preflight refused" for name, _ in events)
    writer.recovery_blocked = False
    controller.shutdown("done")


def test_discard_requires_reason_and_recovery_preserves_pending_frames(tmp_path: Path) -> None:
    """Discard is deliberate, audited, and unavailable while rollback evidence is blocked."""
    controller, _, writer, events = setup_controller(tmp_path)
    controller.command("start")
    controller.tick()
    with pytest.raises(ValueError, match="reason"):
        controller.command("discard")
    writer.recovery_blocked = True
    with pytest.raises(RuntimeError, match="recovery"):
        controller.command("discard", reason="capture fault")
    assert controller.recorder.pending_frame_count == 1
    writer.recovery_blocked = False
    controller.command("discard", reason="operator interruption")
    assert controller.recorder.pending_frame_count == 0
    assert any(name == "episode discarded" for name, _ in events)
    controller.shutdown("done")


def test_signal_shutdown_reports_lost_pending_frames_and_closes_sources(tmp_path: Path) -> None:
    """Normal quit refuses memory loss; forced signal cleanup records the exact lost count."""
    controller, _, writer, events = setup_controller(tmp_path)
    controller.command("start")
    controller.tick()
    with pytest.raises(RuntimeError, match="pending frames"):
        controller.shutdown("page quit")
    controller.shutdown("SIGTERM", force=True)
    assert writer.finalized == 1
    loss = [detail for name, detail in events if name == "pending frames lost"]
    assert loss[0]["pending_frame_count"] == 1 and loss[0]["reason"] == "SIGTERM"
    assert controller.recorder.state_source.closed


def test_countdown_and_cancellation_are_server_owned(tmp_path: Path) -> None:
    """No page timer can start early, duplicate a countdown or cancel pending capture."""
    controller, clock, _, events = setup_controller(tmp_path)
    controller.command("countdown", seconds=3)
    with pytest.raises(RuntimeError, match="countdown"):
        controller.command("start")
    clock.now = 2
    controller.tick()
    assert controller.snapshot()["countdown_s"] == 1
    controller.command("cancel_countdown")
    clock.now = 4
    controller.tick()
    assert controller.recorder.state is RecorderState.IDLE
    controller.command("countdown", seconds=1)
    clock.now = 5
    controller.tick()
    assert controller.recorder.state is RecorderState.RECORDING
    assert any(name == "episode countdown cancelled" for name, _ in events)
    controller.shutdown("done", force=True)


def test_capture_failure_stops_but_retains_previous_frames(tmp_path: Path) -> None:
    """Source errors stop sampling without losing accepted memory or pretending it was saved."""
    controller, clock, _, events = setup_controller(tmp_path)
    controller.command("start")
    controller.tick()
    controller.recorder.state_source.close()
    clock.now = 1
    controller.tick()
    assert controller.recorder.state is RecorderState.STOPPED
    assert controller.recorder.pending_frame_count == 1
    assert "Capture failed" in controller.snapshot()["error"]
    assert any(detail.get("reason") == "capture fault" for _, detail in events)
    controller.shutdown("done", force=True)


def test_preview_never_invokes_capture_and_reports_closed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request thread reads latest sources without advancing capture or acquiring its lock."""
    controller, _, _, _ = setup_controller(tmp_path)
    monkeypatch.setattr(controller.recorder, "capture_once", lambda: pytest.fail("capture called"))
    assert controller.preview_frame("front").data.shape == (24, 32, 3)
    assert controller.snapshot()["cameras"]["front"]["stale"] is False
    controller.shutdown("done")
    assert controller.snapshot()["cameras"]["front"]["stale"] is True


def test_worker_owns_mutations_and_shutdown_with_no_real_sleeps(tmp_path: Path) -> None:
    """Queued commands synchronize with one worker; tests wait on results rather than sleeping."""
    controller, _, writer, _ = setup_controller(tmp_path)
    identities: list[int] = []
    controller.recorder._command_publisher_guard = lambda topics: identities.append(
        threading.get_ident()
    )
    controller.start_worker()
    controller.command("start")
    controller.command("stop")
    controller.command("failure")
    controller.shutdown("done")
    assert identities == [controller._thread.ident]
    assert identities[0] != threading.get_ident()
    assert writer.finalized == 1


@pytest.mark.parametrize("floor", [0, -1, True, 1.5, float("inf")])
def test_disk_floor_cannot_be_disabled(tmp_path: Path, floor: Any) -> None:
    """Invalid or zero thresholds do not create a hidden disk-guard bypass."""
    controller, _, _, _ = setup_controller(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        RecordingController(
            controller.recorder,
            emit_event=lambda *args: None,
            on_saved=lambda episode: {},
            min_free_bytes=floor,
        )
    controller.shutdown("done")


def test_smoke_stays_one_episode_until_review_and_disposal(tmp_path: Path) -> None:
    """The console cannot extend the recorder's disposable one-episode semantics."""
    controller, _, writer, _ = setup_controller(tmp_path)
    contract = replace(controller.recorder.config.contract, recording_purpose="disposable_smoke")
    controller.recorder.config = replace(controller.recorder.config, contract=contract, smoke=True)
    controller.command("start")
    controller.tick()
    controller.command("stop")
    controller.command("success")
    with pytest.raises(RuntimeError, match="smoke is complete"):
        controller.command("start")
    assert len(writer.episodes) == 1
    controller.shutdown("dispose smoke")

"""Verify the disposable smoke capture leaves a citable report with frame and camera facts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_console_service import ConsoleFixture, values
from test_physical_recorder import FakeClock, FakeWriter

from synria_lerobot import recorder
from synria_lerobot.console.app import PRIVATE_SETTINGS, smoke_episode_facts
from synria_lerobot.physical_contract import ImageFrame, PhysicalState


class ArrayFrameSource:
    """A camera that returns a small real image stamped by the fake clock on every read."""

    def __init__(self, label: str, clock: FakeClock) -> None:
        self.label = label
        self.clock = clock
        self.closed = False

    def read(self) -> ImageFrame:
        return ImageFrame(np.full((8, 8, 3), 127, dtype=np.uint8), self.clock(),
                          native_resolution=(8, 8), source_id=self.label)

    def close(self) -> None:
        self.closed = True


class ClockStateSource:
    """A follower at rest whose samples are stamped by the fake clock."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.closed = False

    def read(self) -> PhysicalState:
        now = self.clock()
        return PhysicalState((0.0,) * 6, 0.01, now, now, ros_arrival_stamp_s=now)

    def close(self) -> None:
        self.closed = True


def pending_at_least(controller: Any, count: int) -> Any:
    """A predicate that holds once the capture holds at least ``count`` frames."""
    return lambda: controller.snapshot()["pending_frame_count"] >= count


def wait_for(predicate: Any, timeout_s: float = 5.0) -> None:
    """Poll the capture worker until the predicate holds."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("capture worker did not reach the expected state")


def test_smoke_close_writes_a_report_with_frame_and_camera_facts(tmp_path: Path) -> None:
    """Five samples at 15 FPS over a third of a second: counts, rates and dropped frames."""
    pytest.importorskip("cv2")
    clock = FakeClock()

    @contextmanager
    def fake_builder(args: argparse.Namespace, **kwargs: Any) -> Iterator[Any]:
        """Synthetic sources behind the real disposable contract."""
        assert args.smoke
        config = replace(recorder.recording_config_from_args(
            args, dataset_path=tmp_path / "temporary-smoke" / "dataset",
        ), state_source_provenance=None)
        capture = recorder.PhysicalEpisodeRecorder(
            config=config, state_source=ClockStateSource(clock),
            action_source=recorder.NextStateActionSource(),
            wrist_source=ArrayFrameSource("wrist", clock),
            front_source=ArrayFrameSource("front", clock),
            writer=FakeWriter(tmp_path), clock=clock,
        )
        try:
            yield capture
        finally:
            capture.close()

    workspace = tmp_path / "workspace"
    with ConsoleFixture(workspace, builder=fake_builder, clock=clock) as service:
        service.mutate(["smoke"], {
            **values(), "state_source": "standalone_driver", "wrist_camera": "fake-wrist",
            "front_camera": "fake-front", "follower_serial": "SECRET-SERIAL",
            "power_state_start": "on",
        })
        assert service.controller is not None
        controller = service.controller
        wait_for(pending_at_least(controller, 1))
        for step in range(1, 5):
            clock.now = step / 15
            wait_for(pending_at_least(controller, step + 1))
        clock.now = 1 / 3
        service.mutate(["record", "stop"], {})
        saved = service.mutate(["record", "success"], {})
        assert saved["state"] == "idle"
        assert not saved.get("error"), saved
        assert not saved.get("catalog_error"), saved
        service.close_session()
        reports = sorted((workspace / "smoke-reports").glob("smoke_physical_*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["recording_purpose"] == "disposable_smoke"
        assert report["source_kind"] == "physical"
        assert report["evidence_class"] == "physical capture"
        assert report["started_utc"] and report["ended_utc"] >= report["started_utc"]
        assert report["exit_reason"] == "operator closed" and report["disposal_succeeded"]
        assert "SECRET-SERIAL" not in reports[0].read_text()
        assert not set(report["settings"]) & PRIVATE_SETTINGS
        assert report["settings"]["fps"] == 15
        assert report["settings"]["wrist_camera"] == "fake-wrist"
        [episode] = report["episodes"]
        assert episode["frame_count"] == 5 and episode["requested_fps"] == 15
        assert episode["duration_s"] == pytest.approx(1 / 3)
        assert episode["expected_frame_count"] == 5 and episode["dropped_frames"] == 0
        assert episode["achieved_sample_rate_hz"] == pytest.approx(15.0)
        for camera in ("wrist_camera", "front_camera"):
            assert episode[camera]["distinct_frames"] == 5
            assert episode[camera]["repeated_frames"] == 0
            assert episode[camera]["delivered_fps"] == pytest.approx(15.0)
        assert episode["gate_passed"] is not None and episode["operator_label"] == "success"
        written = [e for e in service.catalog.events() if e["kind"] == "smoke_report_written"]
        assert len(written) == 1 and written[0]["detail"]["path"] == str(reports[0])
        assert written[0]["detail"]["episodes"] == 1 and written[0]["session"] is None


def test_smoke_episode_facts_count_dropped_and_repeated_frames() -> None:
    """A capture that fell behind shows dropped frames; a stalled camera shows repeats."""
    from synria_lerobot.physical_contract import CONTRACT_VERSION, ImageFrame, PhysicalFrame
    from synria_lerobot.recorder import OperatorLabel, RecordedPhysicalEpisode
    from synria_lerobot.task_registry import DISPOSABLE_SMOKE, load_task_registry

    task = load_task_registry(Path("config/synria_tasks.json"))["die_into_cup"]
    wrist_stamps = [0.0, 0.1, 0.1, 0.3]
    frames = [
        PhysicalFrame(
            PhysicalState((0.0,) * 6, 0.01, t, t, ros_arrival_stamp_s=t), (0.0,) * 7, t,
            ImageFrame(b"w", w), ImageFrame(b"f", t), sample_monotonic_timestamp_s=t,
        )
        for t, w in zip([0.0, 0.1, 0.2, 0.3], wrist_stamps, strict=True)
    ]
    episode = RecordedPhysicalEpisode(
        0, frames, 0.0, 1.0, OperatorLabel.SUCCESS, recorder.ActionSourceKind.NEXT_STATE,
        CONTRACT_VERSION, "50mm", smoke=True,
        task_definition=task, declared_recording_purpose=DISPOSABLE_SMOKE,
    )
    facts = smoke_episode_facts(episode, 10.0)
    assert facts["frame_count"] == 4 and facts["expected_frame_count"] == 10
    assert facts["dropped_frames"] == 6
    assert facts["wrist_camera"] == {"distinct_frames": 3, "delivered_fps": 3.0,
                                     "repeated_frames": 1}
    assert facts["front_camera"]["distinct_frames"] == 4

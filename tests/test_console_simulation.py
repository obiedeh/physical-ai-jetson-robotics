"""Verify the simulation source: purpose, fixed sim topics, not-applicable arm facts, labels."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_console_service import ConsoleFixture, values
from test_task_registry import synthetic_registry

from synria_lerobot import recorder
from synria_lerobot.console.app import ConsoleService, announce_url
from synria_lerobot.console.readiness import (
    ISAAC_RECORD_HELP,
    SIM_TOPICS,
    SIMULATION_CONFIRMATIONS,
    ReadinessReport,
)
from synria_lerobot.task_registry import (
    QUALIFYING,
    SIMULATION,
    SIMULATION_UNTIMED_WINDOW_S,
    load_task_source,
    recording_purpose,
)


def sim_args(tmp_path: Path, registry: Path, *, smoke: bool = False) -> argparse.Namespace:
    """The namespace the console builds for a simulation session."""
    return argparse.Namespace(
        task_registry=registry, task_id="die_into_cup", gripper_type="50mm",
        action_source="leader", state_has_velocity=False, action_lookahead_steps=1,
        repo_id="local/synria-sim-test", dataset_path=tmp_path / "dataset", smoke=smoke,
        fps=15, image_width=224, image_height=224, state_source="isaac_simulation",
        follower_topic=SIM_TOPICS["state"], guard_command_topic=[],
        leader_topic=SIM_TOPICS["action"], wrist_camera=SIM_TOPICS["wrist"],
        front_camera=SIM_TOPICS["front"], state_startup_timeout_s=10.0,
    )


def test_simulation_purpose_has_its_own_window_and_is_never_qualifying(tmp_path: Path) -> None:
    """An untimed task gets the simulation window; a timed task keeps its real window."""
    untimed_registry = synthetic_registry(tmp_path / "untimed.json", None, None)
    untimed = load_task_source(untimed_registry)["die_into_cup"]
    window = untimed.recording_window(SIMULATION)
    assert (window["min_episode_s"], window["max_episode_s"]) == SIMULATION_UNTIMED_WINDOW_S
    timed = load_task_source(synthetic_registry(tmp_path / "timed.json", 4, 40))["die_into_cup"]
    assert timed.recording_window(SIMULATION) == timed.recording_window(QUALIFYING)
    assert "simulation" in timed.metadata(SIMULATION)["episode_window_override"]["reason"]
    assert recording_purpose(False, SIMULATION) == SIMULATION
    with pytest.raises(ValueError):
        recording_purpose(True, SIMULATION)


def test_simulation_config_declares_sim_state_source_and_refuses_physical_claims(
    tmp_path: Path,
) -> None:
    """Simulation takes carry the simulation purpose; a sim practice run is a disposable smoke."""
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    config = recorder.recording_config_from_args(sim_args(tmp_path, registry), simulation=True)
    assert config.contract.recording_purpose == SIMULATION
    assert config.state_source_provenance is not None
    assert config.state_source_provenance.state_source == "isaac_simulation"
    with pytest.raises(ValueError, match="not eligible"):
        config.contract.require_qualifying()
    smoke = recorder.recording_config_from_args(sim_args(tmp_path, registry, smoke=True),
                                                simulation=True)
    assert smoke.contract.recording_purpose == "disposable_smoke"
    physical = sim_args(tmp_path, registry)
    physical.state_source = "standalone_driver"
    with pytest.raises(ValueError, match="isaac_simulation"):
        recorder.recording_config_from_args(physical, simulation=True)
    with pytest.raises(ValueError, match="distinct launch modes"):
        recorder.recording_config_from_args(sim_args(tmp_path, registry), synthetic_demo=True,
                                            simulation=True)


def test_image_message_decoding_accepts_rgb8_and_bgr8_and_resizes() -> None:
    """A 224 px rgb8 payload is stored as is; bgr8 is swapped; other sizes are resized."""
    native = np.zeros((4, 6, 3), dtype=np.uint8)
    native[..., 0] = 200
    stored, kept = recorder.decode_image_message("rgb8", 4, 6, 18, native.tobytes(), (6, 4))
    assert stored.shape == (4, 6, 3) and stored[0, 0, 0] == 200 and kept is stored
    swapped, _ = recorder.decode_image_message("bgr8", 4, 6, 18, native.tobytes(), (6, 4))
    assert swapped[0, 0, 2] == 200 and swapped[0, 0, 0] == 0
    resized, _ = recorder.decode_image_message("rgb8", 4, 6, 18, native.tobytes(), (3, 2))
    assert resized.shape == (2, 3, 3) and resized[0, 0, 0] == 200
    with pytest.raises(ValueError, match="encoding"):
        recorder.decode_image_message("mono8", 4, 6, 6, bytes(24), (6, 4))
    with pytest.raises(ValueError, match="shorter"):
        recorder.decode_image_message("rgb8", 4, 6, 18, bytes(10), (6, 4))


def test_simulation_session_fixes_sim_topics_and_needs_no_real_arm_facts(tmp_path: Path) -> None:
    """No serial or power state; sources are the Isaac topics; the dataset is marked simulation."""
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    with ConsoleFixture(tmp_path / "workspace", simulation=True, registry=registry) as service:
        with pytest.raises(ValueError, match="Tick the box to confirm you have checked the arm"):
            service.create_session({**values(), "fps": 15})
        session = service.create_session({**values(), "fps": 15, "arm_checked": True})
        assert session["settings"]["arm_checked"] is True
        assert session["settings"]["arm_checked_utc"].endswith("+00:00")
        assert session["settings"]["arm_checked_statement"] == (
            "I have physically checked the arm: the area around it is clear and the leader arm "
            "drives the sim arm.")
        assert session["source_kind"] == "simulation"
        assert session["recording_purpose"] == "simulation"
        assert session["settings"]["state_source"] == "isaac_simulation"
        assert session["settings"]["wrist_camera"] == SIM_TOPICS["wrist"]
        assert session["settings"]["front_camera"] == SIM_TOPICS["front"]
        assert session["settings"]["follower_topic"] == SIM_TOPICS["state"]
        assert session["settings"]["action_source"] == "leader"
        assert session["settings"]["leader_topic"] == SIM_TOPICS["action"]
        assert session["repo_id"].startswith("local/synria-sim-")
        assert "sessions-simulation" in session["dataset_path"]
        assert service.state()["simulation"] is True
        assert service.state()["source_kind"] == "simulation"
        cameras = service.read(["cameras"], {})["cameras"]
        assert cameras == [SIM_TOPICS["wrist"], SIM_TOPICS["front"]]
        assert service.read(["serial-connections"], {})["connections"] == []
        assert service.setup_suggestions()["recommended"]["action_source"] == "leader"
    with ConsoleFixture(tmp_path / "physical", registry=registry) as physical:
        physical_session = physical.create_session({
            **values(), "state_source": "standalone_driver", "wrist_camera": "fake-wrist",
            "front_camera": "fake-front", "follower_serial": "fake", "power_state_start": "fake",
        })
        assert physical_session["source_kind"] == "physical"
    with pytest.raises(ValueError, match="distinct launch modes"):
        ConsoleService(tmp_path / "both", demo=True, simulation=True, registry=registry)


def test_simulation_readiness_marks_real_arm_facts_not_applicable_and_names_isaac(
    tmp_path: Path,
) -> None:
    """Serial, power and limit verification are not applicable; a missing Isaac feed is named."""
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    seen: dict[str, Any] = {}

    @contextmanager
    def refusing_builder(args: argparse.Namespace, **kwargs: Any):
        """Stand in for the recorder builder: note the mode and fail like an absent Isaac feed."""
        seen.update(kwargs)
        raise RuntimeError("No state sample received on /isaac_joint_states after waiting 10 s")
        yield  # pragma: no cover

    with ConsoleFixture(tmp_path / "workspace", simulation=True, registry=registry,
                        builder=refusing_builder) as service:
        confirmations = {name: True for name in SIMULATION_CONFIRMATIONS}
        assert list(confirmations) == ["arm_checked"]
        report = service.check_readiness({"settings": {**values(), "fps": 15},
                                          "confirmations": confirmations})
        checks = report["checks"]
        assert checks["recording_metadata"]["status"] == "not_applicable"
        assert "real arm" in checks["recording_metadata"]["message"]
        assert checks["limits_verification"]["status"] == "not_applicable"
        assert "joint limit gate still applies" in checks["limits_verification"]["message"]
        assert checks["operator_safety"]["passed"] is True
        assert checks["camera_mapping"]["passed"] is True
        assert checks["isaac_record_mode"]["passed"] is False
        assert checks["isaac_record_mode"]["message"] == ISAAC_RECORD_HELP
        assert checks["diagnostic"]["message"] == ISAAC_RECORD_HELP
        assert "/isaac_joint_states" in checks["diagnostic"]["details"]["technical"]
        assert "Start Isaac record mode" in ISAAC_RECORD_HELP and "/isaac" not in ISAAC_RECORD_HELP
        assert seen["simulation"] is True and seen["synthetic_demo"] is False
        assert report["label"].startswith("SIMULATION")
        half = service.check_readiness({"settings": {**values(), "fps": 15},
                                        "confirmations": {"arm_checked": False}})
        assert half["checks"]["operator_safety"]["passed"] is False
        assert (half["checks"]["diagnostic"]["message"]
                == "Tick the box to confirm you have checked the arm")


def test_not_applicable_counts_as_complete_but_is_labelled() -> None:
    """A not-applicable fact completes the meter without claiming a physical verification."""
    report = ReadinessReport(lambda: 0.0, demo=False, simulation=True)
    report.begin()
    assert "isaac_record_mode" in report.checks
    report.not_applicable("limits_verification", "Not applicable in simulation")
    assert report.checks["limits_verification"] == {
        "passed": True, "status": "not_applicable", "message": "Not applicable in simulation"}
    report.finish()
    assert report.snapshot()["label"].startswith("SIMULATION · ")
    assert report.snapshot()["simulation"] is True


def test_simulation_url_announcement_names_the_mode() -> None:
    """The printed launch line says SIMULATION so a log reader cannot mistake it for physical."""
    import io
    stream = io.StringIO()
    announce_url("http://127.0.0.1:8768/#token=abc", demo=False, simulation=True, stream=stream)
    assert stream.getvalue().startswith("Synria Teleop Console SIMULATION: ")


def test_practice_run_and_open_session_wait_for_isaac_in_operator_words(tmp_path: Path) -> None:
    """With Isaac down, smoke and open-session refuse in Isaac wording, not the raw timeout."""
    from test_console_isaac import Fakes, supervisor
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    down = supervisor(tmp_path, Fakes([]))
    with ConsoleFixture(tmp_path / "ws", simulation=True, registry=registry, isaac=down,
                        builder=lambda *a, **k: pytest.fail("opened sources")) as service:
        with pytest.raises(ValueError, match="Start Isaac record mode"):
            service.mutate(["smoke"], {**values(), "fps": 15, "arm_checked": True})
        session = service.create_session({**values(), "fps": 15, "arm_checked": True})
        with pytest.raises(ValueError, match="Start Isaac record mode"):
            service.open_session(session["id"])


def test_simulation_shows_the_side_view_as_a_preview_that_is_never_recorded(tmp_path: Path) -> None:
    """In simulation an open session carries a preview-only scene source; physical mode has none."""
    pytest.importorskip("cv2")
    from dataclasses import replace

    from test_console_isaac import RECORD, Fakes, proc, supervisor
    from test_console_smoke_report import ArrayFrameSource, ClockStateSource
    from test_physical_recorder import FakeClock, FakeWriter

    from synria_lerobot.physical_contract import ImageFrame

    clock = FakeClock()

    class FakeScene:
        """The side view subscriber: one frame on demand, closed with the session."""

        def __init__(self, topic: str) -> None:
            self.topic = topic
            self.closed = False

        def read(self) -> ImageFrame:
            return ImageFrame(np.full((8, 8, 3), 90, dtype=np.uint8), clock(),
                              native_resolution=(8, 8), source_id=self.topic)

        def close(self) -> None:
            self.closed = True

    scenes: list[FakeScene] = []

    def scene_source(topic: str) -> FakeScene:
        scenes.append(FakeScene(topic))
        return scenes[-1]

    @contextmanager
    def fake_builder(args: argparse.Namespace, **kwargs: Any) -> Any:
        """Synthetic sim sources behind the real simulation contract."""
        config = replace(recorder.recording_config_from_args(
            args, simulation=True, dataset_path=tmp_path / "practice" / "dataset",
        ), state_source_provenance=None)
        class FakeGoalSource:
            """The scene's object pose for the declared goal channels, fixed."""

            def read(self) -> recorder.GoalSample:
                return recorder.GoalSample((0.3, 0.0, 0.0) * len(config.contract.goal_channels),
                                           clock())

            def close(self) -> None:
                pass

        capture = recorder.PhysicalEpisodeRecorder(
            config=config, state_source=ClockStateSource(clock),
            action_source=recorder.LeaderActionSource(ClockStateSource(clock)),
            wrist_source=ArrayFrameSource("wrist", clock),
            front_source=ArrayFrameSource("front", clock),
            writer=FakeWriter(tmp_path), clock=clock,
            goal_source=FakeGoalSource() if config.contract.goal_channels else None,
        )
        try:
            yield capture
        finally:
            capture.close()

    ready = {101: {"commit": "abc1234", "scene_loaded": True, "topics_started": True}}
    running = supervisor(tmp_path, Fakes([proc(101, RECORD)], rate=15.2, markers=ready))
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    with ConsoleFixture(tmp_path / "ws", simulation=True, registry=registry, isaac=running,
                        builder=fake_builder, clock=clock, scene_source=scene_source) as service:
        with pytest.raises(RuntimeError, match="no scene preview is open"):
            service.media(["preview", "scene"], {})
        service.mutate(["smoke"], {**values(), "fps": 15, "arm_checked": True})
        assert service.state()["scene_preview"] is True
        assert [scene.topic for scene in scenes] == [SIM_TOPICS["scene"]]
        assert SIM_TOPICS["scene"] not in (SIM_TOPICS["wrist"], SIM_TOPICS["front"])
        jpeg = service.media(["preview", "scene"], {})
        assert jpeg[:2] == b"\xff\xd8"
        service.close_session()
        assert scenes[0].closed and service.state()["scene_preview"] is False
    with ConsoleFixture(tmp_path / "physical", registry=registry,
                        scene_source=scene_source) as physical:
        assert physical.state()["scene_preview"] is False


def test_a_sim_session_from_an_older_camera_layout_is_refused_in_operator_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored topics that are not this launch mode's recorded views refuse to reopen."""
    from test_console_isaac import RECORD, Fakes, proc, supervisor

    from synria_lerobot.console import app as console_app

    ready = {101: {"commit": "abc1234", "scene_loaded": True, "topics_started": True}}
    running = supervisor(tmp_path, Fakes([proc(101, RECORD)], rate=15.2, markers=ready))
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)

    def builder_reached(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("builder reached")

    with ConsoleFixture(tmp_path / "ws", simulation=True, registry=registry, isaac=running,
                        builder=builder_reached) as service:
        session = service.create_session({**values(), "fps": 15, "arm_checked": True})
        with pytest.raises(RuntimeError, match="builder reached"):
            service.open_session(session["id"])      # current layout: the guard lets it through
        monkeypatch.setitem(console_app.SIM_TOPICS, "front", "/isaac/elsewhere/image_raw")
        with pytest.raises(ValueError, match="older simulation camera layout") as refusal:
            service.open_session(session["id"])
        assert "front /isaac/overhead_camera/image_raw (now /isaac/elsewhere/image_raw)" in str(
            refusal.value)
        assert "Start a new session" in str(refusal.value)


def test_simulation_sessions_declare_the_task_goal_channels() -> None:
    """Every scene task declares its object; the place task also the mark; unknown none."""
    from synria_lerobot.console.isaac import sim_goal_channels

    assert sim_goal_channels("pick_cup") == ["object=/isaac/pick_objects/cup/pose"]
    assert sim_goal_channels("die_into_cup") == ["object=/isaac/pick_objects/die/pose"]
    assert sim_goal_channels("cup_pick_and_place") == [
        "object=/isaac/pick_objects/cup/pose", "target=/isaac/pick_objects/place_target/pose",
    ]
    assert sim_goal_channels("not_a_task") == []

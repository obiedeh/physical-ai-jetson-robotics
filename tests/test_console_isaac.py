"""Verify Isaac record mode supervision with fake process lists, signals and a fake launcher."""
from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest
from test_console_service import ConsoleFixture
from test_task_registry import synthetic_registry

from synria_lerobot.console import isaac
from synria_lerobot.console.isaac import (
    IsaacSupervisor,
    ProcessInfo,
    classify,
    launcher_from_config,
)

KIT = "/home/user/isaacsim/kit/python/bin/python3"
SCENE = "/home/user/github/teleop/scripts/isaac_sim/alicia_d_live_scene.py"
RECORD = (f"{KIT} {SCENE} --no-obstacles --pick-objects --physics-drive --sim-cameras "
          "--record-cameras --publish-sim-topics")
PHYSICS = f"{KIT} {SCENE} --no-obstacles --pick-objects --physics-drive --sim-cameras"
APP = "/home/user/isaacsim/kit/kit /home/user/isaacsim/apps/isaacsim.exp.full.kit"
LAB = "/home/user/IsaacLab/_isaac_sim/python.sh scripts/train.py --task Isaac-Lab-Thing"
DRIVER = ("/opt/ros/jazzy/lib/alicia_d_driver/alicia_d_driver_node --ros-args "
          "-p port:=/dev/serial/by-id/usb-1a86_X-if00")


def proc(pid: int, cmdline: str, started: str = "2026-10-12T22:00:00+00:00") -> ProcessInfo:
    """A fake process row."""
    return ProcessInfo(pid, cmdline, started)


class Fakes:
    """Injectable fakes: a process table, signals sent, spawns made, markers and rates."""

    def __init__(self, processes: list[ProcessInfo], rate: float | None = None,
                 markers: dict[int, dict[str, Any]] | None = None) -> None:
        self.processes = list(processes)
        self.signals: list[tuple[int, int]] = []
        self.spawned: list[tuple[list[str], dict[str, str], Path, Path]] = []
        self.rate = rate
        self.markers = markers or {}
        self.now = 1_000.0
        self.die_on_term = True

    def list_processes(self) -> list[ProcessInfo]:
        return list(self.processes)

    def send_signal(self, pid: int, sig: int) -> None:
        alive = any(p.pid == pid for p in self.processes)
        if not alive:
            raise ProcessLookupError(pid)
        if sig == 0:
            return
        self.signals.append((pid, sig))
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and self.die_on_term):
            self.processes = [p for p in self.processes if p.pid != pid]

    def spawn(self, argv: list[str], env: dict[str, str], log: Path, cwd: Path) -> int:
        self.spawned.append((argv, env, log, cwd))
        return 4242

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def supervisor(tmp_path: Path, fakes: Fakes, *, launcher: Path | None = None,
               commit: str = "abc1234") -> IsaacSupervisor:
    """A supervisor wired to the fakes; the launcher is a real empty file so the path validates."""
    if launcher is None:
        launcher = tmp_path / "scripts" / "run_alicia_stack.sh"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text("#!/bin/bash\n")
    return IsaacSupervisor(
        tmp_path / "workspace", tmp_path / "workspace" / "isaac-logs", launcher=launcher,
        list_processes=fakes.list_processes, gpu_memory=lambda: {101: 4002, 202: 3376},
        markers=lambda pid: fakes.markers.get(
            pid, {"commit": None, "scene_loaded": False, "topics_started": False}),
        topic_rate=lambda: fakes.rate, send_signal=fakes.send_signal, spawn=fakes.spawn,
        commit=lambda path: commit, clock=fakes.clock, sleep=fakes.sleep,
    )


def test_classification_names_isaac_families_and_never_the_driver() -> None:
    """The teleop scene, Isaac Sim app, Isaac Lab and Kit are Isaac; driver and shells are not."""
    assert classify(proc(1, RECORD)) == "teleop scene"
    assert classify(proc(2, APP)) == "Isaac Sim app"
    assert classify(proc(3, LAB)) == "Isaac Lab"
    kit = proc(4, "/home/user/isaacsim/kit/python/bin/python3 -m something")
    assert classify(kit) == "Kit process"
    assert classify(proc(5, DRIVER)) is None
    assert classify(proc(6, "/bin/bash /home/user/isaacsim/python.sh " + SCENE)) is None
    assert classify(proc(7, "python -m synria_lerobot.console.app --simulation")) is None
    assert "<device>" in isaac.masked(DRIVER) and "by-id" not in isaac.masked(DRIVER)


def test_states_not_running_running_other_and_starting(tmp_path: Path) -> None:
    """Each state names its button; a healthy record instance on the current commit is reused."""
    nothing = supervisor(tmp_path, Fakes([proc(9, DRIVER)]))
    state = nothing.state()
    assert state["status"] == "not_running"
    assert state["button"]["label"] == "Start Isaac record mode"
    assert state["driver"]["running"] is True
    assert "joint_commands_enabled:=false" in state["driver"]["read_only_command"]
    ready = {101: {"commit": "abc1234", "scene_loaded": True, "topics_started": True}}
    healthy = supervisor(tmp_path, Fakes([proc(101, RECORD)], rate=15.2, markers=ready))
    state = healthy.state()
    assert state["status"] == "running" and state["button"]["action"] is None
    assert state["can_stop"] is True
    keys = ("process_up", "scene_loaded", "topics_started", "state_rate_hz", "healthy")
    assert {k: state["progress"][k] for k in keys} == {
        "process_up": True, "scene_loaded": True, "topics_started": True,
        "state_rate_hz": 15.2, "healthy": True}
    assert state["progress"]["waiting"] is None and state["progress"]["topics"]["complete"] is True
    assert healthy.start()["reused"] is True
    other = supervisor(tmp_path, Fakes([proc(202, APP), proc(9, DRIVER)]))
    state = other.state()
    assert state["status"] == "other_running"
    assert state["button"]["label"] == "Close running Isaac and start record mode"
    assert [item["pid"] for item in state["button"]["closes"]] == [202]
    assert state["instances"][0]["gpu_mb"] == 3376
    assert state["driver"]["running"] is True
    assert all(item["kind"] != "driver" for item in state["instances"])
    slow = supervisor(tmp_path, Fakes([proc(101, PHYSICS)]))
    # Physics mode without the sim topics is not record mode.
    assert slow.state()["status"] == "other_running"
    up_not_publishing = supervisor(tmp_path, Fakes([proc(101, RECORD)], rate=0.0))
    assert up_not_publishing.state()["status"] == "starting"


def test_start_closes_only_this_users_isaac_then_launches_detached_without_the_driver(
    tmp_path: Path,
) -> None:
    """SIGTERM, then SIGKILL after the timeout, then the launcher in record mode, SKIP_DRIVER=1."""
    fakes = Fakes([proc(202, APP), proc(303, LAB), proc(9, DRIVER)])
    fakes.die_on_term = False
    sup = supervisor(tmp_path, fakes)
    result = sup.start()
    sent = [sig for _, sig in fakes.signals]
    assert sent == [signal.SIGTERM, signal.SIGTERM, signal.SIGKILL, signal.SIGKILL]
    assert {pid for pid, _ in fakes.signals} == {202, 303}       # never the driver
    assert all(row["result"] == "killed after timeout" for row in result["closed"])
    argv, env, log, cwd = fakes.spawned[0]
    assert argv == [str(sup.launcher), "record"] and env["SKIP_DRIVER"] == "1"
    assert log.parent == tmp_path / "workspace" / "isaac-logs" and cwd == sup.launcher.parent.parent
    assert result["status"] == "starting" and result["launch_pid"] == 4242
    with pytest.raises(RuntimeError, match="already starting"):
        sup.start()
    fakes.now += isaac.START_TIMEOUT_S + 1
    assert sup.state()["failure"].startswith("Isaac did not reach record mode in")
    fakes.processes.append(proc(101, RECORD))
    fakes.rate = 15.0
    sup2 = supervisor(tmp_path, fakes)
    sup2.start()
    assert sup2.state()["status"] == "starting" or sup2.state()["status"] == "running"
    fakes.markers[101] = {"commit": "abc1234", "scene_loaded": True, "topics_started": True}
    assert sup2.state()["status"] == "running"


def test_stop_closes_scene_instances_only(tmp_path: Path) -> None:
    """Stop Isaac terminates teleop scene processes, never the driver or an unrelated process."""
    fakes = Fakes([proc(101, RECORD), proc(9, DRIVER), proc(55, "/usr/bin/python3 other.py")])
    sup = supervisor(tmp_path, fakes)
    result = sup.stop()
    assert fakes.signals == [(101, signal.SIGTERM)] and result["status"] == "not_running"


def test_launcher_path_comes_from_environment_or_workspace_and_is_validated(tmp_path: Path) -> None:
    """No launcher means no button; a configured path must be an absolute, existing launcher."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert launcher_from_config(workspace, {}) is None
    good = tmp_path / "teleop" / "scripts" / "run_alicia_stack.sh"
    good.parent.mkdir(parents=True)
    good.write_text("#!/bin/bash\n")
    assert launcher_from_config(workspace, {isaac.LAUNCHER_ENV: str(good)}) == good
    (workspace / isaac.LAUNCHER_FILE).write_text(f'{{"launcher": "{good}"}}')
    assert launcher_from_config(workspace, {}) == good
    with pytest.raises(ValueError, match="absolute path"):
        launcher_from_config(workspace, {isaac.LAUNCHER_ENV: "scripts/run_alicia_stack.sh"})
    with pytest.raises(ValueError, match="absolute path"):
        launcher_from_config(workspace, {isaac.LAUNCHER_ENV: str(tmp_path / "evil.sh")})
    sup = IsaacSupervisor(workspace, workspace / "logs", launcher=None, list_processes=list,
                          gpu_memory=dict, topic_rate=lambda: None)
    assert sup.state()["available"] is False
    with pytest.raises(ValueError, match="not configured"):
        sup.start()


def test_console_routes_are_simulation_only_and_take_no_parameters(tmp_path: Path) -> None:
    """Physical mode has no Isaac control; simulation exposes state, start and stop."""
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    fakes = Fakes([])
    sup = supervisor(tmp_path, fakes)
    with ConsoleFixture(tmp_path / "sim", simulation=True, registry=registry, isaac=sup) as service:
        assert service.read(["isaac"], {})["status"] == "not_running"
        with pytest.raises(ValueError, match="no parameters"):
            service.mutate(["isaac", "start"], {"command": "rm -rf /"})
        started = service.mutate(["isaac", "start"], {})
        assert started["status"] == "starting" and fakes.spawned[0][0][1] == "record"
        assert service.mutate(["isaac", "stop"], {})["status"] in {"not_running", "starting"}
    with ConsoleFixture(tmp_path / "physical", registry=registry) as physical:
        assert physical.read(["isaac"], {})["available"] is False
        with pytest.raises(ValueError, match="simulation mode"):
            physical.mutate(["isaac", "start"], {})


def test_read_only_driver_start_is_fixed_and_only_when_no_driver_runs(tmp_path: Path) -> None:
    """The driver command is the read-only launch form, no serial; a running driver blocks it."""
    fakes = Fakes([])
    sup = supervisor(tmp_path, fakes)
    result = sup.start_driver()
    argv, env, log, cwd = fakes.spawned[0]
    assert argv[:2] == ["bash", "-c"] and len(argv) == 3
    assert "joint_commands_enabled:=false" in argv[2] and "torque_off_on_start:=false" in argv[2]
    assert "true" not in argv[2] and "/dev/" not in argv[2] and "port:=" not in argv[2]
    assert argv[2].endswith("exec " + isaac.READ_ONLY_DRIVER_COMMAND)
    assert log.name.startswith("arm_driver_readonly_")
    assert log.parent == tmp_path / "workspace" / "isaac-logs"
    assert result["driver_launch_pid"] == 4242 and sup.driver()["can_start"] is True
    running = supervisor(tmp_path, Fakes([proc(9, DRIVER)]))
    assert running.driver()["can_start"] is False
    with pytest.raises(RuntimeError, match="already running"):
        running.start_driver()
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    with ConsoleFixture(tmp_path / "sim", simulation=True, registry=registry, isaac=sup) as service:
        with pytest.raises(ValueError, match="no parameters"):
            service.mutate(["isaac", "start-driver"], {"port": "/dev/ttyACM0"})
        assert service.mutate(["isaac", "start-driver"], {})["driver_launch_pid"] == 4242


def test_readiness_waits_for_detected_isaac_record_mode(tmp_path: Path) -> None:
    """With a launcher configured, the check refuses in operator words until record mode is seen."""
    from contextlib import contextmanager
    registry = synthetic_registry(tmp_path / "tasks.json", 4, 40)
    opened: list[str] = []

    @contextmanager
    def builder(args: Any, **kwargs: Any):
        opened.append("opened")
        raise RuntimeError("No image received on /isaac/wrist_camera/image_raw")
        yield  # pragma: no cover

    down = supervisor(tmp_path, Fakes([]))
    settings = {"name": "t", "task_id": "die_into_cup", "gripper_type": "50mm", "fps": 15,
                "operator": "o", "scene": "s"}
    with ConsoleFixture(tmp_path / "down", simulation=True, registry=registry, isaac=down,
                        builder=builder) as service:
        report = service.check_readiness({"settings": settings,
                                          "confirmations": {"arm_checked": True}})
        assert opened == []
        assert report["checks"]["isaac_record_mode"]["passed"] is False
        assert report["checks"]["diagnostic"]["message"].startswith("Start Isaac record mode first")
        assert "physical" not in report["checks"]["diagnostic"]["message"]
    @contextmanager
    def busy_camera(args: Any, **kwargs: Any):
        opened.append("opened")
        raise RuntimeError("camera device busy")
        yield  # pragma: no cover

    ready = {101: {"commit": "abc1234", "scene_loaded": True, "topics_started": True}}
    up = supervisor(tmp_path, Fakes([proc(101, RECORD)], rate=15.1, markers=ready))
    with ConsoleFixture(tmp_path / "up", simulation=True, registry=registry, isaac=up,
                        builder=busy_camera) as service:
        report = service.check_readiness({"settings": settings,
                                          "confirmations": {"arm_checked": True}})
        assert opened == ["opened"] and report["checks"]["isaac_record_mode"]["passed"] is True


def test_steady_needs_all_three_topics_over_a_full_window_with_tolerance() -> None:
    """14.8 Hz with short gaps is steady; a silent camera, a long gap or a partial window is not."""
    good = {"rates": {"state": 14.8, "wrist": 14.8, "front": 14.8},
            "max_gap_s": {"state": 0.072, "wrist": 0.072, "front": 0.072},
            "complete": True, "error": None}
    assert isaac.steady(good)
    assert not isaac.steady({**good, "rates": {**good["rates"], "front": None}})
    assert not isaac.steady({**good, "rates": {**good["rates"], "wrist": 12.0}})
    assert not isaac.steady({**good, "max_gap_s": {**good["max_gap_s"], "state": 0.8}})
    assert not isaac.steady({**good, "complete": False})
    assert not isaac.steady({**good, "error": "ROS is unavailable"})
    assert not isaac.steady(None)


def test_waiting_words_name_the_step_and_the_time(tmp_path: Path) -> None:
    """The panel says what it waits for, not a bare refusal, and times out with the log path."""
    assert isaac.waiting_words(False, False, False, None, 3) == "Waiting for the Isaac process, 3 s"
    assert isaac.waiting_words(True, False, False, None, 12) == "Isaac is loading the scene, 12 s"
    assert (isaac.waiting_words(True, True, False, None, 20)
            == "Scene loaded, waiting for the sim topics, 20 s")
    assert (isaac.waiting_words(True, True, True, {"complete": False}, 21)
            == "Scene loaded, measuring the sim topics, 21 s")
    low = {"complete": True, "rates": {"state": 14.9, "wrist": 2.0, "front": None}}
    assert isaac.waiting_words(True, True, True, low, 25) == (
        "Scene loaded, waiting for camera feeds and state at 15 Hz (wrist 2.0 Hz, front silent), "
        "25 s")
    ready = {101: {"commit": "abc1234", "scene_loaded": True, "topics_started": True}}
    fakes = Fakes([proc(101, RECORD)], rate=None, markers=ready)
    sup = supervisor(tmp_path, fakes)
    sup.start.__func__  # noqa: B018  (start is exercised elsewhere; here the record comes from a timed-out launch)
    sup._write_launch({"status": "starting", "launch_pid": 1, "started_at": fakes.now - 30,
                       "commit": "abc1234", "log": "/tmp/x.log"})
    state = sup.state()
    assert state["status"] == "starting"
    assert state["label"].startswith("Scene loaded, measuring the sim topics, 30 s")
    fakes.now += isaac.START_TIMEOUT_S
    state = sup.state()
    assert state["failure"].startswith("Isaac did not reach record mode in")
    assert state["launch_log"] == "/tmp/x.log"
    fakes.rate = 14.8
    state = sup.state()
    assert state["status"] == "running" and state["failure"] is None


def test_a_launch_whose_isaac_process_dies_fails_at_once_with_the_log_errors(
    tmp_path: Path,
) -> None:
    """A dead Isaac process is reported on the next poll with its traceback; Relaunch is offered."""
    fakes = Fakes([proc(9, DRIVER), proc(4242, "bash /x/run_alicia_stack.sh record")])
    sup = supervisor(tmp_path, fakes)
    result = sup.start()
    log = Path(result["last_launch"]["log"])
    isaac_log = tmp_path / "alicia_isaac_launch.log"
    log.write_text("  Launching Isaac Sim...\n  Isaac Sim PID: 777\n  Logs: "
                   f"{isaac_log}\n\nStack started in 'record' mode.\n")
    fakes.processes.append(proc(777, "bash /x/scripts/isaac_sim/run_alicia_d_live_scene.sh"))
    fakes.now += 5
    state = sup.state()
    assert state["status"] == "starting" and state["failure"] is None
    assert state["last_launch"]["isaac_pid"] == 777
    assert state["isaac_log"] == str(isaac_log)
    isaac_log.write_text(
        "2026-10-17T03:03:20Z [7,873ms] [Warning] [omni.usd] something harmless\n"
        "2026-10-17T03:03:20Z [7,914ms] [Error] [omni.kit.app._impl] [py stderr]: "
        "Traceback (most recent call last):\n"
        "2026-10-17T03:03:20Z [7,914ms] [Error] [omni.kit.app._impl] [py stderr]:   File "
        "\"scene.py\", line 610, in spawn_pick_objects\n"
        "2026-10-17T03:03:20Z [7,915ms] [Error] [omni.kit.app._impl] [py stderr]: "
        "pxr.Tf.ErrorException: Path must be an absolute path: <>\n"
        "2026-10-17T03:03:21Z [8,795ms] [Warning] [carb] Recursive unloadAllPlugins() detected!\n"
    )
    fakes.processes = [p for p in fakes.processes if p.pid not in {777, 4242}]
    fakes.now += 4
    state = sup.state()
    assert state["status"] == "not_running" and state["label"] == "Last launch failed"
    assert state["failure"] == ("Isaac process 777 exited after 9 s before reaching record "
                                "mode; see the Isaac log")
    assert state["failure_detail"] == [
        "Traceback (most recent call last):",
        'File "scene.py", line 610, in spawn_pick_objects',
        "pxr.Tf.ErrorException: Path must be an absolute path: <>",
    ]
    assert state["button"] == {"action": "start", "label": "Relaunch Isaac record mode",
                               "closes": []}
    assert state["last_launch"]["status"] == "failed"
    relaunch = sup.start()
    assert relaunch["status"] == "starting" and relaunch["reused"] is False
    assert len(fakes.spawned) == 2 and relaunch["failure"] is None
    assert relaunch["last_launch"].get("isaac_pid") is None


def test_a_launcher_that_exits_without_an_isaac_pid_fails_after_the_grace_period(
    tmp_path: Path,
) -> None:
    """No pid line and a dead launcher means the launcher failed; a missing log still waits."""
    fakes = Fakes([proc(9, DRIVER)])
    sup = supervisor(tmp_path, fakes)
    result = sup.start()
    fakes.now += isaac.LAUNCHER_GRACE_S + 1
    assert sup.state()["status"] == "starting"           # no log at all: nothing to judge yet
    Path(result["last_launch"]["log"]).write_text("Mode: record\n  Driver already running.\n")
    fakes.now += 1
    state = sup.state()
    assert state["failure"] == ("The launcher exited after 17 s without starting Isaac; "
                                "see the launch log")
    assert state["failure_detail"] == []
    assert state["button"]["label"] == "Relaunch Isaac record mode"


def test_failure_tail_keeps_error_lines_without_prefixes_and_tolerates_missing_logs(
    tmp_path: Path,
) -> None:
    """Only error, fatal, traceback and kill lines are kept, newest last, bounded in number."""
    assert isaac.failure_tail(None) == [] and isaac.failure_tail(tmp_path / "missing.log") == []
    log = tmp_path / "isaac.log"
    lines = [f"2026-10-17T03:03:20Z [{i},000ms] [Error] [omni.kit.app._impl] [py stderr]: e{i}"
             for i in range(20)]
    lines.append("2026-10-17T03:03:21Z [21,000ms] [Fatal] [carb] out of memory")
    lines.append("Killed")
    lines.append("2026-10-17T03:03:21Z [21,000ms] [Info] [carb] fine")
    log.write_text("\n".join(lines) + "\n")
    tail = isaac.failure_tail(log)
    assert len(tail) == isaac.FAILURE_TAIL_LINES
    assert tail[-3:] == ["e19", "out of memory", "Killed"]
    assert isaac.launch_facts(tmp_path / "nope.log") == {
        "isaac_pid": None, "isaac_log": None, "log_exists": False}

"""Supervise Isaac record mode from the simulation console: find, close, start and stop it.

The console never runs text from the page. The only commands are fixed: the teleop
stack launcher in record mode with SKIP_DRIVER=1 (so the arm driver is never started
by the page), git rev-parse on the teleop checkout, and nvidia-smi for GPU memory.
Only this user's Isaac Sim, Isaac Lab or Kit processes are ever closed; the arm driver
and everything else are never touched. The launcher path comes from the
SYNRIA_ISAAC_LAUNCHER environment variable or <workspace>/isaac_launcher.json, never
from the repository. Process listing, signalling, spawning, log reading and topic
rate measurement are injectable so the logic is tested with fakes only.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# SIGKILL has no Windows equivalent; the supervisor is Linux-only (it reads /proc), but
# the module must import and type-check on every platform the suite runs on.
_KILL = getattr(signal, "SIGKILL", signal.SIGTERM)

LAUNCHER_ENV = "SYNRIA_ISAAC_LAUNCHER"
LAUNCHER_FILE = "isaac_launcher.json"
LAUNCHER_NAME = "run_alicia_stack.sh"
STATE_FILE = "isaac_record_launch.json"
SIM_STATE_TOPIC = "/isaac_joint_states"
# The recorded trio; the sim's side view is a page preview and is not part of record-mode health.
SIM_TOPICS = {"state": SIM_STATE_TOPIC, "wrist": "/isaac/wrist_camera/image_raw",
              "front": "/isaac/overhead_camera/image_raw"}
# The scene publishes every pick object's world pose; a simulation take declares the task's
# object as its "object" goal channel and, for the place task, the mark as "target", so the
# dataset carries the goal the strategy of 2026-10-18 requires of every new policy.
SIM_OBJECT_POSE_TOPIC = "/isaac/pick_objects/{name}/pose"
SIM_TASK_OBJECTS = {"die_into_cup": "die", "pick_cup": "cup", "cup_pick_and_place": "cup",
                    "roll_and_dump": "cup", "cup_return": "cup"}
SIM_TASK_TARGETS = {"cup_pick_and_place": "place_target"}
RATE_WINDOW_S = 3.0
HEALTHY_RATE_HZ = 13.5          # 15 Hz requested; a skipped render frame now and then is tolerated
MAX_GAP_S = 0.5                 # a longer silence on any sim topic is not "publishing steadily"
RECORD_FLAGS = ("--record-cameras", "--publish-sim-topics")
TERMINATE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 240.0
LAUNCHER_GRACE_S = 15.0         # the launcher prints the Isaac pid within seconds or it failed
FAILURE_TAIL_LINES = 12
_ISAAC_PID = re.compile(r"Isaac Sim PID:\s*(\d+)")
_ISAAC_LOG = re.compile(r"Logs:\s*(\S+)")
_LOG_PREFIX = re.compile(r"^\S+ \[[\d,]+ms\] \[(\w+)\] \[[^\]]+\] (?:\[py stderr\]: )?")
DRIVER_PATTERN = "alicia_d_driver_node"
READ_ONLY_DRIVER_ARGS = ("ros2", "launch", "alicia_d_driver", "alicia_d_driver.launch.py",
                         "joint_commands_enabled:=false", "torque_off_on_start:=false")
READ_ONLY_DRIVER_COMMAND = " ".join(READ_ONLY_DRIVER_ARGS)
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
_SERIAL = re.compile(r"/dev/serial/by-id/\S+|/dev/tty\S+")


@dataclass(frozen=True)
class ProcessInfo:
    """What the supervisor needs to know about one process of this user."""

    pid: int
    cmdline: str
    started_utc: str
    exe: str = ""


@dataclass
class IsaacProcess:
    """One Isaac Sim, Isaac Lab or Kit process, classified for the panel."""

    pid: int
    kind: str
    started_utc: str
    summary: str
    record_mode: bool = False
    gpu_mb: int | None = None
    commit: str | None = None
    scene_loaded: bool = False
    topics_started: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Plain facts for the page."""
        return asdict(self)


def classify(process: ProcessInfo) -> str | None:
    """Name the Isaac family a process belongs to, or None when it is not one.

    The arm driver, shells, launch wrappers and this console are never Isaac.
    """
    line = process.cmdline
    lower = line.lower()
    if DRIVER_PATTERN in line or "synria_lerobot.console" in line:
        return None
    if lower.startswith(("/bin/bash", "bash ", "sh ", "/bin/sh", "timeout ")):
        return None
    if "alicia_d_live_scene.py" in line and ("kit/python" in line or "python" in line):
        return "teleop scene"
    if "isaaclab" in lower.replace("_", "").replace("-", ""):
        return "Isaac Lab"
    if "/kit/kit" in line or "isaacsim.exp" in line:
        return "Isaac Sim app"
    if "/isaacsim/" in line or "omniverse" in lower:
        return "Kit process"
    return None


def masked(cmdline: str, limit: int = 120) -> str:
    """A short command summary with any serial device path removed."""
    return _SERIAL.sub("<device>", cmdline)[:limit]


def read_proc_processes(uid: int | None = None) -> list[ProcessInfo]:
    """This user's processes from /proc with their command lines and start times."""
    # POSIX only, like /proc itself; the Windows stubs lack getuid and sysconf.
    uid = os.getuid() if uid is None else uid  # type: ignore[attr-defined]
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            btime = next(int(line.split()[1]) for line in handle if line.startswith("btime"))
        ticks = os.sysconf("SC_CLK_TCK")  # type: ignore[attr-defined]
    except (OSError, StopIteration, ValueError):
        btime, ticks = 0, 100
    out: list[ProcessInfo] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            started = datetime.fromtimestamp(btime + int(fields[19]) / ticks, timezone.utc)
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
            except OSError:
                exe = ""
        except (OSError, IndexError, ValueError):
            continue
        if cmdline:
            out.append(ProcessInfo(pid, cmdline, started.isoformat(timespec="seconds"), exe))
    return out


def read_gpu_memory() -> dict[int, int]:
    """GPU memory per pid in MB from nvidia-smi, a fixed command; empty when unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = int(parts[1])
    return usage


def read_log_markers(pid: int) -> dict[str, Any]:
    """Progress markers from the Kit log the process holds open: commit, scene, topics."""
    markers: dict[str, Any] = {"commit": None, "scene_loaded": False, "topics_started": False}
    try:
        names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return markers
    for name in names:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{name}")
        except OSError:
            continue
        if not target.endswith(".log") or "/kit/logs/" not in target:
            continue
        try:
            text = Path(target).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = re.search(r"SCENE COMMIT (\w+)", text)
        markers["commit"] = found.group(1) if found else markers["commit"]
        markers["scene_loaded"] = markers["scene_loaded"] or "TELEOP MODE:" in text
        markers["topics_started"] = markers["topics_started"] or "SIM TOPICS: publishing" in text
    return markers


def pid_suffix(log: Path) -> str:
    """A short distinct suffix for a launch log name that already exists."""
    suffix = 1
    while (log.parent / f"{log.stem}_{suffix}{log.suffix}").exists():
        suffix += 1
    return str(suffix)


def launch_facts(log: Path) -> dict[str, Any]:
    """The Isaac pid and Isaac log path the launcher printed, once it has; both None before."""
    facts: dict[str, Any] = {"isaac_pid": None, "isaac_log": None, "log_exists": False}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return facts
    facts["log_exists"] = True
    pid = _ISAAC_PID.search(text)
    path = _ISAAC_LOG.search(text)
    facts["isaac_pid"] = int(pid.group(1)) if pid else None
    facts["isaac_log"] = path.group(1) if path else None
    return facts


def failure_tail(log: Path | None, limit: int = FAILURE_TAIL_LINES) -> list[str]:
    """The last error lines of an Isaac log, prefixes stripped, so the panel can show why.

    Kit logs every Python traceback line as an [Error] entry; those lines, plus any
    fatal or crash markers, are kept. An unreadable or missing log gives an empty list.
    """
    if log is None:
        return []
    try:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    kept: list[str] = []
    for line in lines:
        match = _LOG_PREFIX.match(line)
        level = match.group(1).lower() if match else ""
        body = line[match.end():] if match else line
        if level in {"error", "fatal"} or "Traceback" in body or "Killed" in body:
            body = body.strip()
            if body:
                kept.append(body[:240])
    return kept[-limit:]


class SimTopicMonitor:
    """Background subscriber that measures the three sim topics over a sliding window.

    Runs in its own ROS context and thread only while a record-mode instance exists.
    snapshot() gives per-topic rates over RATE_WINDOW_S, the longest silence, and
    whether a full window has been observed yet; nothing is published.
    """

    def __init__(self, window_s: float = RATE_WINDOW_S) -> None:
        """Nothing runs until start()."""
        self.window_s = float(window_s)
        self._stamps: dict[str, deque] = {name: deque() for name in SIM_TOPICS}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at: float | None = None
        self.error: str | None = None

    def start(self) -> None:
        """Start the subscriber thread once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self.started_at = time.monotonic()
        for stamps in self._stamps.values():
            stamps.clear()
        self._thread = threading.Thread(target=self._run, name="synria-sim-topic-monitor",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and release the ROS context."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3.0)
        self._thread = None
        self.started_at = None

    def _record(self, name: str) -> None:
        """Keep the arrival time of one message, dropping stamps older than the window."""
        now = time.monotonic()
        with self._lock:
            stamps = self._stamps[name]
            stamps.append(now)
            while stamps and now - stamps[0] > self.window_s * 2:
                stamps.popleft()

    def _run(self) -> None:
        """Own a ROS context, subscribe to the three topics, spin until stopped."""
        try:
            import rclpy  # type: ignore[import-not-found]
            from rclpy.context import Context  # type: ignore[import-not-found]
            from rclpy.executors import SingleThreadedExecutor  # type: ignore[import-not-found]
            from sensor_msgs.msg import Image, JointState  # type: ignore[import-not-found]
        except ImportError as exc:
            self.error = f"ROS is unavailable in the console environment: {exc}"
            return
        context = Context()
        try:
            rclpy.init(context=context)
            node = rclpy.create_node("synria_console_sim_monitor", context=context)
            node.create_subscription(JointState, SIM_TOPICS["state"],
                                     lambda m: self._record("state"), 10)
            node.create_subscription(Image, SIM_TOPICS["wrist"], lambda m: self._record("wrist"), 5)
            node.create_subscription(Image, SIM_TOPICS["front"], lambda m: self._record("front"), 5)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            while not self._stop.is_set():
                executor.spin_once(timeout_sec=0.1)
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
        except Exception as exc:  # noqa: BLE001
            self.error = f"sim topic monitor failed: {exc}"
        finally:
            try:
                rclpy.shutdown(context=context)
            except Exception:  # noqa: BLE001
                pass

    def snapshot(self) -> dict[str, Any]:
        """Rates per topic over the window, the longest gap, and whether a full window passed."""
        now = time.monotonic()
        elapsed = 0.0 if self.started_at is None else now - self.started_at
        complete = elapsed >= self.window_s
        rates: dict[str, float | None] = {}
        gaps: dict[str, float | None] = {}
        with self._lock:
            for name, stamps in self._stamps.items():
                recent = [s for s in stamps if now - s <= self.window_s]
                span = self.window_s if complete else max(elapsed, 1e-6)
                rates[name] = len(recent) / span if (complete or recent) else None
                if recent:
                    pairs = zip(recent, recent[1:], strict=False)
                    gaps[name] = max([b - a for a, b in pairs] + [now - recent[-1]])
                else:
                    gaps[name] = None
        return {"rates": rates, "max_gap_s": gaps, "complete": complete,
                "elapsed_s": round(elapsed, 1), "window_s": self.window_s, "error": self.error}


def steady(snapshot: dict[str, Any] | None) -> bool:
    """All three sim topics at or above HEALTHY_RATE_HZ over a full window with no long silence."""
    if not snapshot or not snapshot.get("complete") or snapshot.get("error"):
        return False
    rates, gaps = snapshot.get("rates", {}), snapshot.get("max_gap_s", {})
    return all(
        rates.get(name) is not None and rates[name] >= HEALTHY_RATE_HZ
        and gaps.get(name) is not None and gaps[name] <= MAX_GAP_S
        for name in SIM_TOPICS
    )


def waiting_words(record_up: bool, scene_loaded: bool, topics_started: bool,
                  snapshot: dict[str, Any] | None, elapsed_s: float | None) -> str:
    """Why the instance is not yet Running, in operator words, with the time waited."""
    since = f", {int(elapsed_s)} s" if elapsed_s is not None else ""
    if not record_up:
        return "Waiting for the Isaac process" + since
    if not scene_loaded:
        return "Isaac is loading the scene" + since
    if not topics_started:
        return "Scene loaded, waiting for the sim topics" + since
    if snapshot and snapshot.get("error"):
        return f"Scene loaded; the console cannot measure the sim topics: {snapshot['error']}"
    if not snapshot or not snapshot.get("complete"):
        return "Scene loaded, measuring the sim topics" + since
    rates = snapshot.get("rates", {})
    low = [name for name in SIM_TOPICS if rates.get(name) is None or rates[name] < HEALTHY_RATE_HZ]
    if low:
        shown = ", ".join(
            f"{name} {rates[name]:.1f} Hz" if rates.get(name) is not None else f"{name} silent"
            for name in low
        )
        return f"Scene loaded, waiting for camera feeds and state at 15 Hz ({shown})" + since
    return "Scene loaded, sim topics paused" + since


def launcher_from_config(workspace: Path, environ: dict[str, str] | None = None) -> Path | None:
    """The teleop launcher path from the environment or the workspace file; None when unset."""
    source: Mapping[str, str] = os.environ if environ is None else environ
    value = source.get(LAUNCHER_ENV, "")
    if not value:
        path = workspace / LAUNCHER_FILE
        if path.is_file():
            try:
                value = str(json.loads(path.read_text(encoding="utf-8")).get("launcher", ""))
            except (OSError, ValueError):
                value = ""
    if not value:
        return None
    launcher = Path(value)
    if not launcher.is_absolute() or launcher.name != LAUNCHER_NAME or not launcher.is_file():
        raise ValueError(f"Isaac launcher must be an absolute path to an existing {LAUNCHER_NAME}")
    return launcher


def teleop_commit(launcher: Path, run: Callable[..., Any] = subprocess.run) -> str:
    """Short commit of the teleop checkout the launcher lives in, a fixed git command."""
    repository = launcher.resolve().parent.parent
    try:
        result = run(["git", "-C", str(repository), "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True, timeout=5, check=False)
        return (result.stdout or "").strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class IsaacSupervisor:
    """Find this user's Isaac processes, close them on request, start and stop record mode."""

    def __init__(
        self, workspace: Path, log_dir: Path, *, launcher: Path | None,
        list_processes: Callable[[], list[ProcessInfo]] = read_proc_processes,
        gpu_memory: Callable[[], dict[int, int]] = read_gpu_memory,
        markers: Callable[[int], dict[str, Any]] = read_log_markers,
        monitor: SimTopicMonitor | None = None,
        topic_rate: Callable[[], float | None] | None = None,
        send_signal: Callable[[int, int], None] = os.kill,
        spawn: Callable[[list[str], dict[str, str], Path, Path], int] | None = None,
        commit: Callable[[Path], str] = teleop_commit,
        clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Keep the fixed launcher and the injectable process, signal, spawn and probe helpers."""
        self.workspace = workspace
        self.log_dir = log_dir
        self.launcher = launcher
        self._list = list_processes
        self._gpu = gpu_memory
        self._markers = markers
        self._monitor = monitor if monitor is not None else SimTopicMonitor()
        self._rate = topic_rate
        self._signal = send_signal
        self._spawn = spawn or self._spawn_detached
        self._commit = commit
        self._clock = clock
        self._sleep = sleep

    # ---- observation ---------------------------------------------------------------------
    def scan(self) -> list[IsaacProcess]:
        """Every Isaac process of this user, with record-mode flags, log progress and GPU use."""
        gpu = self._gpu()
        found: list[IsaacProcess] = []
        for process in self._list():
            kind = classify(process)
            if kind is None:
                continue
            record = all(flag in process.cmdline for flag in RECORD_FLAGS)
            facts = self._markers(process.pid) if kind == "teleop scene" else {}
            found.append(IsaacProcess(
                pid=process.pid, kind=kind, started_utc=process.started_utc,
                summary=masked(process.cmdline), record_mode=record, gpu_mb=gpu.get(process.pid),
                commit=facts.get("commit"), scene_loaded=bool(facts.get("scene_loaded")),
                topics_started=bool(facts.get("topics_started")),
            ))
        return sorted(found, key=lambda item: item.pid)

    def driver(self) -> dict[str, Any]:
        """Whether the arm driver runs; the page may start the read-only form, never stop it."""
        for process in self._list():
            shell = process.cmdline.lower().startswith(("bash", "/bin/bash"))
            if DRIVER_PATTERN in process.cmdline and not shell:
                return {"running": True, "since_utc": process.started_utc,
                        "read_only_command": READ_ONLY_DRIVER_COMMAND, "can_start": False}
        return {"running": False, "since_utc": None, "read_only_command": READ_ONLY_DRIVER_COMMAND,
                "can_start": self.launcher is not None}

    def driver_argv(self) -> list[str]:
        """The one fixed driver command: the launch file's read-only form in the teleop environment.

        The ROS distribution and the teleop checkout's install overlay (found from the
        launcher path) are sourced so the driver package resolves; the port is auto-detected
        by the launch file, so no serial or device path appears anywhere.
        """
        if self.launcher is None:
            raise ValueError(
                "Isaac launcher path is not configured; the driver environment comes from it")
        overlay = self.launcher.resolve().parent.parent / "install" / "setup.bash"
        script = (f"source {ROS_SETUP} && {{ [ -f {overlay} ] && source {overlay}; }}; "
                  f"exec {READ_ONLY_DRIVER_COMMAND}")
        return ["bash", "-c", script]

    def start_driver(self) -> dict[str, Any]:
        """Start the read-only arm driver, detached, only when none is running."""
        if self.driver()["running"]:
            raise RuntimeError("An arm driver is already running; the page never restarts it")
        argv = self.driver_argv()
        if self.launcher is None:
            raise ValueError("Isaac launcher path is not configured")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log = self.log_dir / f"arm_driver_readonly_{stamp}.log"
        pid = self._spawn(argv, dict(os.environ), log, self.launcher.resolve().parent.parent)
        return {**self.state(), "driver_launch_pid": pid, "driver_log": str(log)}

    def _launch_record(self) -> dict[str, Any] | None:
        """The last launch this console started, if its record is still meaningful."""
        path = self.log_dir / STATE_FILE
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def state(self) -> dict[str, Any]:
        """Status, instances, progress, driver and the one button the page should show."""
        instances = self.scan()
        record = [item for item in instances if item.kind == "teleop scene" and item.record_mode]
        others = [item for item in instances if item not in record]
        launch = self._launch_record()
        current = self._commit(self.launcher) if self.launcher else "unknown"
        snapshot = self._topics_snapshot(bool(record))
        rate = (snapshot or {}).get("rates", {}).get("state")
        healthy = bool(record) and steady(snapshot)
        starting = False
        launch_age = None
        if launch and launch.get("status") in {"starting", "failed", "running"}:
            launch_age = self._clock() - float(launch.get("started_at", 0))
        if launch and launch.get("status") == "starting":
            if healthy:
                launch = {**launch, "status": "running"}
                self._write_launch(launch)
            else:
                launch = self._watch_launch(launch, launch_age or 0.0, bool(record))
                starting = launch.get("status") == "starting"
        elif launch and launch.get("status") == "failed" and healthy:
            launch = {**launch, "status": "running", "error": None}
            self._write_launch(launch)
        if starting:
            status, label = "starting", "Starting Isaac record mode"
        elif healthy and not others:
            status, label = "running", "Running in record mode"
        elif others:
            status, label = "other_running", "Another Isaac is running"
        elif record:
            status, label = "starting", "Isaac record mode is up, not yet publishing steadily"
        else:
            status, label = "not_running", "Not running"
        waiting = None
        if status == "starting":
            waiting = waiting_words(bool(record), any(item.scene_loaded for item in record),
                                    any(item.topics_started for item in record), snapshot,
                                    launch_age)
            label = waiting
        same_commit = bool(record) and all(item.commit in (None, current) for item in record)
        if status == "running" and not same_commit:
            label = "Running in record mode on an older teleop commit"
        failed = launch if launch and launch.get("status") == "failed" else None
        button: dict[str, Any]
        if status == "not_running" and failed:
            label = "Last launch failed"
            button = {"action": "start", "label": "Relaunch Isaac record mode", "closes": []}
        elif status == "not_running":
            button = {"action": "start", "label": "Start Isaac record mode", "closes": []}
        elif status == "other_running":
            button = {"action": "start", "label": "Close running Isaac and start record mode",
                      "closes": [item.as_dict() for item in instances]}
        elif status == "running" and not same_commit:
            button = {"action": "start", "label": "Restart Isaac on the current teleop commit",
                      "closes": [item.as_dict() for item in instances]}
        elif status == "starting":
            button = {"action": None, "label": "Starting, please wait", "closes": []}
        else:
            button = {"action": None, "label": "Record mode is up; reusing it", "closes": []}
        return {
            "available": self.launcher is not None, "status": status, "label": label,
            "launcher": str(self.launcher) if self.launcher else None, "teleop_commit": current,
            "instances": [item.as_dict() for item in instances],
            "progress": {"process_up": bool(record),
                         "scene_loaded": any(item.scene_loaded for item in record),
                         "topics_started": any(item.topics_started for item in record),
                         "state_rate_hz": rate, "healthy": healthy, "waiting": waiting,
                         "topics": snapshot},
            "driver": self.driver(), "button": button, "can_stop": bool(record),
            "last_launch": launch, "failure": failed.get("error") if failed else None,
            "failure_detail": list(failed.get("error_detail") or []) if failed else [],
            "launch_log": launch.get("log") if launch else None,
            "isaac_log": launch.get("isaac_log") if launch else None,
        }

    def _watch_launch(self, launch: dict[str, Any], age_s: float,
                      record_up: bool) -> dict[str, Any]:
        """Catch a launch that died instead of waiting the full timeout for it.

        The launcher prints the Isaac pid and the Isaac log path within seconds; both are
        kept in the launch record. The launch is failed at once when that Isaac process is
        gone with no record-mode instance in its place, when the launcher itself exited
        without ever printing a pid, or, as the last resort, when the start timeout
        passes. A failure carries the last error lines of the Isaac log so the panel can
        say why, and the button becomes Relaunch.
        """
        updated = dict(launch)
        log = launch.get("log")
        facts = launch_facts(Path(log)) if log else {"log_exists": False}
        if launch.get("isaac_pid") is None and facts.get("isaac_pid") is not None:
            updated["isaac_pid"] = facts["isaac_pid"]
            updated["isaac_log"] = facts.get("isaac_log")
        isaac_pid = updated.get("isaac_pid")
        launcher_pid = launch.get("launch_pid")
        error = None
        if record_up:
            error = None
        elif isaac_pid is not None and not self._alive(int(isaac_pid)):
            error = (f"Isaac process {isaac_pid} exited after {int(age_s)} s before reaching "
                     "record mode; see the Isaac log")
        elif (isaac_pid is None and facts.get("log_exists") and launcher_pid is not None
              and not self._alive(int(launcher_pid)) and age_s > LAUNCHER_GRACE_S):
            error = (f"The launcher exited after {int(age_s)} s without starting Isaac; "
                     "see the launch log")
        if error is None and age_s > START_TIMEOUT_S:
            error = (f"Isaac did not reach record mode in {int(START_TIMEOUT_S)} s; "
                     "see the launch log")
        if error is not None:
            detail = failure_tail(updated.get("isaac_log"))
            updated.update({"status": "failed", "error": error, "error_detail": detail})
        if updated != launch:
            self._write_launch(updated)
        return updated

    def _topics_snapshot(self, record_up: bool) -> dict[str, Any] | None:
        """Rates of the sim topics: the injected probe in tests, else the background monitor."""
        if self._rate is not None:
            value = self._rate()
            if value is None:
                return None
            return {"rates": {name: value for name in SIM_TOPICS},
                    "max_gap_s": {name: 0.07 for name in SIM_TOPICS}, "complete": True,
                    "elapsed_s": RATE_WINDOW_S, "window_s": RATE_WINDOW_S, "error": None}
        if not record_up:
            self._monitor.stop()
            return None
        self._monitor.start()
        return self._monitor.snapshot()

    # ---- actions -------------------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        """Close every other Isaac process of this user, then launch record mode detached."""
        if self.launcher is None:
            raise ValueError("Isaac launcher path is not configured; set "
                             f"{LAUNCHER_ENV} or {LAUNCHER_FILE} in the workspace")
        current = self.state()
        if current["status"] == "starting":
            raise RuntimeError("Isaac record mode is already starting; wait for it")
        if current["status"] == "running" and current["button"]["action"] is None:
            return {**current, "reused": True}
        closed = self._terminate([IsaacProcess(**item) for item in current["button"]["closes"]])
        started = self._clock()
        started_utc = datetime.fromtimestamp(started, timezone.utc)
        stamp = started_utc.strftime("%Y%m%d_%H%M%S")
        log = self.log_dir / f"isaac_record_{stamp}.log"
        if log.exists():
            # A relaunch within the same second must not read the previous launch's pid line.
            log = self.log_dir / f"isaac_record_{stamp}_{pid_suffix(log)}.log"
        env = {**os.environ, "SKIP_DRIVER": "1"}
        pid = self._spawn([str(self.launcher), "record"], env, log, self.launcher.parent.parent)
        self._write_launch({"status": "starting", "launch_pid": pid, "started_at": started,
                            "started_utc": started_utc.isoformat(timespec="seconds"),
                            "commit": self._commit(self.launcher), "log": str(log),
                            "closed": closed})
        return {**self.state(), "closed": closed, "launch_pid": pid, "reused": False}

    def stop(self) -> dict[str, Any]:
        """Close this user's teleop scene instances only; the arm driver is never touched."""
        scenes = [item for item in self.scan() if item.kind == "teleop scene"]
        closed = self._terminate(scenes)
        launch = self._launch_record()
        if launch:
            self._write_launch({**launch, "status": "stopped"})
        return {**self.state(), "closed": closed}

    def _terminate(self, targets: list[IsaacProcess]) -> list[dict[str, Any]]:
        """SIGTERM each target, wait, then SIGKILL whatever is still alive; report what happened."""
        closed: list[dict[str, Any]] = []
        for item in targets:
            if item.kind not in {"teleop scene", "Isaac Sim app", "Isaac Lab", "Kit process"}:
                continue
            try:
                self._signal(item.pid, signal.SIGTERM)
            except ProcessLookupError:
                closed.append({**item.as_dict(), "result": "already gone"})
                continue
            closed.append({**item.as_dict(), "result": "terminated"})
        deadline = self._clock() + TERMINATE_TIMEOUT_S
        pending = [row for row in closed if row["result"] == "terminated"]
        while pending and self._clock() < deadline:
            self._sleep(0.5)
            pending = [row for row in pending if self._alive(row["pid"])]
        for row in pending:
            try:
                self._signal(row["pid"], _KILL)
                row["result"] = "killed after timeout"
            except ProcessLookupError:
                row["result"] = "terminated"
        return closed

    def _alive(self, pid: int) -> bool:
        """Whether a pid still exists, through the same signalling helper (signal 0)."""
        try:
            self._signal(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _write_launch(self, record: dict[str, Any]) -> None:
        """Persist the launch record so a console restart keeps supervising the same launch."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / STATE_FILE
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _spawn_detached(argv: list[str], env: dict[str, str], log: Path, cwd: Path) -> int:
        """Start the fixed launcher in its own session with its output in the log file."""
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as handle:
            process = subprocess.Popen(argv, env=env, cwd=str(cwd), stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        return process.pid


def sim_goal_channels(task_id: str) -> list[str]:
    """The recorder's ``NAME=TOPIC`` goal channels for one registry task in the scene.

    Every task declares its object; the place task also declares the mark. A task the
    table does not know declares nothing, so an unknown task never records a guess.
    """
    channels = []
    obj = SIM_TASK_OBJECTS.get(task_id)
    if obj is not None:
        channels.append("object=" + SIM_OBJECT_POSE_TOPIC.format(name=obj))
    target = SIM_TASK_TARGETS.get(task_id)
    if target is not None:
        channels.append("target=" + SIM_OBJECT_POSE_TOPIC.format(name=target))
    return channels

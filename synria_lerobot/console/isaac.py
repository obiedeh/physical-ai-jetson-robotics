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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

LAUNCHER_ENV = "SYNRIA_ISAAC_LAUNCHER"
LAUNCHER_FILE = "isaac_launcher.json"
LAUNCHER_NAME = "run_alicia_stack.sh"
STATE_FILE = "isaac_record_launch.json"
SIM_STATE_TOPIC = "/isaac_joint_states"
RECORD_FLAGS = ("--record-cameras", "--publish-sim-topics")
TERMINATE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 240.0
HEALTHY_RATE_HZ = 14.0
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
    uid = os.getuid() if uid is None else uid
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            btime = next(int(line.split()[1]) for line in handle if line.startswith("btime"))
        ticks = os.sysconf("SC_CLK_TCK")
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


def measure_topic_rate(topic: str = SIM_STATE_TOPIC, window_s: float = 1.0) -> float | None:
    """Messages per second on a sim topic over a short window, or None when ROS is unavailable."""
    try:
        import rclpy  # type: ignore[import-not-found]
        from rclpy.context import Context  # type: ignore[import-not-found]
        from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
    except ImportError:
        return None
    context = Context()
    count = 0

    def on_message(message: Any) -> None:
        """Count one message."""
        nonlocal count
        count += 1

    try:
        rclpy.init(context=context)
        node = rclpy.create_node("synria_console_isaac_probe", context=context)
        node.create_subscription(JointState, topic, on_message, 10)
        deadline = time.monotonic() + window_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05, context=context)
        node.destroy_node()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            rclpy.shutdown(context=context)
        except Exception:  # noqa: BLE001
            pass
    return count / window_s


def launcher_from_config(workspace: Path, environ: dict[str, str] | None = None) -> Path | None:
    """The teleop launcher path from the environment or the workspace file; None when unset."""
    environ = os.environ if environ is None else environ
    value = environ.get(LAUNCHER_ENV, "")
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
        topic_rate: Callable[[], float | None] = measure_topic_rate,
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
        """Whether the arm driver is running; the page can start only the read-only form, never stop it."""
        for process in self._list():
            if DRIVER_PATTERN in process.cmdline and not process.cmdline.lower().startswith(("bash", "/bin/bash")):
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
            raise ValueError("Isaac launcher path is not configured; the driver environment comes from it")
        overlay = self.launcher.resolve().parent.parent / "install" / "setup.bash"
        script = (f"source {ROS_SETUP} && {{ [ -f {overlay} ] && source {overlay}; }}; "
                  f"exec {READ_ONLY_DRIVER_COMMAND}")
        return ["bash", "-c", script]

    def start_driver(self) -> dict[str, Any]:
        """Start the read-only arm driver, detached, only when none is running."""
        if self.driver()["running"]:
            raise RuntimeError("An arm driver is already running; the page never restarts it")
        log = self.log_dir / f"arm_driver_readonly_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        pid = self._spawn(self.driver_argv(), dict(os.environ), log, self.launcher.resolve().parent.parent)
        return {**self.state(), "driver_launch_pid": pid, "driver_log": str(log)}

    def _launch_record(self) -> dict[str, Any] | None:
        """The last launch this console started, if its record is still meaningful."""
        path = self.log_dir / STATE_FILE
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def state(self) -> dict[str, Any]:
        """Status, instances, progress, driver and the one button the page should show."""
        instances = self.scan()
        record = [item for item in instances if item.kind == "teleop scene" and item.record_mode]
        others = [item for item in instances if item not in record]
        launch = self._launch_record()
        current = self._commit(self.launcher) if self.launcher else "unknown"
        rate = self._rate() if record else None
        healthy = bool(record) and rate is not None and rate >= HEALTHY_RATE_HZ
        starting = False
        if launch and launch.get("status") == "starting":
            age = self._clock() - float(launch.get("started_at", 0))
            if healthy:
                launch = {**launch, "status": "running"}
                self._write_launch(launch)
            elif age > START_TIMEOUT_S:
                launch = {**launch, "status": "failed", "error": "Isaac did not reach record mode in time"}
                self._write_launch(launch)
            else:
                starting = True
        if starting:
            status, label = "starting", "Starting Isaac record mode"
        elif healthy and not others:
            status, label = "running", "Running in record mode"
        elif others:
            status, label = "other_running", "Another Isaac is running"
        elif record:
            status, label = "starting", "Isaac record mode is up but not yet publishing at 15 Hz"
        else:
            status, label = "not_running", "Not running"
        same_commit = bool(record) and all(item.commit in (None, current) for item in record)
        if status == "running" and not same_commit:
            label = "Running in record mode on an older teleop commit"
        if status == "not_running":
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
        failed = launch if launch and launch.get("status") == "failed" else None
        return {
            "available": self.launcher is not None, "status": status, "label": label,
            "launcher": str(self.launcher) if self.launcher else None, "teleop_commit": current,
            "instances": [item.as_dict() for item in instances],
            "progress": {"process_up": bool(record), "scene_loaded": any(item.scene_loaded for item in record),
                         "topics_started": any(item.topics_started for item in record),
                         "state_rate_hz": rate, "healthy": healthy},
            "driver": self.driver(), "button": button, "can_stop": bool(record),
            "last_launch": launch, "failure": failed.get("error") if failed else None,
            "launch_log": launch.get("log") if launch else None,
        }

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
        log = self.log_dir / f"isaac_record_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        env = {**os.environ, "SKIP_DRIVER": "1"}
        pid = self._spawn([str(self.launcher), "record"], env, log, self.launcher.parent.parent)
        self._write_launch({"status": "starting", "launch_pid": pid, "started_at": self._clock(),
                            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "commit": self._commit(self.launcher), "log": str(log), "closed": closed})
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
                self._signal(row["pid"], signal.SIGKILL)
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

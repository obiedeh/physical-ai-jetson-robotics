"""Own console sessions and launch a local browser without controlling the robot."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, TextIO

from synria_lerobot.console.catalog import Catalog
from synria_lerobot.console.controller import RecordingController
from synria_lerobot.console.playback import EpisodePlayback, confined_file, encode_rgb
from synria_lerobot.console.readiness import (
    CONFIRMATIONS,
    ReadinessReport,
    inspect_samples,
    require_distinct_cameras,
    runtime_dependencies,
)
from synria_lerobot.console.server import make_server
from synria_lerobot.console.timing import TimingSession
from synria_lerobot.recorder import (
    PhysicalEpisodeRecorder,
    RecordedPhysicalEpisode,
    build_recording_session,
    recording_config_from_args,
)
from synria_lerobot.task_planner import (
    ENVIRONMENT_VARIABLE,
    EXTERNAL_SOURCE,
    ExternalPlanner,
    chain_check,
    describe_rules,
    plan_goal,
    slug,
    validate_breakdown,
)
from synria_lerobot.task_registry import (
    LIBRARY_SNAPSHOT_VERSION,
    TaskDefinition,
    is_library_task_id,
    library_snapshot_fields,
    load_task_registry,
    load_task_source,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY / "config" / "synria_tasks.json"
GUIDE_DOCUMENTS = {
    "runbook": "docs/ludo_flagship/D1_OPERATOR_RUNBOOK.md",
    "protocol": "docs/ludo_flagship/D1_DATASET_PROTOCOL.md",
    "decisions": "docs/ludo_flagship/DECISIONS.md",
    "safety": "reports/synria/phase0a_safety_recovery.md",
    "first-motion": "reports/synria/first_safe_motion.md",
    "limits": "config/synria_limits.yaml",
}


def _joint_state_source(topic: str) -> Any:
    """The recorder's own read-only joint-state subscriber, for timing on the bench."""
    from synria_lerobot.recorder import RosJointStateSource

    return RosJointStateSource(topic, node_name="synria_console_timer",
                               first_sample_timeout_s=10.0)


def library_task_definition(task: dict[str, Any]) -> TaskDefinition:
    """Build the exact task snapshot the recorder binds from a library task row."""
    window = task.get("episode_window") or {"min_episode_s": None, "max_episode_s": None}
    return TaskDefinition(
        task["task_id"], task["instruction"], task["success_rule"],
        tuple(task["scene_before_take"]), window.get("min_episode_s"), window.get("max_episode_s"),
    )


def _git_sha() -> str:
    """Capture the local software revision without reading any robot or remote service."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def list_cameras(directory: Path = Path("/dev/v4l/by-id")) -> dict[str, Any]:
    """List stable camera identifiers without opening devices or querying their drivers."""
    if not directory.is_dir():
        return {"cameras": [], "message": "No stable /dev/v4l/by-id camera entries are available."}
    return {"cameras": sorted(str(item) for item in directory.iterdir()), "message": ""}


def list_serial_connections(directory: Path = Path("/dev/serial/by-id")) -> dict[str, Any]:
    """List OS connection identities without opening ports or identifying a physical arm."""
    devices = sorted(str(item) for item in directory.iterdir()) if directory.is_dir() else []
    return {"connections": devices, "message": (
        "USB connection IDs only; association with the follower is unverified. "
        "These are not manufacturer arm serial numbers."
        if devices else "No stable USB serial connection IDs are available on this host."
    )}


def local_follower_serials(workspace: Path) -> list[str]:
    """Read private, operator-registered manufacturer serial choices outside the repository."""
    path = workspace / "follower_serial_options.json"
    if path.is_symlink() or not path.exists():
        return []
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    return sorted({value.strip() for value in values
                   if isinstance(value, str) and 0 < len(value.strip()) <= 128},
                  key=str.casefold)


class WorkspaceLock:
    """Hold an exclusive workspace marker and never silently remove another instance's lock."""

    def __init__(self, workspace: Path) -> None:
        """Acquire the console-only sibling lock before opening its catalog."""
        workspace.mkdir(parents=True, exist_ok=True)
        self.path = workspace / ".console.lock"
        self.token = secrets.token_hex(24)
        try:
            with self.path.open("x", encoding="utf-8") as output:
                output.write(self.token)
        except FileExistsError as error:
            raise RuntimeError("another console owns this workspace; preserve its lock") from error

    def close(self) -> None:
        """Release only this instance's unchanged marker, retaining unexplained locks."""
        if self.path.is_symlink() or self.path.read_text(encoding="utf-8") != self.token:
            raise RuntimeError("workspace lock changed; refusing to remove it")
        self.path.unlink()


class ConsoleService:
    """Coordinate catalog facts and one recorder context; browser clients own no capture state."""

    def __init__(
        self, workspace: Path, *, demo: bool = False, registry: Path = DEFAULT_REGISTRY,
        min_free_bytes: int = 1_000_000_000,
        builder: Callable[..., AbstractContextManager[PhysicalEpisodeRecorder]] = (
            build_recording_session
        ),
        controller_factory: Callable[..., RecordingController] = RecordingController,
        playback: EpisodePlayback | None = None, clock: Callable[[], float] = time.monotonic,
        planner: ExternalPlanner | None = None,
        timing_source_factory: Callable[[str], Any] | None = None,
        timing_poll_s: float = 0.02,
    ) -> None:
        """Inject fake builders or clocks while retaining the production construction path."""
        resolved = workspace.resolve()
        broad = {Path(resolved.anchor), Path.home().resolve(), Path.cwd().resolve()}
        if (workspace.is_symlink() or resolved in broad or resolved == REPOSITORY
                or REPOSITORY in resolved.parents):
            raise ValueError("console workspace must be outside the repository and not a symlink")
        if type(min_free_bytes) is not int or min_free_bytes <= 0:
            raise ValueError("free disk floor must be a positive integer")
        if not registry.is_file() or not (REPOSITORY / "config/synria_limits.yaml").is_file():
            raise ValueError(
                "console requires the repository config files; use an editable repository "
                "installation in the dedicated recording environment"
            )
        self.workspace = resolved
        self.demo = demo
        self.registry = registry
        self.min_free_bytes = min_free_bytes
        self.builder = builder
        self.controller_factory = controller_factory
        self.clock = clock
        self.playback = playback or EpisodePlayback()
        self.planner = planner
        self._timing_source_factory = timing_source_factory or _joint_state_source
        self._timing_poll_s = timing_poll_s
        self._timing: TimingSession | None = None
        self._lock = threading.RLock()
        self._workspace_lock = WorkspaceLock(resolved)
        catalog = None
        try:
            catalog = Catalog(resolved, repository=REPOSITORY)
            self.catalog = catalog
            self.catalog.reconcile()
        except BaseException:
            try:
                if catalog is not None:
                    catalog.close()
            finally:
                self._workspace_lock.close()
            raise
        self.controller: RecordingController | None = None
        self._context: AbstractContextManager[PhysicalEpisodeRecorder] | None = None
        self._recorder: PhysicalEpisodeRecorder | None = None
        self._active: dict[str, Any] | None = None
        self._run_id: int | None = None
        self._smoke_episodes: list[dict[str, Any]] = []
        self._preflight: dict[str, Any] = {}
        self.readiness = ReadinessReport(clock, demo=demo)
        self._closed = False
        self.quit_requested = threading.Event()

    def _settings(
        self, payload: dict[str, Any], *, smoke: bool = False, diagnostic: bool = False,
    ) -> dict[str, Any]:
        """Validate shared settings; only no-episode diagnostics may omit recording metadata.

        Diagnostic scope is selected internally, never accepted from form input.
        Session creation and disposable episode recording still require identity
        and starting power documentation as well as all source safety checks.
        """
        allowed = {
            "name", "task_id", "gripper_type", "state_source", "follower_topic",
            "action_source", "action_lookahead_steps", "fps", "image_width", "image_height",
            "wrist_camera", "front_camera", "guard_command_topic", "state_has_velocity",
            "leader_topic", "state_startup_timeout_s", "operator", "follower_serial",
            "leader_serial", "scene", "power_state_start", "target_episodes", "notes",
            "follower_usb_id",
        }
        if set(payload) - allowed:
            raise ValueError("unknown session settings; paths and task windows are server-owned")
        required = ["name", "task_id", "gripper_type", "fps", "operator", "scene"]
        if not self.demo:
            required += ["state_source", "wrist_camera", "front_camera"]
            if not diagnostic:
                required += ["follower_serial", "power_state_start"]
        missing = [key for key in required
                   if key not in payload or str(payload[key]).strip() == ""]
        if missing:
            raise ValueError("Required fields are missing: " + ", ".join(missing)
                             + ". A USB connection ID does not replace the manufacturer serial.")
        if type(payload.get("state_has_velocity", False)) is not bool:
            raise ValueError("velocity mode must be a boolean")
        defaults: dict[str, Any] = {
            "follower_topic": "/joint_states", "action_source": "next_state",
            "action_lookahead_steps": 1, "image_width": 224, "image_height": 224,
            "guard_command_topic": [], "state_has_velocity": False,
            "leader_topic": "/leader/joint_states", "state_startup_timeout_s": 10.0,
        }
        settings = {**defaults, **{key: value for key, value in payload.items() if key in {
            *defaults, "task_id", "gripper_type", "fps", "state_source", "wrist_camera",
            "front_camera",
        }}}
        if self.demo:
            settings.update(state_source="synthetic_demo", wrist_camera="synthetic:wrist",
                            front_camera="synthetic:front")
        elif payload.get("follower_usb_id"):
            connection = payload["follower_usb_id"]
            if not isinstance(connection, str) or connection not in list_serial_connections()[
                "connections"
            ]:
                raise ValueError("Selected USB connection ID is unavailable; refresh the list")
            settings["follower_usb_id"] = connection
        timeout = settings["state_startup_timeout_s"]
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("state startup wait must be positive and finite")
        if not isinstance(settings["guard_command_topic"], list):
            raise ValueError("guarded topics must be a list of absolute topic names")
        self._config(settings, "local/console-validation", self.workspace / "validation", smoke)
        return settings

    def _task_source(self, task_id: Any, version: int | None = None) -> Path:
        """The file the recorder reads a task from: the registry, or a library snapshot.

        A snapshot is written once per task version under the workspace and never edited,
        so a dataset keeps pointing at the exact version it was recorded against.
        """
        if not is_library_task_id(task_id):
            return self.registry
        task = self.catalog.library_task(task_id, version)
        goal = self.catalog.goal(task["goal"])
        path = self.workspace / "library-snapshots" / f"{task_id}-v{task['version']}.json"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": LIBRARY_SNAPSHOT_VERSION, "task_source": "library",
                "goal_id": goal["id"], "goal_title": goal["title"],
                "library_version": task["version"], "chain_position": task["chain_position"],
                "task": library_task_definition(task).as_dict(),
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
            temporary.replace(path)
        return path

    def _args(
        self, settings: dict[str, Any], repo_id: str, dataset: Path, smoke: bool,
        task_source: Path | None = None,
    ) -> argparse.Namespace:
        """Restore one canonical CLI-compatible namespace from immutable session settings."""
        return argparse.Namespace(
            **settings, task_registry=task_source or self._task_source(settings.get("task_id")),
            repo_id=repo_id, dataset_path=dataset, smoke=smoke,
        )

    def _config(
        self, settings: dict[str, Any], repo_id: str, dataset: Path, smoke: bool,
    ) -> Any:
        """Use the recorder's configuration factory rather than copying its contract checks."""
        return recording_config_from_args(
            self._args(settings, repo_id, dataset, smoke), synthetic_demo=self.demo,
        )

    def check_readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an explicitly authorized disposable connection probe, never an episode.

        This uses the exact recorder builder and its graph/rate checks. An unset
        task window may be diagnosed under disposable purpose but never becomes
        qualifying-ready. Live session ownership and recording remain untouched.
        """
        from synria_lerobot.quality_gates import load_limits

        if self.controller is not None:
            raise RuntimeError("Close the current sources before checking new-session connections")
        if set(payload) != {"settings", "confirmations"} or not all(
            isinstance(payload[name], dict) for name in payload
        ):
            raise ValueError("readiness requires settings and explicit operator confirmations")
        report = self.readiness
        report.begin()
        report.form_settings = dict(payload["settings"])
        try:
            self.catalog.record_event(None, "readiness_started", {
                "operator": payload["settings"].get("operator", ""),
                "confirmations": payload["confirmations"], "demo": self.demo,
            })
            confirmations = payload["confirmations"]
            confirmed = self.demo or (
                set(confirmations) == set(CONFIRMATIONS)
                and all(value is True for value in confirmations.values())
            )
            report.check("operator_safety", confirmed,
                         "Synthetic only" if self.demo else
                         ("Operator confirmed the runbook safety and physical connection checklist"
                          if confirmed else "All physical safety confirmations are required"))
            if not confirmed:
                raise ValueError("Confirm the physical safety checklist before opening sources")
            missing_metadata = [] if self.demo else [
                name for name in ("follower_serial", "power_state_start")
                if not str(payload["settings"].get(name, "")).strip()
            ]
            report.check("recording_metadata", not missing_metadata,
                         "Required before recording: " + ", ".join(missing_metadata)
                         + ". Connection checks can continue; unknown values are not inferred."
                         if missing_metadata else "Recording identity and power fields supplied")
            try:
                settings = self._settings(payload["settings"], smoke=not self.demo,
                                          diagnostic=True)
            except (ValueError, RuntimeError) as error:
                report.check("form_configuration", False, str(error))
                raise
            report.check("form_configuration", True, "Shared recorder settings validated")
            try:
                self._config(settings, "local/readiness", self.workspace / "unused", False)
                report.check("qualifying_configuration", True, "Task contract configured")
            except (ValueError, RuntimeError) as error:
                report.check("qualifying_configuration", False, str(error))
            limits = load_limits(REPOSITORY / "config" / "synria_limits.yaml")
            report.check("limits_verification", self.demo or limits.verified,
                         "Synthetic only" if self.demo else
                         ("Operator verification recorded" if limits.verified else
                          "limits unverified by operator; qualifying collection is not ready"))
            free = shutil.disk_usage(self.workspace).free
            report.check("disk_space", free >= self.min_free_bytes,
                         "Available recording workspace space", remaining=free,
                         minimum=self.min_free_bytes)
            if free < self.min_free_bytes:
                raise ValueError("Free disk space is below the recording floor")
            if not self.demo:
                available = list_cameras()["cameras"]
                if any(settings[key] not in available for key in ("wrist_camera", "front_camera")):
                    raise ValueError("Select two available stable camera IDs; refresh mapping")
                require_distinct_cameras(settings["wrist_camera"], settings["front_camera"])
            report.check("camera_mapping", True, "Distinct identities selected; no auto-mapping")
            dependencies = runtime_dependencies(demo=self.demo)
            report.check("runtime_dependencies", all(dependencies.values()),
                         "Recording environment dependency availability", details=dependencies)
            if not all(dependencies.values()):
                raise ValueError("Recording dependencies are missing; use the runbook environment "
                                 "with robot-learning, vision, ffmpeg and ROS for physical sources")
            with tempfile.TemporaryDirectory(prefix=".readiness-", dir=self.workspace) as temporary:
                args = self._args(settings, "local/readiness", Path(temporary) / "dataset",
                                  not self.demo)
                report.launching("source_startup", "Opening read-only sources; no driver launch")
                with self.builder(args, synthetic_demo=self.demo, clock=self.clock,
                                  preflight=report.observe) as recorder:
                    report.check("source_startup", True, "Shared source preflight completed")
                    inspect_samples(recorder, report, limits)
            report.check("cleanup", True, "Diagnostic sources closed; no episode started or saved")
        except Exception as error:
            if report.checks.get("source_startup", {}).get("status") == "launching":
                report.check("source_startup", False, str(error))
            report.check("diagnostic", False, str(error))
        finally:
            report.finish()
            self.catalog.record_event(None, "readiness_completed", report.snapshot())
        return report.snapshot()

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate the shared physical contract before committing a named session mirror."""
        settings = self._settings(payload)
        source = self._task_source(settings["task_id"])
        task = load_task_source(source)[settings["task_id"]]
        values = {key: payload.get(key, "") for key in (
            "name", "operator", "scene", "follower_serial", "leader_serial",
            "power_state_start", "notes",
        )}
        values.update(
            source_kind="synthetic_demo" if self.demo else "physical",
            recording_purpose="synthetic_demo" if self.demo else "qualifying",
            task_id=task.task_id, task_definition_sha256=task.sha256,
            settings=settings, target_episodes=int(payload.get("target_episodes", 100)),
            git_sha=_git_sha(), host=socket.gethostname(),
            repo_id="local/synria-console-" + secrets.token_hex(8),
        )
        binding = library_snapshot_fields(source)
        if binding:
            values.update(task_source="library", task_snapshot_path=str(source),
                          goal_id=binding["goal_id"], library_version=binding["library_version"],
                          chain_position=binding["chain_position"])
            # A version with a session bound to it is frozen; edits create the next version.
            self.catalog.freeze_library_task(task.task_id, binding["library_version"])
        return self.catalog.create_session(values)

    def _audit(self, kind: str, detail: dict[str, Any]) -> None:
        """Link controller events to the active session, or explicitly transient smoke context."""
        active = self._active
        session_id = active["id"] if active and active["id"] != "smoke" else None
        normalized = kind.replace(" ", "_")
        if normalized == "episode_saved" and session_id is not None:
            # The successful catalog callback already committed this event with its row.
            return
        self.catalog.record_event(session_id, normalized, detail, detail.get("episode_index"))

    def _saved(self, episode: RecordedPhysicalEpisode) -> dict[str, Any]:
        """Evaluate committed sidecars with existing gates before inserting catalog facts."""
        from synria_lerobot.quality_gates import (
            GateConfig,
            episode_quality_record,
            evaluate_episode,
            load_limits,
        )

        assert self._recorder is not None and self._active is not None
        record = episode_quality_record(episode, fps=self._recorder.config.fps)
        report = evaluate_episode(
            record, load_limits(REPOSITORY / "config" / "synria_limits.yaml"),
            GateConfig(min_episode_s=episode.min_episode_s, max_episode_s=episode.max_episode_s),
        ).as_dict()
        if self._active["id"] == "smoke":
            row = {"episode_index": episode.episode_index,
                   "operator_label": episode.operator_label.value,
                   "duration_s": episode.duration_s, "frame_count": len(episode.frames),
                   "achieved_sample_rate_hz": episode.achieved_sample_rate_hz,
                   "gate_passed": report["passed"], "failed_gates": report["failed_gates"]}
            self._smoke_episodes.append(row)
            return row
        return self.catalog.record_saved_episode(
            self._active["id"], episode.episode_index, capture_run=self._run_id,
            gate_report=report,
        )

    def open_session(self, session_id: str) -> dict[str, Any]:
        """Restore exact persisted settings and run the CLI's complete source preflight."""
        with self._lock:
            if self.controller is not None:
                raise RuntimeError("close the current capture run before opening another session")
            session = self.catalog.get_session(session_id)
            if session["status"] in {"trashed", "purged"}:
                raise RuntimeError("restore this session before opening it")
            if (session["source_kind"] == "synthetic_demo") != self.demo:
                raise ValueError("session source kind differs from this launch mode")
            source = (Path(session["task_snapshot_path"]) if session.get("task_snapshot_path")
                      else self.registry)
            task = load_task_source(source)[session["task_id"]]
            if task.sha256 != session["task_definition_sha256"]:
                raise ValueError("stored task definition differs from its task source")
            self.catalog.reconcile(session_id)
            refreshed = self.catalog.get_session(session_id)
            if refreshed.get("recovery_blocked"):
                raise RuntimeError(refreshed.get("recovery_error", "recording recovery blocked"))
            return self._open(session, smoke=False)

    def _open(self, session: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
        """Own source lifetime and preserve every actual preflight result, including refusals."""
        settings = session["settings"]
        source = (Path(session["task_snapshot_path"]) if session.get("task_snapshot_path")
                  else None)
        args = self._args(settings, session["repo_id"], Path(session["dataset_path"]), smoke,
                          task_source=source)
        check_names = (
            "configuration", "state_source", "state_rate", "state_rate_evidence", "velocity",
            "command_publishers", "dataset_lock_and_recovery", "wrist_camera", "front_camera",
        )
        self._preflight = {"status": "running", "checks": {
            name: {"status": "not run"} for name in check_names
        }}

        def observe_preflight(name: str, result: dict[str, Any]) -> None:
            """Retain builder-reported outcomes without inferring success for unrun checks."""
            self._preflight["checks"][name] = result

        context = self.builder(
            args, synthetic_demo=self.demo, clock=self.clock, preflight=observe_preflight,
        )
        try:
            recorder = context.__enter__()
        except BaseException as error:
            self._preflight.update(status="refused", error=str(error))
            self.catalog.record_event(
                None if smoke else session["id"], "preflight_refused", self._preflight,
            )
            if not smoke:
                refused_run = self.catalog.start_capture_run(
                    session["id"], git_sha=_git_sha(), host=socket.gethostname(),
                    preflight=self._preflight,
                )
                self.catalog.end_capture_run(refused_run, f"preflight refused: {error}")
            raise
        self._context, self._recorder, self._active = context, recorder, session
        if smoke:
            session["dataset_path"] = str(recorder.config.dataset_path)
        try:
            measurement = recorder.config.state_rate_measurement
            self._preflight.update(status="passed")
            free = shutil.disk_usage(self.workspace).free
            self._preflight["checks"]["free_disk_bytes"] = {
                "remaining": free, "minimum": self.min_free_bytes,
                "passed": free >= self.min_free_bytes,
            }
            if not smoke:
                self.catalog.set_busy(session["id"], True)
                self.catalog.update_session(session["id"], status="active")
                self._run_id = self.catalog.start_capture_run(
                    session["id"], git_sha=_git_sha(), host=socket.gethostname(),
                    preflight=self._preflight,
                    measured_state_rate=measurement.rate_hz if measurement else None,
                    callback_count=measurement.message_count if measurement else None,
                    maximum_gap=measurement.max_callback_gap_s if measurement else None,
                )
            else:
                self._audit("capture run started", {"recording_purpose": "disposable_smoke"})
            self.controller = self.controller_factory(
                recorder, emit_event=self._audit, on_saved=self._saved,
                min_free_bytes=self.min_free_bytes,
                disk_free=lambda: shutil.disk_usage(self.workspace).free,
            )
            self.controller.start_worker()
        except BaseException as error:
            self.close_session(reason=f"capture startup failed: {error}", force=True)
            raise
        return self.state()

    def close_session(self, *, reason: str = "operator closed", force: bool = False) -> None:
        """Finalize sources while refusing voluntary shutdown with memory-only pending frames."""
        with self._lock:
            if self.controller is None and self._context is None:
                return
            controller = self.controller
            if controller is not None and not force and controller.snapshot()["state"] not in {
                "idle", "closed",
            }:
                raise RuntimeError("pending frames must be saved or discarded before closing")
            errors: list[str] = []

            def attempt(action: Callable[[], Any]) -> None:
                """Attempt every independent cleanup while retaining all refusal messages."""
                try:
                    action()
                except BaseException as error:
                    errors.append(str(error))

            if controller is not None:
                attempt(lambda: controller.shutdown(reason, force=force))
            context = self._context
            if context is not None:
                attempt(lambda: context.__exit__(None, None, None))
            if self._run_id is not None:
                run_id = self._run_id
                exit_reason = reason + ("; cleanup errors: " + "; ".join(errors) if errors else "")
                attempt(lambda: self.catalog.end_capture_run(run_id, exit_reason))
            if self._active is not None and self._active["id"] == "smoke":
                dataset = Path(self._active["dataset_path"])
                disposed = not dataset.exists() and not dataset.is_symlink()
                if not disposed:
                    errors.append("disposable smoke dataset remains after cleanup")
                attempt(lambda: self.catalog.record_event(
                    None, "capture_run_ended", {
                        "recording_purpose": "disposable_smoke", "exit_reason": reason,
                        "disposal_succeeded": disposed, "cleanup_errors": list(errors),
                    },
                ))
            if self._active is not None and self._active["id"] != "smoke":
                identity = self._active["id"]
                attempt(lambda: self.catalog.set_busy(identity, False))
                attempt(lambda: self.catalog.update_session(identity, status="closed"))
                if errors:
                    attempt(lambda: self.catalog.record_event(
                        identity, "capture_cleanup_failed", {"errors": list(errors)},
                    ))
            self.controller = None
            self._context = self._recorder = self._active = None
            self._run_id = None
            self._smoke_episodes = []
            if errors:
                raise RuntimeError("capture cleanup failed: " + "; ".join(errors))

    def state(self) -> dict[str, Any]:
        """Return enough server-owned state to reconstruct the page after any reload."""
        status = self.controller.snapshot() if self.controller is not None else {"state": "closed"}
        active = self._active
        if active is not None and active["id"] != "smoke":
            active = self.catalog.get_session(active["id"])
        return {
            "demo": self.demo, "active_session": active, "capture": status,
            "preflight": self._preflight,
            "free_disk_bytes": shutil.disk_usage(self.workspace).free,
            "min_free_bytes": self.min_free_bytes,
        }

    def _session(self, session_id: str) -> dict[str, Any]:
        """Resolve an opaque session identity, never a client-selected disk location."""
        if session_id == "smoke" and self._active is not None and self._active["id"] == "smoke":
            return self._active
        session = self.catalog.get_session(session_id)
        if session["status"] == "purged":
            raise FileNotFoundError("session was purged; only its tombstone remains")
        return session

    def task_choices(self) -> list[dict[str, Any]]:
        """Published tasks first, then library tasks grouped under their goal, for the dropdown."""
        choices = [{**task.as_dict(), "task_source": "registry", "group": "Built-in tasks",
                    "display_name": task.task_id.replace("_", " ").capitalize()}
                   for task in load_task_registry(self.registry).values()]
        for goal in self.catalog.goals():
            for task in goal["tasks"]:
                definition = library_task_definition(task)
                choices.append({
                    **definition.as_dict(), "task_source": "library", "group": goal["title"],
                    "display_name": f"{goal['title']}: {task['chain_position']} {task['name']}",
                    "goal_id": goal["id"], "goal_title": goal["title"],
                    "library_version": task["version"], "chain_position": task["chain_position"],
                    "kind": task.get("kind"), "frozen": task["frozen"],
                    "start_state": task.get("start_state"), "end_state": task.get("end_state"),
                    "suggested_takes": task.get("suggested_takes"),
                })
        return choices

    def plan_goal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Break a sentence down with the configured planner; nothing is saved here."""
        sentence = payload.get("sentence")
        if type(sentence) is not str or len(sentence) > 400:
            raise ValueError("type one sentence of up to 400 characters")
        answers = payload.get("answers", {})
        if type(answers) is not dict or any(
            type(key) is not str or type(value) is not str for key, value in answers.items()
        ):
            raise ValueError("answers must map question keys to short text")
        result = plan_goal(sentence, external=self.planner, answers=answers)
        response = result.as_dict()
        if result.breakdown is not None:
            response["warnings"] = [w.as_dict() for w in chain_check(result.breakdown)]
            response["proposed"] = self._proposed_ids(result.breakdown.goal.title,
                                                      len(result.breakdown.sub_tasks))
        return response

    def _proposed_ids(self, title: str, count: int) -> dict[str, Any]:
        """Stable ids for a goal and its sub-tasks, derived from the goal title."""
        goal_id = f"lib_{slug(title)}"
        return {"goal_id": goal_id,
                "task_ids": [f"{goal_id}_{position}" for position in range(1, count + 1)]}

    def save_goal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate an edited breakdown and store it as library task versions."""
        if set(payload) != {"breakdown", "planner_source"}:
            raise ValueError("saving needs the breakdown and the planner source")
        breakdown = validate_breakdown(payload["breakdown"])
        source = payload["planner_source"]
        if source not in {"built-in planner", EXTERNAL_SOURCE, "operator"}:
            raise ValueError("unknown planner source")
        proposed = self._proposed_ids(breakdown.goal.title, len(breakdown.sub_tasks))
        goal = {"id": proposed["goal_id"], "title": breakdown.goal.title,
                "sentence": breakdown.goal.sentence, "rationale": breakdown.goal.rationale}
        sub_tasks = []
        for task_id, step in zip(proposed["task_ids"], breakdown.sub_tasks, strict=True):
            sub_tasks.append({"task_id": task_id, **step.as_dict(),
                              "episode_window": {"min_episode_s": None, "max_episode_s": None}})
        warnings = [w.as_dict() for w in chain_check(breakdown)]
        saved = self.catalog.save_goal(goal, sub_tasks, planner_source=source)
        return {"goal": saved, "warnings": warnings}

    def time_library_task(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Store an operator-confirmed take window with the timing runs that proposed it."""
        if not is_library_task_id(task_id):
            raise ValueError("only library tasks are timed here; registry timing is reviewed")
        runs = payload.get("runs", [])
        if type(runs) is not list or len(runs) > 20:
            raise ValueError("timing runs must be a short list")
        return self.catalog.set_library_window(
            task_id, payload.get("min_episode_s"), payload.get("max_episode_s"), runs=runs,
        )

    def start_timing(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Time three to five moves from the arm feed; never while anything else owns the feed."""
        if self.demo:
            raise ValueError("Time it is not available in demo mode; it listens to the real arm")
        if not is_library_task_id(task_id):
            raise ValueError("only library tasks are timed here; registry timing is reviewed")
        self.catalog.library_task(task_id)
        topic = payload.get("topic", "/joint_states")
        if type(topic) is not str or not topic.startswith("/"):
            raise ValueError("the arm feed topic must be an absolute topic name")
        with self._lock:
            if self.controller is not None or self.readiness.running:
                raise RuntimeError("pause the session or wait for the setup check before timing")
            if self._timing is not None and self._timing.running:
                raise RuntimeError("a timing pass is already running")
            session = TimingSession(task_id, lambda: self._timing_source_factory(topic),
                                    poll_s=self._timing_poll_s)
            self._timing = session
            session.start()
            self.catalog.record_event(None, "timing_started", {"task_id": task_id, "topic": topic})
            return session.snapshot()

    def timing_status(self, task_id: str) -> dict[str, Any]:
        """Progress of the timing pass for this task, or nothing running."""
        session = self._timing
        if session is None or session.task_id != task_id:
            return {"task_id": task_id, "running": False, "phase": "not started", "runs": [],
                    "runs_wanted": "3 to 5", "proposed_window": None, "rest_learned": False,
                    "error": None}
        return session.snapshot()

    def stop_timing(self, task_id: str) -> dict[str, Any]:
        """Stop listening and release the arm feed subscription."""
        session = self._timing
        if session is not None and session.task_id == task_id:
            session.stop()
            self.catalog.record_event(None, "timing_stopped",
                                      {"task_id": task_id, "runs": len(session.timer.runs)})
        return self.timing_status(task_id)

    def confirm_timing(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Store the window the operator confirmed, with the runs that proposed it."""
        session = self._timing
        runs = [run.as_dict() for run in session.timer.runs] if (
            session is not None and session.task_id == task_id
        ) else []
        if session is not None and session.task_id == task_id:
            session.stop()
        return self.time_library_task(task_id, {**payload, "runs": runs})

    def export_goal(self, goal_id: str) -> dict[str, Any]:
        """A registry-shaped JSON fragment of a goal's tasks for a reviewed commit."""
        goal = self.catalog.goal(goal_id)
        return {
            "note": "Library tasks exported for review. The published registry in "
                    "config/synria_tasks.json stays the source for evidence claims until a "
                    "reviewed commit adds these.",
            "goal": {"id": goal["id"], "title": goal["title"], "sentence": goal["sentence"]},
            "tasks": [library_task_definition(task).as_dict() for task in goal["tasks"]],
        }

    def setup_suggestions(self) -> dict[str, Any]:
        """Combine host facts, runbook candidates and prior declarations without probing devices."""
        previous: dict[str, Any] = {}
        reusable = {"operator", "gripper_type", "state_source", "wrist_camera", "front_camera",
                    "scene", "follower_topic", "leader_topic", "action_source"}
        for event in reversed(self.catalog.events()):
            detail = event.get("detail", {})
            if event["kind"] == "readiness_completed" and detail.get("demo") is self.demo:
                previous = {key: value for key, value in detail.get("form_settings", {}).items()
                            if key in reusable and value not in (None, "")}
                break
        return {
            "host": socket.gethostname(), "account": getpass.getuser(),
            "free_disk_bytes": shutil.disk_usage(self.workspace).free,
            "previous_operator_entries": previous,
            "known_follower_serials": (
                [] if self.demo else sorted(
                    set(self.catalog.known_follower_serials())
                    | set(local_follower_serials(self.workspace)), key=str.casefold,
                )
            ),
            "recommended": {"fps": 15, "image_width": 224, "image_height": 224,
                            "action_lookahead_steps": 1, "state_startup_timeout_s": 10,
                            "target_episodes": 100, "action_source": "next_state",
                            "follower_topic": "/joint_states",
                            "leader_topic": "/leader/joint_states"},
            "notice": "Recommended values are runbook candidates, not measurements. "
                      "Previous operator entries need review for this session. "
                      "The ROS state topic does not expose the manufacturer serial; saved "
                      "physical serials are offered for reuse, and first use requires reading "
                      "the follower label. Power and safety confirmations are never inferred.",
        }

    def read(self, route: list[str], query: dict[str, list[str]]) -> Any:
        """Expose catalog and exact playback facts without treating GET as a command."""
        if route == ["operator-guide"]:
            return {
                "environment": {
                    "python": sys.executable, "host": socket.gethostname(),
                    "ros_domain": os.environ.get("ROS_DOMAIN_ID", "unset (default domain)"),
                    "ros_distribution": os.environ.get("ROS_DISTRO", "not sourced"),
                    "middleware": os.environ.get("RMW_IMPLEMENTATION", "distribution default"),
                },
                "documents": GUIDE_DOCUMENTS,
                "limits_verified": self.demo or self._operator_limits_verified(),
                "demo": self.demo,
            }
        if len(route) == 2 and route[0] == "help" and route[1] in GUIDE_DOCUMENTS:
            path = GUIDE_DOCUMENTS[route[1]]
            return {"path": path, "text": (REPOSITORY / path).read_text(encoding="utf-8")}
        if route == ["software"]:
            try:
                dependencies = runtime_dependencies(demo=self.demo)
            except (ImportError, ValueError, OSError) as error:
                return {"status": "not_ready", "label": "Not ready", "dependencies": {},
                        "message": f"Software availability check failed: {error}"}
            ready = all(dependencies.values())
            return {
                "status": "ready" if ready else "not_ready",
                "label": "Ready" if ready else "Not ready", "dependencies": dependencies,
                "message": (
                    "Recording libraries are discoverable. This does not verify hardware "
                    "connections, library compatibility or recording readiness."
                    if ready else "Missing: " + ", ".join(
                        name for name, available in dependencies.items() if not available
                    ) + ". Restart with the RTX console launcher in the recording environment. "
                    "No driver, controller, bridge or teleoperation will be launched."
                ),
            }
        if route == ["state"]:
            return self.state()
        if route == ["readiness"]:
            return self.readiness.snapshot()
        if route == ["setup-suggestions"]:
            return self.setup_suggestions()
        if route == ["tasks"]:
            return self.task_choices()
        if route == ["library"]:
            return {
                "goals": self.catalog.goals(), "rules": describe_rules(),
                "planner": {"external_configured": self.planner is not None,
                            "source": EXTERNAL_SOURCE if self.planner else "built-in planner"},
            }
        if len(route) == 2 and route[0] == "library-export":
            return self.export_goal(route[1])
        if len(route) == 3 and route[0] == "library" and route[2] == "time":
            return self.timing_status(route[1])
        if route == ["cameras"]:
            return list_cameras() if not self.demo else {
                "cameras": ["synthetic:wrist", "synthetic:front"], "message": "Synthetic demo only",
            }
        if route == ["serial-connections"]:
            return list_serial_connections() if not self.demo else {
                "connections": [], "message": "Synthetic demo has no physical USB connection.",
            }
        if route == ["sessions"]:
            return self.catalog.sessions(
                search=query.get("search", [""])[0], include_trashed=True,
            )
        if len(route) == 2 and route[0] == "session":
            return self._session(route[1])
        if len(route) == 2 and route[0] == "episodes":
            return self._smoke_episodes if route[1] == "smoke" else self.catalog.episodes(route[1])
        if len(route) == 3 and route[0] == "provenance":
            session = self._session(route[1])
            root = Path(session["dataset_path"])
            result = {"contract": json.loads(
                confined_file(root, "physical_contract.json").read_text(encoding="utf-8"),
            )}
            for key, filename in (
                ("episode", "physical_episode_metadata.jsonl"),
                ("capture", "physical_capture_provenance.jsonl"),
            ):
                records = [json.loads(line) for line in confined_file(root, filename).read_text(
                    encoding="utf-8",
                ).splitlines()]
                result[key] = next(row for row in records if row["episode_index"] == int(route[2]))
            return result
        if len(route) == 4 and route[0] == "frame":
            session = self._session(route[1])
            return self.playback.frame(
                Path(session["dataset_path"]), session["repo_id"], int(route[2]), int(route[3]),
            )[0]
        raise KeyError("unknown read-only API route")

    def _operator_limits_verified(self) -> bool:
        """Expose declared limits verification without changing the file or certifying hardware."""
        from synria_lerobot.quality_gates import load_limits

        return load_limits(REPOSITORY / "config" / "synria_limits.yaml").verified

    def media(self, route: list[str], query: dict[str, list[str]]) -> bytes:
        """Encode live previews, the last setup-check samples, or stored episode artifacts."""
        if len(route) == 2 and route[0] == "preview" and route[1] in {"wrist", "front"}:
            controller = self.controller
            if controller is None:
                raise RuntimeError("no session is open")
            frame = controller.preview_frame(route[1])
            return encode_rgb(frame.data)
        if len(route) == 2 and route[0] == "readiness" and route[1] in {"wrist", "front"}:
            # The sample the last setup check already read; no camera is opened here.
            return encode_rgb(self.readiness.thumbnails[route[1]])
        if len(route) == 3 and route[0] == "still":
            session = self._session(route[1])
            return self.playback.final_still(Path(session["dataset_path"]), int(route[2]))
        if len(route) == 5 and route[0] == "frame" and route[4] in {"wrist", "front"}:
            session = self._session(route[1])
            return self.playback.frame(
                Path(session["dataset_path"]), session["repo_id"], int(route[2]), int(route[3]),
            )[1][route[4]]
        raise KeyError("unknown media route")

    def mutate(self, route: list[str], payload: dict[str, Any]) -> Any:
        """Apply authenticated operator requests while retaining all recorder and catalog guards."""
        with self._lock:
            if route == ["readiness"]:
                return self.check_readiness(payload)
            if route == ["library", "plan"]:
                return self.plan_goal(payload)
            if route == ["library", "save"]:
                return self.save_goal(payload)
            if len(route) == 3 and route[0] == "library" and route[2] == "window":
                return self.time_library_task(route[1], payload)
            if len(route) == 4 and route[0] == "library" and route[2] == "time":
                if route[3] == "start":
                    return self.start_timing(route[1], payload)
                if route[3] == "stop":
                    return self.stop_timing(route[1])
                if route[3] == "confirm":
                    return self.confirm_timing(route[1], payload)
            if route == ["sessions"]:
                return self.create_session(payload)
            if route == ["reindex"]:
                if self.controller is not None:
                    raise RuntimeError("close the capture run before reindexing")
                return self.catalog.reconcile()
            if route == ["smoke"]:
                if self.demo:
                    raise ValueError("synthetic demo and disposable physical smoke are distinct")
                if self.controller is not None:
                    raise RuntimeError("close the current session before disposable smoke")
                settings = self._settings(payload, smoke=True)
                session = {"id": "smoke", "name": f"Disposable smoke: {payload['name']}",
                           "settings": settings, "repo_id": "local/synria-console-smoke",
                           "dataset_path": str(self.workspace / "unused-smoke-root"),
                           "source_kind": "synthetic_demo" if self.demo else "physical",
                           "recording_purpose": "disposable_smoke"}
                self._open(session, smoke=True)
                assert self.controller is not None
                self.controller.command("start")
                return self.state()
            if route == ["shutdown"]:
                self.close_session(reason="operator quit")
                self.quit_requested.set()
                return {"closed": True}
            if len(route) == 2 and route[0] == "record":
                if self.controller is None:
                    raise RuntimeError("open a session before recording")
                if route[1] == "countdown":
                    return self.controller.command("countdown", seconds=payload.get("seconds", 3))
                return self.controller.command(route[1], reason=payload.get("reason", ""))
            if len(route) == 3 and route[0] == "session":
                session_id, action = route[1:]
                if action == "open":
                    return self.open_session(session_id)
                if action == "close":
                    if self._active is not None and self._active["id"] == session_id:
                        self.close_session()
                    elif session_id != "smoke":
                        self.catalog.update_session(session_id, status="closed")
                    return self.state()
                if action == "finish":
                    # Finished is a label, not a lock: Open reverses it so takes can be added.
                    if session_id == "smoke":
                        raise ValueError("a practice run is never finished as a session")
                    if self._active is not None and self._active["id"] == session_id:
                        self.close_session()
                    self.catalog.update_session(session_id, status="finished")
                    return self.state()
                if action == "rename":
                    return self.catalog.update_session(session_id, name=payload["name"])
                if action == "trash":
                    return self.catalog.trash_session(session_id, **payload)
                if action == "restore":
                    return self.catalog.restore_session(session_id)
                if action == "purge":
                    return self.catalog.purge_session(session_id, **payload)
            if len(route) == 4 and route[0] == "episode":
                session_id, index, action = route[1:]
                if action in {"exclude", "restore"}:
                    return self.catalog.exclude_episode(
                        session_id, int(index), action=action,
                        reason_code=payload.get("reason_code", "capture_fault"),
                        note=payload.get("note", ""), operator=payload["operator"],
                    )
                if action == "note":
                    self.catalog.note_episode(session_id, int(index), payload["note"])
                    return {"saved": True}
            raise KeyError("unknown mutation route")

    def close(self, *, force: bool = False, reason: str = "shutdown") -> None:
        """Close all owned resources; only signals may intentionally lose pending memory frames."""
        if self._closed:
            return
        cleanup_error: BaseException | None = None
        if self._timing is not None:
            self._timing.stop()
        try:
            self.close_session(reason=reason, force=force)
        except BaseException as error:
            if self.controller is not None:
                raise
            cleanup_error = error
        try:
            self.catalog.close()
        finally:
            self._workspace_lock.close()
            self._closed = True
        if cleanup_error is not None:
            raise cleanup_error


def announce_url(url: str, *, demo: bool, stream: TextIO | None = None) -> None:
    """Print the launch URL and flush it so a redirected log shows the token immediately.

    A background launch with stdout sent to a file otherwise keeps the only copy of
    the per-launch token in a block buffer until the process exits.
    """
    output = sys.stdout if stream is None else stream
    print(f"Synria Teleop Console {'SYNTHETIC DEMO' if demo else 'read-only'}: {url}",
          file=output, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Launch the packaged console, opening the browser with its secret only in the URL fragment."""
    parser = argparse.ArgumentParser(description="Read-only Synria Teleop Console")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--min-free-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args(argv)
    if args.demo:
        try:
            import lerobot.datasets.lerobot_dataset  # noqa: F401
        except ImportError as error:
            raise SystemExit("Demo mode needs the robot-learning extra in Python 3.12.") from error
    if args.host not in {"127.0.0.1", "localhost"}:
        if not args.allow_remote:
            parser.error("non-loopback bind requires --allow-remote")
        print("WARNING: remote HTTP exposes private recordings; use a trusted isolated network.",
              flush=True)
    planner = ExternalPlanner.from_environment(os.environ.get(ENVIRONMENT_VARIABLE))
    service = ConsoleService(args.workspace, demo=args.demo, min_free_bytes=args.min_free_bytes,
                             planner=planner)
    if planner is not None:
        print("External task planner command is configured; the built-in planner answers "
              "when it fails.", flush=True)
    if args.reindex:
        try:
            print(json.dumps(service.catalog.reconcile(), indent=2))
        finally:
            service.close()
        return 0
    token = secrets.token_urlsafe(32)
    try:
        server = make_server(
            service, token=token, host=args.host, port=args.port, allow_remote=args.allow_remote,
        )
    except BaseException:
        service.close()
        raise
    server.timeout = 0.25
    interrupted: list[str] = []

    def request_shutdown(signum: int, frame: Any) -> None:
        """Ask the server loop to perform orderly forced cleanup outside the signal handler."""
        interrupted.append(signal.Signals(signum).name)
        service.quit_requested.set()

    previous: dict[int, Any] = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_shutdown)
        url = f"http://{args.host}:{server.server_address[1]}/#token={token}"
        announce_url(url, demo=args.demo)
        if not args.no_browser:
            webbrowser.open(url)
        while not service.quit_requested.is_set():
            server.handle_request()
    finally:
        server.server_close()
        try:
            service.close(
                force=bool(interrupted), reason=interrupted[-1] if interrupted else "shutdown",
            )
        finally:
            for restored_signal, previous_handler in previous.items():
                signal.signal(restored_signal, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

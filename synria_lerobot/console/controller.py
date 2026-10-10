"""Single-owner recording state with queued operator commands and independent preview.

The worker is the only caller of recorder mutations. HTTP request threads read
immutable status snapshots or the sources' latest-value caches; closing a page
cannot interrupt capture or remove its mandatory deadline.
"""

from __future__ import annotations

import copy
import math
import queue
import shutil
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from synria_lerobot.physical_contract import ImageFrame
from synria_lerobot.recorder import (
    PhysicalEpisodeRecorder,
    RecordedPhysicalEpisode,
    RecorderState,
    capture_due,
)


@dataclass
class _Request:
    """Keep one caller's command and result together across the owner-thread boundary."""

    name: str
    arguments: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    cancelled: bool = False


class RecordingController:
    """Own a recorder lifecycle without adding any command-capable robot interface."""

    COMMANDS = frozenset(
        {
            "start",
            "stop",
            "success",
            "failure",
            "retry",
            "discard",
            "countdown",
            "cancel_countdown",
        }
    )

    def __init__(
        self,
        recorder: PhysicalEpisodeRecorder,
        *,
        emit_event: Callable[[str, dict[str, Any]], None],
        on_saved: Callable[[RecordedPhysicalEpisode], dict[str, Any]],
        min_free_bytes: int,
        disk_free: Callable[[], int] | None = None,
    ) -> None:
        """Require a positive disk floor and callbacks that follow committed dataset writes."""
        if type(min_free_bytes) is not int or min_free_bytes <= 0:
            raise ValueError("minimum free disk space must be a positive integer")
        self.recorder = recorder
        self._emit_event = emit_event
        self._on_saved = on_saved
        self.min_free_bytes = min_free_bytes
        self._disk_free = disk_free or (
            lambda: shutil.disk_usage(recorder.config.dataset_path.parent).free
        )
        self._requests: queue.Queue[_Request] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._next_sample = recorder.clock()
        self._countdown_deadline: float | None = None
        self._last_episode: dict[str, Any] | None = None
        self._error: str | None = None
        self._catalog_error: str | None = None
        self._saving = False
        self._counters = {"saved": 0, "success": 0, "failure": 0, "discarded": 0, "save_failed": 0}
        self._trace: deque[dict[str, Any]] = deque(maxlen=120)
        self._publish()

    def _audit(self, kind: str, detail: dict[str, Any]) -> None:
        """Expose catalog failure separately from irreversible dataset transaction outcomes."""
        try:
            self._emit_event(kind, detail)
        except Exception as error:
            self._catalog_error = f"audit write failed: {error}; reindex and inspect the catalog"

    def _free_bytes(self) -> int:
        """Treat invalid disk observations as refusal, never as unlimited free space."""
        value = self._disk_free()
        if type(value) is not int or value < 0:
            raise ValueError("free disk space observation must be a nonnegative integer")
        return value

    def _start(self) -> None:
        """Apply disk and existing recorder guards before starting capture."""
        if self._catalog_error is not None:
            raise RuntimeError("catalog needs reconciliation; close, reindex and reopen first")
        if self.recorder.config.smoke and self._counters["saved"]:
            raise RuntimeError("disposable smoke is complete; review it and close the session")
        free = self._free_bytes()
        if free < self.min_free_bytes:
            raise RuntimeError(
                f"free disk space {free} bytes is below the {self.min_free_bytes} byte floor"
            )
        self.recorder.start()
        self._next_sample = self.recorder.clock()
        self._error = None
        self._audit("episode started", {"episode_index": self.recorder.next_episode_index})

    def _save(self, command: str) -> None:
        """Commit through the recorder first; catalog errors cannot make save retryable."""
        if command != "retry" and self.recorder.pending_label is not None:
            raise RuntimeError("the failed save already has a label; use retry or discard")
        self._saving = True
        self._publish()
        if command == "retry":
            self._audit("episode retried", {"episode_index": self.recorder.next_episode_index})
        try:
            episode = {
                "success": self.recorder.mark_success,
                "failure": self.recorder.mark_failure,
                "retry": self.recorder.retry,
            }[command]()
        except Exception as error:
            self._counters["save_failed"] += 1
            self._error = f"Episode save failed: {error}. Frames retained; retry or discard."
            self._audit(
                "episode save failed",
                {
                    "episode_index": self.recorder.next_episode_index,
                    "message": str(error),
                    "pending_frame_count": self.recorder.pending_frame_count,
                    "recovery_blocked": self.recorder.writer.recovery_blocked,
                },
            )
            raise
        finally:
            self._saving = False
        self._error = None
        self._counters["saved"] += 1
        self._counters[episode.operator_label.value] += 1
        self._last_episode = {
            "episode_index": episode.episode_index,
            "operator_label": episode.operator_label.value,
            "duration_s": episode.duration_s,
            "frame_count": len(episode.frames),
            "achieved_sample_rate_hz": episode.achieved_sample_rate_hz,
            "requested_fps": self.recorder.config.fps,
        }
        try:
            self._last_episode.update(self._on_saved(episode))
        except Exception as error:
            self._catalog_error = (
                f"Episode {episode.episode_index} was saved; catalog update failed: {error}. "
                "Reindex from disk; do not retry this saved episode."
            )
        self._audit("episode saved", dict(self._last_episode))

    def _execute(self, name: str, arguments: dict[str, Any]) -> None:
        """Apply one owner-thread transition using the recorder's existing legal states."""
        if self._stopped.is_set():
            raise RuntimeError("recording controller is closed")
        if name == "_shutdown":
            reason = str(arguments.get("reason", "shutdown"))
            pending = self.recorder.state is not RecorderState.IDLE
            if pending and not arguments.get("force", False):
                raise RuntimeError(
                    "pending frames must be saved or explicitly discarded before quit"
                )
            if pending:
                self._audit(
                    "pending frames lost",
                    {
                        "reason": reason,
                        "episode_index": self.recorder.next_episode_index,
                        "pending_frame_count": self.recorder.pending_frame_count,
                        "pending_label": self.recorder.pending_label,
                    },
                )
            self._countdown_deadline = None
            try:
                self.recorder.close()
            finally:
                self._stopped.set()
                self._audit(
                    "controller shutdown", {"reason": reason, "pending_frames_lost": pending}
                )
            return
        if name == "countdown":
            if self._catalog_error is not None:
                raise RuntimeError("catalog needs reconciliation; close, reindex and reopen first")
            seconds = arguments.get("seconds", 3)
            if (
                type(seconds) not in (int, float)
                or not math.isfinite(seconds)
                or not 0 < seconds <= 60
            ):
                raise ValueError("start countdown must be finite and within 0 to 60 seconds")
            if (
                self.recorder.state is not RecorderState.IDLE
                or self.recorder.writer.recovery_blocked
            ):
                raise RuntimeError("countdown requires an idle recorder without blocked recovery")
            if self._countdown_deadline is not None:
                raise RuntimeError("a start countdown is already pending")
            self._countdown_deadline = self.recorder.clock() + seconds
            self._audit("episode countdown started", {"seconds": seconds})
        elif name == "cancel_countdown":
            if self._countdown_deadline is None:
                raise RuntimeError("no start countdown is pending")
            self._countdown_deadline = None
            self._audit("episode countdown cancelled", {})
        elif name == "start":
            if self._countdown_deadline is not None:
                raise RuntimeError("cancel the pending countdown before starting directly")
            self._start()
        elif name == "stop":
            self.recorder.stop()
            self._audit(
                "episode stopped",
                {"episode_index": self.recorder.next_episode_index, "reason": "operator"},
            )
        elif name in {"success", "failure", "retry"}:
            self._save(name)
        elif name == "discard":
            reason = arguments.get("reason", "")
            if type(reason) is not str or not reason.strip():
                raise ValueError("discard requires an explicit reason and confirmation")
            if self.recorder.state is RecorderState.IDLE:
                raise RuntimeError("no pending capture to discard")
            count = self.recorder.pending_frame_count
            self.recorder.discard()
            self._counters["discarded"] += 1
            self._error = None
            self._audit(
                "episode discarded",
                {
                    "reason": reason,
                    "pending_frame_count": count,
                    "episode_index": self.recorder.next_episode_index,
                },
            )
        else:
            raise ValueError("unknown recording command")

    def _publish(self) -> None:
        """Copy small owner metadata without holding the lock across capture, save or encoding."""
        state = self.recorder.state
        blocked = self.recorder.writer.recovery_blocked
        closed = self._stopped.is_set()
        has_frames = self.recorder.pending_frame_count > 0
        countdown = self._countdown_deadline
        smoke_complete = self.recorder.config.smoke and bool(self._counters["saved"])
        snapshot = {
            "state": "closed" if self._stopped.is_set() else state.value,
            "episode_index": self.recorder.next_episode_index,
            "elapsed_s": self.recorder.elapsed_s,
            "pending_frame_count": self.recorder.pending_frame_count,
            "pending_label": self.recorder.pending_label,
            "min_episode_s": self.recorder.config.min_episode_s,
            "max_episode_s": self.recorder.config.max_episode_s,
            "requested_fps": self.recorder.config.fps,
            "remaining_s": max(0.0, self.recorder.config.hard_cap_s - self.recorder.elapsed_s),
            "countdown_s": None
            if countdown is None
            else max(0.0, countdown - self.recorder.clock()),
            "recovery_blocked": blocked,
            "saving": self._saving,
            "error": self._error,
            "catalog_error": self._catalog_error,
            "last_episode": copy.deepcopy(self._last_episode),
            "counters": dict(self._counters),
            "recording_purpose": self.recorder.config.contract.recording_purpose,
            "controls": {
                "start": (
                    not closed
                    and state is RecorderState.IDLE
                    and not blocked
                    and countdown is None
                    and not smoke_complete
                    and self._catalog_error is None
                ),
                "stop": not closed and state is RecorderState.RECORDING,
                "success": (
                    not closed
                    and has_frames
                    and state is RecorderState.STOPPED
                    and not blocked
                    and not self._saving
                    and self.recorder.pending_label is None
                ),
                "failure": (
                    not closed
                    and has_frames
                    and state is RecorderState.STOPPED
                    and not blocked
                    and not self._saving
                    and self.recorder.pending_label is None
                ),
                "retry": not closed
                and has_frames
                and state is RecorderState.STOPPED
                and self.recorder.pending_label is not None
                and not blocked
                and not self._saving,
                "discard": (
                    not closed
                    and state is not RecorderState.IDLE
                    and not blocked
                    and not self._saving
                ),
            },
        }
        with self._snapshot_lock:
            self._snapshot = snapshot

    def tick(self) -> None:
        """Advance owner timing once; tests can drive a fake clock without sleeping."""
        if self._thread is not None and threading.current_thread() is not self._thread:
            raise RuntimeError("only the capture worker may advance recording time")
        if self._stopped.is_set():
            return
        now = self.recorder.clock()
        if self._countdown_deadline is not None and now >= self._countdown_deadline:
            self._countdown_deadline = None
            try:
                self._start()
            except Exception as error:
                self._error = str(error)
                self._audit("preflight refused", {"message": str(error), "check": "episode start"})
        if self.recorder.state is RecorderState.RECORDING:
            try:
                self._next_sample, capped = capture_due(self.recorder, self._next_sample, now=now)
                if capped:
                    self._audit(
                        "episode stopped",
                        {"reason": "hard cap", "episode_index": self.recorder.next_episode_index},
                    )
            except Exception as error:
                self._error = f"Capture failed: {error}; pending frames retained."
                if self.recorder.state is RecorderState.RECORDING:
                    self.recorder.stop()
                self._audit(
                    "episode stopped",
                    {
                        "reason": "capture fault",
                        "message": str(error),
                        "episode_index": self.recorder.next_episode_index,
                    },
                )
        self._publish()

    def _perform(self, request: _Request) -> None:
        """Complete exactly one queued transition and publish its state even after refusal."""
        if request.cancelled:
            request.done.set()
            return
        try:
            self._execute(request.name, request.arguments)
        except BaseException as error:
            request.error = error
            self._error = str(error)
            if request.name == "start":
                self._audit("preflight refused", {"message": str(error), "check": "episode start"})
        finally:
            self._publish()
            with self._snapshot_lock:
                request.result = copy.deepcopy(self._snapshot)
            request.done.set()

    def _run(self) -> None:
        """Capture independently of browser presence while servicing bounded-latency commands."""
        try:
            while not self._stopped.is_set():
                self.tick()
                timeout = 0.05
                if self.recorder.state is RecorderState.RECORDING:
                    timeout = max(0.0, min(timeout, self._next_sample - self.recorder.clock()))
                try:
                    request = self._requests.get(timeout=timeout)
                except queue.Empty:
                    continue
                self._perform(request)
        finally:
            if not self._stopped.is_set():
                try:
                    self._execute("_shutdown", {"reason": "capture worker failed", "force": True})
                except BaseException as error:
                    self._error = f"capture worker cleanup failed: {error}"
                self._publish()
            while True:
                try:
                    pending = self._requests.get_nowait()
                except queue.Empty:
                    break
                pending.error = RuntimeError(
                    "recording controller stopped before command execution"
                )
                pending.done.set()

    def start_worker(self) -> None:
        """Start one source-independent capture owner; repeated starts are refused."""
        if self._thread is not None or self._stopped.is_set():
            raise RuntimeError("recording worker already started or closed")
        self._thread = threading.Thread(
            target=self._run, name="synria-console-capture", daemon=True
        )
        self._thread.start()

    def _submit(self, name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Marshal a transition to the owner or drive it directly in deterministic unit tests."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("command wait must be positive and finite")
        request = _Request(name, arguments)
        if self._thread is None:
            self._perform(request)
        elif self._stopped.is_set() or not self._thread.is_alive():
            raise RuntimeError("recording controller is closed")
        else:
            self._requests.put(request)
            if not request.done.wait(timeout_s):
                request.cancelled = True
                raise TimeoutError(
                    "command still running or cancelled; refresh state before retrying"
                )
        if request.error is not None:
            raise request.error
        return request.result or {}

    def command(
        self,
        name: str,
        *,
        reason: str = "",
        seconds: float = 3.0,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Accept only operator capture actions, never lifecycle or motion commands."""
        if name not in self.COMMANDS:
            raise ValueError("unknown recording command")
        return self._submit(name, {"reason": reason, "seconds": seconds}, timeout_s)

    def preview_frame(self, name: str) -> ImageFrame:
        """Read the same latest cache used by capture; callers encode without controller locks."""
        if self._stopped.is_set():
            raise RuntimeError("recording controller is closed")
        if name not in {"wrist", "front"}:
            raise ValueError("unknown preview source")
        source = self.recorder.wrist_source if name == "wrist" else self.recorder.front_source
        return source.read()

    def snapshot(self) -> dict[str, Any]:
        """Restore page state and source-health facts without touching capture timing."""
        with self._snapshot_lock:
            result = copy.deepcopy(self._snapshot)
        result["cameras"] = {}
        for name in ("wrist", "front"):
            try:
                frame = self.preview_frame(name)
                age = self.recorder.clock() - frame.monotonic_timestamp_s
                result["cameras"][name] = {
                    "source_id": frame.source_id,
                    "native_resolution": frame.native_resolution,
                    "stored_resolution": [
                        self.recorder.config.image_width,
                        self.recorder.config.image_height,
                    ],
                    "age_s": age,
                    "stale": not 0 <= age <= 0.2,
                }
            except Exception as error:
                result["cameras"][name] = {"error": str(error), "stale": True}
        try:
            if self._stopped.is_set():
                raise RuntimeError("state source is closed")
            state = self.recorder.state_source.read()
            sample = {
                "timestamp_s": state.monotonic_timestamp_s,
                "values": list(state.observation_vector()[:7]),
            }
            with self._snapshot_lock:
                if not self._trace or self._trace[-1]["timestamp_s"] != sample["timestamp_s"]:
                    self._trace.append(sample)
                trace = copy.deepcopy(list(self._trace))
            age = self.recorder.clock() - state.monotonic_timestamp_s
            measurement = self.recorder.config.state_rate_measurement
            result["follower"] = {
                **sample,
                "age_s": age,
                "stale": not 0 <= age <= 0.2,
                "rate_hz": measurement.rate_hz if measurement else None,
                "trace": trace,
            }
        except Exception as error:
            result["follower"] = {"error": str(error), "stale": True}
        try:
            result["free_disk_bytes"] = self._free_bytes()
        except Exception as error:
            result["disk_error"] = str(error)
        result["min_free_bytes"] = self.min_free_bytes
        return result

    def shutdown(self, reason: str, *, force: bool = False) -> None:
        """Refuse voluntary pending loss; signal shutdown records loss before closing sources."""
        if self._stopped.is_set():
            return
        self._submit("_shutdown", {"reason": reason, "force": force}, 60.0)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("capture worker has not stopped; resources remain owned")

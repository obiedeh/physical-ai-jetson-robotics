"""Time a task from the arm feed, read-only: movement start to rest, with the gripper cycle.

This is the motion timer approach from the bench: learn the rest noise for a few
seconds, then a move starts when any joint or the gripper leaves its rest value by
more than its threshold and ends when everything has stayed within the band for the
settle time. The reported duration excludes the settle time. Nothing here publishes,
commands or configures anything; it only consumes joint samples it is given.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

JOINT_THRESHOLD_MIN_RAD = math.radians(1.5)
GRIPPER_THRESHOLD_MIN_M = 0.001
DEFAULT_NOISE_S = 3.0
DEFAULT_SETTLE_S = 2.0
MIN_RUNS = 3
MAX_RUNS = 5


@dataclass(frozen=True)
class TimedRun:
    """One move from first motion to rest."""

    index: int
    duration_s: float
    peak_deg: float
    peak_joint: str
    gripper_travel_mm: float
    gripper_cycled: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the console API and the task version's timing record."""
        return {"index": self.index, "duration_s": round(self.duration_s, 2),
                "peak_deg": round(self.peak_deg, 1), "peak_joint": self.peak_joint,
                "gripper_travel_mm": round(self.gripper_travel_mm, 1),
                "gripper_cycled": self.gripper_cycled}


@dataclass
class MotionTimer:
    """A state machine fed with (time, seven values) samples; it never touches a device."""

    noise_s: float = DEFAULT_NOISE_S
    settle_s: float = DEFAULT_SETTLE_S
    max_runs: int = MAX_RUNS
    phase: str = "learning rest"
    rest: tuple[float, ...] | None = None
    thresholds: tuple[float, ...] = ()
    runs: list[TimedRun] = field(default_factory=list)
    _start_s: float | None = field(default=None, repr=False)
    _noise: list[tuple[float, ...]] = field(default_factory=list, repr=False)
    _moving_since: float | None = field(default=None, repr=False)
    _peak: tuple[float, str] = field(default=(0.0, ""), repr=False)
    _gripper: tuple[float, float] = field(default=(0.0, 0.0), repr=False)
    _quiet: deque[tuple[float, bool]] = field(default_factory=deque, repr=False)

    def feed(self, now_s: float, values: tuple[float, ...]) -> TimedRun | None:
        """Classify one sample; return a finished run when a move settles."""
        if len(values) != 7 or any(not math.isfinite(value) for value in values):
            return None
        if self._start_s is None:
            self._start_s = now_s
        if self.rest is None:
            self._noise.append(values)
            if now_s - self._start_s >= self.noise_s and len(self._noise) >= 10:
                self._learn_rest()
                self.phase = "waiting for a move"
            return None
        if len(self.runs) >= self.max_runs:
            self.phase = "done"
            return None
        away = [abs(value - rest) > threshold
                for value, rest, threshold in zip(values, self.rest, self.thresholds, strict=True)]
        if self._moving_since is None:
            if any(away):
                self._moving_since = now_s
                self._peak = (0.0, "")
                self._gripper = (values[6], values[6])
                self.phase = f"move {len(self.runs) + 1} in progress"
            return None
        deviation = max(
            (abs(values[index] - self.rest[index]), f"Joint{index + 1}") for index in range(6)
        )
        if math.degrees(deviation[0]) > self._peak[0]:
            self._peak = (math.degrees(deviation[0]), deviation[1])
        self._gripper = (min(self._gripper[0], values[6]), max(self._gripper[1], values[6]))
        self._quiet.append((now_s, not any(away)))
        while self._quiet and now_s - self._quiet[0][0] > self.settle_s:
            self._quiet.popleft()
        settled = (
            self._quiet and all(quiet for _, quiet in self._quiet)
            and now_s - self._quiet[0][0] >= self.settle_s * 0.95
        )
        if not settled:
            return None
        duration = self._quiet[0][0] - self._moving_since
        travel_mm = (self._gripper[1] - self._gripper[0]) * 1000.0
        run = TimedRun(len(self.runs) + 1, max(0.0, duration), self._peak[0], self._peak[1],
                       travel_mm, travel_mm >= 2.0)
        self.runs.append(run)
        self._moving_since = None
        self._quiet.clear()
        self.phase = "done" if len(self.runs) >= self.max_runs else "waiting for a move"
        return run

    def _learn_rest(self) -> None:
        """Rest pose is the median of the noise window; thresholds scale with its spread."""
        columns = list(zip(*self._noise, strict=True))
        self.rest = tuple(statistics.median(column) for column in columns)
        spread = [statistics.pstdev(column) for column in columns]
        self.thresholds = tuple(
            max(6.0 * spread[index], JOINT_THRESHOLD_MIN_RAD) for index in range(6)
        ) + (max(6.0 * spread[6], GRIPPER_THRESHOLD_MIN_M),)

    def proposed_window(self) -> dict[str, float] | None:
        """Propose a take window from the runs so far; the operator confirms or edits it."""
        if len(self.runs) < MIN_RUNS:
            return None
        durations = [run.duration_s for run in self.runs]
        minimum = max(1.0, math.floor(min(durations) * 0.8))
        maximum = max(minimum + 1.0, math.ceil(max(durations) * 1.3))
        return {"min_episode_s": float(minimum), "max_episode_s": float(maximum)}

    def snapshot(self) -> dict[str, Any]:
        """Progress in words and numbers for the screen."""
        return {
            "phase": self.phase, "runs": [run.as_dict() for run in self.runs],
            "runs_wanted": f"{MIN_RUNS} to {self.max_runs}",
            "proposed_window": self.proposed_window(),
            "rest_learned": self.rest is not None,
        }


class TimingSession:
    """Run one read-only timing pass on a background thread, feeding the timer from a source.

    The source is anything with ``read()`` returning a physical state and ``close()``;
    on the bench it is the same depth-one joint-state subscriber the recorder uses. The
    session owns the source for its lifetime and closes it when it stops.
    """

    def __init__(self, task_id: str, source_factory: Any, *, max_runs: int = MAX_RUNS,
                 poll_s: float = 0.02, sleep: Any = None) -> None:
        """Prepare but do not start; ``start()`` opens the source on its own thread."""
        import threading
        import time

        self.task_id = task_id
        self.timer = MotionTimer(max_runs=max_runs)
        self.error: str | None = None
        self.running = False
        self._factory = source_factory
        self._poll_s = poll_s
        self._sleep = sleep or time.sleep
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="synria-console-timer",
                                        daemon=True)

    def start(self) -> None:
        """Begin listening; the first sample may take the source's startup wait to arrive."""
        self.running = True
        self._thread.start()

    def _run(self) -> None:
        """Poll the source until stopped, done or failed; never publish anything."""
        source = None
        try:
            source = self._factory()
            last_stamp: float | None = None
            while not self._stop.is_set() and self.timer.phase != "done":
                state = source.read()
                stamp = float(state.monotonic_timestamp_s)
                if stamp != last_stamp:
                    last_stamp = stamp
                    self.timer.feed(stamp, tuple(state.observation_vector()[:7]))
                self._sleep(self._poll_s)
        except Exception as error:
            self.error = str(error)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    self.error = self.error or f"closing the arm feed failed: {error}"
            self.running = False

    def stop(self, timeout_s: float = 5.0) -> None:
        """Ask the thread to finish and wait for the source to close."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout_s)

    def snapshot(self) -> dict[str, Any]:
        """Progress for the screen, including any error from the arm feed."""
        return {**self.timer.snapshot(), "task_id": self.task_id, "running": self.running,
                "error": self.error}

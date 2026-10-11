"""The read-only motion timer with synthetic joint samples: rest, moves, settle, proposal."""

from __future__ import annotations

import math

from synria_lerobot.console.timing import MotionTimer, TimedRun

REST = (1.55, 1.56, -0.60, 0.10, -0.81, -0.17, 0.0001)


def feed_rest(timer: MotionTimer, start_s: float, seconds: float, rate_hz: float = 50.0) -> float:
    """Feed resting samples with a little noise; return the time reached."""
    now = start_s
    count = int(seconds * rate_hz)
    for index in range(count):
        now = start_s + index / rate_hz
        wobble = 0.0002 * (1 if index % 2 else -1)
        timer.feed(now, tuple(value + wobble for value in REST))
    return now + 1 / rate_hz


def feed_move(timer: MotionTimer, start_s: float, seconds: float, *, gripper_mm: float = 16.0,
              rate_hz: float = 50.0) -> tuple[float, list[TimedRun]]:
    """Feed a move: joints swing by up to 40 degrees and the gripper closes and reopens."""
    finished: list[TimedRun] = []
    now = start_s
    count = int(seconds * rate_hz)
    for index in range(count):
        now = start_s + index / rate_hz
        phase = index / max(1, count - 1)
        swing = math.radians(40.0) * math.sin(math.pi * phase)
        gripper = REST[6] + (gripper_mm / 1000.0) * math.sin(math.pi * phase)
        values = (REST[0] + swing, REST[1] + swing / 2, REST[2], REST[3], REST[4], REST[5], gripper)
        run = timer.feed(now, values)
        if run is not None:
            finished.append(run)
    return now + 1 / rate_hz, finished


def test_timer_learns_rest_then_times_each_move_to_rest() -> None:
    """Three moves give three runs whose durations exclude the settle time."""
    timer = MotionTimer()
    assert timer.phase == "learning rest"
    now = feed_rest(timer, 0.0, 3.5)
    assert timer.rest is not None and timer.phase == "waiting for a move"
    assert all(threshold >= math.radians(1.5) for threshold in timer.thresholds[:6])
    assert timer.thresholds[6] >= 0.001
    durations = []
    for move_s in (10.0, 12.0, 9.0):
        now, finished = feed_move(timer, now, move_s)
        assert finished == [], "a move is not finished until the arm has settled"
        now = feed_rest(timer, now, 3.0)
        assert len(timer.runs) == len(durations) + 1
        durations.append(timer.runs[-1].duration_s)
    for duration, expected in zip(durations, (10.0, 12.0, 9.0), strict=True):
        assert abs(duration - expected) < 0.6, (duration, expected)
    assert all(run.gripper_cycled and run.gripper_travel_mm > 14 for run in timer.runs)
    assert all(run.peak_joint == "Joint1" and 38 < run.peak_deg < 41 for run in timer.runs)
    assert [run.index for run in timer.runs] == [1, 2, 3]
    window = timer.proposed_window()
    assert window == {"min_episode_s": 7.0, "max_episode_s": 16.0}
    snapshot = timer.snapshot()
    assert snapshot["phase"] == "waiting for a move" and snapshot["runs_wanted"] == "3 to 5"
    assert snapshot["proposed_window"] == window and snapshot["rest_learned"] is True
    assert snapshot["runs"][0]["duration_s"] == round(durations[0], 2)


def test_timer_stops_after_the_maximum_runs_and_ignores_bad_samples() -> None:
    """Five runs end the timing; short or non-finite samples are ignored."""
    timer = MotionTimer(max_runs=5)
    now = feed_rest(timer, 0.0, 3.5)
    for _ in range(5):
        now, _ = feed_move(timer, now, 8.0, gripper_mm=0.0)
        now = feed_rest(timer, now, 3.0)
    assert len(timer.runs) == 5 and timer.phase == "done"
    assert all(not run.gripper_cycled for run in timer.runs), "no gripper travel, no cycle"
    assert timer.feed(now, (1.0, 2.0)) is None
    assert timer.feed(now, (float("nan"),) * 7) is None
    now, _ = feed_move(timer, now, 8.0)
    feed_rest(timer, now, 3.0)
    assert len(timer.runs) == 5, "a sixth move is not timed"


def test_no_proposal_before_three_runs() -> None:
    """The window is proposed only once enough runs exist for a range."""
    timer = MotionTimer()
    now = feed_rest(timer, 0.0, 3.5)
    now, _ = feed_move(timer, now, 10.0)
    feed_rest(timer, now, 3.0)
    assert len(timer.runs) == 1 and timer.proposed_window() is None
    assert timer.snapshot()["proposed_window"] is None

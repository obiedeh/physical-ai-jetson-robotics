"""Offline tests for slam/metrics.py on synthetic pose logs. No ROS, no hardware."""

from __future__ import annotations

import json
import math
from pathlib import Path

from slam.metrics import (
    KIDNAP_LIFT,
    KIDNAP_PLACE,
    LAP_END,
    LAP_START,
    Mark,
    Pose,
    compute_metrics,
    loop_closure_drift,
    pair_marks,
    percentile,
    relocalisation_trials,
    wrap_angle,
)


def _square_lap(
    frame: str, t0: float, side: float, drift_x: float, drift_yaw: float, hz: float = 5.0
) -> tuple[list[Pose], float]:
    """A square lap of ``side`` metres; the last pose returns to the start plus a drift."""
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
    poses: list[Pose] = []
    t = t0
    for (ax, ay), (bx, by) in zip(corners, corners[1:], strict=False):
        n = int(side * hz)
        for k in range(n):
            f = k / n
            poses.append(Pose(t, frame, ax + (bx - ax) * f, ay + (by - ay) * f, 0.0))
            t += 1.0 / hz
    poses.append(Pose(t, frame, drift_x, 0.0, drift_yaw))
    return poses, t


def test_percentile_is_index_based() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == 5.0
    assert percentile([], 50) is None


def test_wrap_angle() -> None:
    assert math.isclose(wrap_angle(math.pi + 0.1), -math.pi + 0.1, abs_tol=1e-9)


def test_pair_marks_sequential_and_drops_unpaired() -> None:
    marks = [Mark(0, LAP_START), Mark(1, LAP_END), Mark(2, LAP_END), Mark(3, LAP_START)]
    pairs = pair_marks(marks, LAP_START, LAP_END)
    assert [(a.t, b.t) for a, b in pairs] == [(0, 1)]


def test_loop_closure_drift_reports_injected_error() -> None:
    poses, t_end = _square_lap("map", 100.0, side=2.0, drift_x=0.12, drift_yaw=math.radians(3.0))
    marks = [Mark(100.0, LAP_START), Mark(t_end, LAP_END)]
    laps = loop_closure_drift(poses, marks, "map")
    assert len(laps) == 1
    lap = laps[0]
    assert math.isclose(lap.translation_error_m, 0.12, abs_tol=1e-3)
    assert math.isclose(lap.heading_error_deg, 3.0, abs_tol=1e-2)
    assert lap.path_length_m > 7.5  # four 2 m sides, discretised
    assert lap.drift_percent is not None and 1.0 < lap.drift_percent < 2.0


def test_loop_closure_skips_lap_without_pose_near_mark() -> None:
    poses, t_end = _square_lap("map", 100.0, side=1.0, drift_x=0.0, drift_yaw=0.0)
    marks = [Mark(100.0, LAP_START), Mark(t_end + 30.0, LAP_END)]
    assert loop_closure_drift(poses, marks, "map") == []


def test_relocalisation_converges_after_place() -> None:
    target = {"x": 1.0, "y": 2.0, "yaw": 0.0}
    place_t = 50.0
    poses: list[Pose] = []
    # 3 s of wrong pose, then within tolerance for 10 s
    for k in range(15):
        poses.append(Pose(place_t + k * 0.2, "map", 0.0, 0.0, 1.0))
    for k in range(50):
        poses.append(Pose(place_t + 3.0 + k * 0.2, "map", 1.02, 2.01, 0.01))
    marks = [Mark(40.0, KIDNAP_LIFT), Mark(place_t, KIDNAP_PLACE)]
    trials = relocalisation_trials(poses, marks, [target], timeout_s=30.0)
    assert len(trials) == 1
    tr = trials[0]
    assert tr.converged
    assert math.isclose(tr.time_to_converge_s or -1, 3.0, abs_tol=0.25)
    assert tr.final_translation_error_m is not None and tr.final_translation_error_m < 0.05


def test_relocalisation_fails_when_never_within_tolerance() -> None:
    target = {"x": 1.0, "y": 2.0, "yaw": 0.0}
    poses = [Pose(50.0 + k * 0.2, "map", 0.0, 0.0, 0.0) for k in range(100)]
    marks = [Mark(40.0, KIDNAP_LIFT), Mark(50.0, KIDNAP_PLACE)]
    tr = relocalisation_trials(poses, marks, [target], timeout_s=15.0)[0]
    assert not tr.converged
    assert tr.time_to_converge_s is None
    assert tr.final_translation_error_m is not None and tr.final_translation_error_m > 2.0


def test_compute_metrics_reads_session_dir_and_names_gaps(tmp_path: Path) -> None:
    poses, t_end = _square_lap("map", 100.0, side=1.0, drift_x=0.05, drift_yaw=0.0)
    odom, _ = _square_lap("odom", 100.0, side=1.0, drift_x=0.20, drift_yaw=0.0)
    with (tmp_path / "poses.jsonl").open("w") as fh:
        for p in poses + odom:
            row = {"t": p.t, "frame": p.frame, "x": p.x, "y": p.y, "yaw": p.yaw}
            fh.write(json.dumps(row) + "\n")
    with (tmp_path / "marks.jsonl").open("w") as fh:
        fh.write(json.dumps({"t": 100.0, "label": LAP_START}) + "\n")
        fh.write(json.dumps({"t": t_end, "label": LAP_END}) + "\n")
    m = compute_metrics(tmp_path)
    assert m["inputs"]["n_poses"] == {"map": len(poses), "odom": len(odom)}
    assert m["inputs"]["poses_sha256"] is not None
    assert m["loop_closure"]["map"]["laps_scored"] == 1
    assert math.isclose(m["loop_closure"]["map"]["translation_error_m"]["p50"], 0.05, abs_tol=1e-3)
    assert math.isclose(m["loop_closure"]["odom"]["translation_error_m"]["p50"], 0.20, abs_tol=1e-3)
    assert m["relocalisation"] is None
    assert any("relocalisation" in s for s in m["not_measured"])

"""Loop-closure drift and re-localisation metrics for rover SLAM sessions.

Inputs are two JSONL files written on the robot during a session:

* ``poses.jsonl``: one row per pose sample,
  ``{"t": <unix s>, "frame": "map" | "odom", "x": m, "y": m, "yaw": rad}``.
  ``map`` rows are the SLAM estimate of the base in the map frame; ``odom``
  rows are the board's raw wheel odometry. Written by
  ``scripts/jetson/slam_pose_log.py``.
* ``marks.jsonl``: operator marks, ``{"t": <unix s>, "label": str}``.
  Written by ``scripts/jetson/slam_session.py mark``.

Nothing here talks to ROS. Every number is recomputed from the files, so a
committed session directory reproduces its own metrics. Percentiles are
index based, ``sorted[round(p * (n - 1))]``, like the rest of the repo.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LAP_START = "lap_start"
LAP_END = "lap_end"
KIDNAP_LIFT = "kidnap_lift"
KIDNAP_PLACE = "kidnap_place"

FRAMES = ("map", "odom")


@dataclass(frozen=True)
class Pose:
    t: float
    frame: str
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Mark:
    t: float
    label: str


@dataclass(frozen=True)
class LapDrift:
    """Closed-loop return error for one lap in one frame."""

    lap: int
    frame: str
    t_start: float
    t_end: float
    duration_s: float
    path_length_m: float
    translation_error_m: float
    heading_error_deg: float
    drift_percent: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KidnapTrial:
    """Re-localisation after the robot was lifted and set down on a taped target."""

    trial: int
    t_lift: float
    t_place: float
    target: dict[str, float]
    converged: bool
    time_to_converge_s: float | None
    final_translation_error_m: float | None
    final_heading_error_deg: float | None
    n_poses_in_window: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_poses(path: Path) -> list[Pose]:
    poses = [
        Pose(float(r["t"]), str(r["frame"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
        for r in read_jsonl(path)
    ]
    return sorted(poses, key=lambda p: p.t)


def load_marks(path: Path) -> list[Mark]:
    marks = [Mark(float(r["t"]), str(r["label"])) for r in read_jsonl(path)]
    return sorted(marks, key=lambda m: m.t)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Geometry and statistics
# ---------------------------------------------------------------------------


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50) or 0.0, 4),
        "p95": round(percentile(values, 95) or 0.0, 4),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def pose_near(
    poses: list[Pose], t: float, frame: str, tolerance_s: float = 0.5
) -> Pose | None:
    """Nearest sample in ``frame`` within ``tolerance_s`` of ``t``."""
    best: Pose | None = None
    best_dt = tolerance_s
    for p in poses:
        if p.frame != frame:
            continue
        dt = abs(p.t - t)
        if dt <= best_dt:
            best, best_dt = p, dt
    return best


def path_length(poses: list[Pose], frame: str, t0: float, t1: float) -> float:
    pts = [p for p in poses if p.frame == frame and t0 <= p.t <= t1]
    total = 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total


def pair_marks(marks: list[Mark], start_label: str, end_label: str) -> list[tuple[Mark, Mark]]:
    """Pair each start mark with the next end mark after it; unpaired marks are dropped."""
    pairs: list[tuple[Mark, Mark]] = []
    pending: Mark | None = None
    for m in marks:
        if m.label == start_label:
            pending = m
        elif m.label == end_label and pending is not None:
            pairs.append((pending, m))
            pending = None
    return pairs


# ---------------------------------------------------------------------------
# Loop-closure drift
# ---------------------------------------------------------------------------


def loop_closure_drift(
    poses: list[Pose], marks: list[Mark], frame: str, tolerance_s: float = 0.5
) -> list[LapDrift]:
    """Return error at ``lap_end`` relative to ``lap_start`` for each marked lap.

    The robot is driven back to the same taped start square, so the true
    displacement is zero and the reported error is the estimator's drift.
    Laps with no pose sample near either mark are skipped; the caller
    reports ``laps_marked`` against ``laps_scored``.
    """
    laps: list[LapDrift] = []
    for i, (start, end) in enumerate(pair_marks(marks, LAP_START, LAP_END), start=1):
        ps = pose_near(poses, start.t, frame, tolerance_s)
        pe = pose_near(poses, end.t, frame, tolerance_s)
        if ps is None or pe is None:
            continue
        err = math.hypot(pe.x - ps.x, pe.y - ps.y)
        length = path_length(poses, frame, start.t, end.t)
        laps.append(
            LapDrift(
                lap=i,
                frame=frame,
                t_start=start.t,
                t_end=end.t,
                duration_s=round(end.t - start.t, 3),
                path_length_m=round(length, 4),
                translation_error_m=round(err, 4),
                heading_error_deg=round(math.degrees(wrap_angle(pe.yaw - ps.yaw)), 3),
                drift_percent=round(err / length * 100.0, 3) if length > 0 else None,
            )
        )
    return laps


def lap_summary(laps: list[LapDrift], laps_marked: int) -> dict[str, Any]:
    return {
        "laps_marked": laps_marked,
        "laps_scored": len(laps),
        "translation_error_m": summarize([lap.translation_error_m for lap in laps]),
        "abs_heading_error_deg": summarize([abs(lap.heading_error_deg) for lap in laps]),
        "drift_percent": summarize(
            [lap.drift_percent for lap in laps if lap.drift_percent is not None]
        ),
        "laps": [lap.to_dict() for lap in laps],
    }


# ---------------------------------------------------------------------------
# Re-localisation after kidnap
# ---------------------------------------------------------------------------


def relocalisation_trials(
    poses: list[Pose],
    marks: list[Mark],
    targets: list[dict[str, float]],
    frame: str = "map",
    converge_m: float = 0.15,
    converge_deg: float = 10.0,
    hold_s: float = 2.0,
    timeout_s: float = 60.0,
) -> list[KidnapTrial]:
    """Score each ``kidnap_lift`` / ``kidnap_place`` pair against a tape-measured target.

    A trial converges at the first sample after ``kidnap_place`` whose error is
    within ``converge_m`` and ``converge_deg`` and stays within them for
    ``hold_s``. Trials beyond the supplied targets are not scored.
    """
    trials: list[KidnapTrial] = []
    pairs = pair_marks(marks, KIDNAP_LIFT, KIDNAP_PLACE)
    for i, ((lift, place), target) in enumerate(zip(pairs, targets, strict=False), start=1):
        window = [
            p for p in poses if p.frame == frame and place.t <= p.t <= place.t + timeout_s
        ]

        def err(p: Pose, tgt: dict[str, float] = target) -> tuple[float, float]:
            d = math.hypot(p.x - tgt["x"], p.y - tgt["y"])
            h = abs(math.degrees(wrap_angle(p.yaw - tgt["yaw"])))
            return d, h

        def within(p: Pose, check: Callable[[Pose], tuple[float, float]] = err) -> bool:
            d, h = check(p)
            return d <= converge_m and h <= converge_deg

        t_conv: float | None = None
        for j, p in enumerate(window):
            if not within(p):
                continue
            hold = [q for q in window[j:] if q.t <= p.t + hold_s]
            if all(within(q) for q in hold) and (hold[-1].t - p.t >= hold_s * 0.5):
                t_conv = p.t - place.t
                break
        final_d, final_h = err(window[-1]) if window else (None, None)
        trials.append(
            KidnapTrial(
                trial=i,
                t_lift=lift.t,
                t_place=place.t,
                target=dict(target),
                converged=t_conv is not None,
                time_to_converge_s=round(t_conv, 3) if t_conv is not None else None,
                final_translation_error_m=round(final_d, 4) if final_d is not None else None,
                final_heading_error_deg=round(final_h, 3) if final_h is not None else None,
                n_poses_in_window=len(window),
            )
        )
    return trials


def kidnap_summary(trials: list[KidnapTrial], trials_marked: int) -> dict[str, Any]:
    converged = [t for t in trials if t.converged]
    return {
        "trials_marked": trials_marked,
        "trials_scored": len(trials),
        "converged": len(converged),
        "success_rate": round(len(converged) / len(trials), 3) if trials else None,
        "time_to_converge_s": summarize(
            [t.time_to_converge_s for t in converged if t.time_to_converge_s is not None]
        ),
        "final_translation_error_m": summarize(
            [
                t.final_translation_error_m
                for t in trials
                if t.final_translation_error_m is not None
            ]
        ),
        "trials": [t.to_dict() for t in trials],
    }


# ---------------------------------------------------------------------------
# Session entry point
# ---------------------------------------------------------------------------


def compute_metrics(
    session_dir: Path,
    targets: list[dict[str, float]] | None = None,
    converge_m: float = 0.15,
    converge_deg: float = 10.0,
    hold_s: float = 2.0,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Compute every metric the session files support and say which they do not."""
    session_dir = Path(session_dir)
    poses_path = session_dir / "poses.jsonl"
    marks_path = session_dir / "marks.jsonl"
    poses = load_poses(poses_path) if poses_path.exists() else []
    marks = load_marks(marks_path) if marks_path.exists() else []
    laps_marked = len(pair_marks(marks, LAP_START, LAP_END))
    kidnaps_marked = len(pair_marks(marks, KIDNAP_LIFT, KIDNAP_PLACE))

    out: dict[str, Any] = {
        "schema": "slam-metrics-v1",
        "inputs": {
            "poses_file": poses_path.name if poses_path.exists() else None,
            "poses_sha256": sha256_file(poses_path) if poses_path.exists() else None,
            "marks_file": marks_path.name if marks_path.exists() else None,
            "marks_sha256": sha256_file(marks_path) if marks_path.exists() else None,
            "n_poses": {f: sum(1 for p in poses if p.frame == f) for f in FRAMES},
            "n_marks": len(marks),
        },
        "loop_closure": {
            f: lap_summary(loop_closure_drift(poses, marks, f), laps_marked) for f in FRAMES
        },
        "relocalisation": None,
        "not_measured": [],
    }
    if laps_marked == 0:
        out["not_measured"].append("loop_closure: no lap_start/lap_end pairs in marks")
    if targets:
        trials = relocalisation_trials(
            poses, marks, targets, converge_m=converge_m, converge_deg=converge_deg,
            hold_s=hold_s, timeout_s=timeout_s,
        )
        out["relocalisation"] = {
            "parameters": {
                "converge_m": converge_m,
                "converge_deg": converge_deg,
                "hold_s": hold_s,
                "timeout_s": timeout_s,
            },
            **kidnap_summary(trials, kidnaps_marked),
        }
    else:
        out["not_measured"].append("relocalisation: no targets supplied")
    return out

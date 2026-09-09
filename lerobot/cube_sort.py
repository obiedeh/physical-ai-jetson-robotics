"""Simulated colored-cube sorting demo for the Synria 6DOF arm.

Simulates the Synria upstream cube-sort workflow (``Synria-Robotics/Alicia-D-ROS2``
cube-sort package) without real hardware:

1. ``MockColorDetector`` — assigns deterministic color labels and synthetic
   board positions to cubes using a seeded random generator.
2. ``CubeSortPlanner`` — maps detected color → target staging zone and
   generates a validated ``JointTrajectory`` via ``arm_control``.
3. ``CubeSortSimulation`` — runs N cubes through detection → planning →
   safety-gate validation, records per-cube outcomes, and builds a summary
   dict for the CLI report writer.

Color → staging zone mapping (aligned with ``STAGING_POSES`` in ``arm_control.demo``):

    red    → right
    green  → left
    blue   → top
    yellow → bottom

All trajectories are validated through ``ArmSafetyGate`` before being
counted as passed. No real arm motion occurs — this is a pure simulation
intended for pre-hardware planning verification.

Usage::

    from lerobot.cube_sort import CubeSortSimulation

    sim = CubeSortSimulation(n_cubes=6, seed=42)
    results = sim.run()
    print(sim.summary(results))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Deterministic color palette — order sets the sampling distribution.
CUBE_COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow")

#: Color → staging zone assignment (matches arm_control.demo.STAGING_POSES keys).
COLOR_TO_ZONE: dict[str, str] = {
    "red": "right",
    "green": "left",
    "blue": "top",
    "yellow": "bottom",
}

#: Board positions (x, y) in metres relative to arm base.
_BOARD_POSITIONS: list[tuple[float, float]] = [
    (0.28, -0.05),
    (0.28,  0.05),
    (0.30,  0.00),
    (0.32, -0.08),
    (0.32,  0.08),
    (0.25,  0.00),
    (0.27,  0.10),
    (0.27, -0.10),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeDetection:
    """Result of the mock color-detection step for one cube."""

    cube_idx: int
    color: str
    position_x_m: float
    position_y_m: float
    position_z_m: float
    confidence: float  # in [0, 1]

    def target_zone(self) -> str:
        """Map detected color to its target staging zone."""
        return COLOR_TO_ZONE[self.color]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cube_idx": self.cube_idx,
            "color": self.color,
            "position_m": [
                round(self.position_x_m, 4),
                round(self.position_y_m, 4),
                round(self.position_z_m, 4),
            ],
            "confidence": round(self.confidence, 4),
            "target_zone": self.target_zone(),
        }


@dataclass(frozen=True)
class CubeSortResult:
    """Outcome of the full detection → planning → safety-gate pipeline for one cube."""

    cube_idx: int
    color: str
    target_zone: str
    detection_confidence: float
    trajectory_steps: int
    trajectory_duration_s: float
    trajectory_valid: bool  # safety gate passed
    safety_errors: list[str]

    @property
    def passed(self) -> bool:
        return self.trajectory_valid and len(self.safety_errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cube_idx": self.cube_idx,
            "color": self.color,
            "target_zone": self.target_zone,
            "detection_confidence": round(self.detection_confidence, 4),
            "trajectory_steps": self.trajectory_steps,
            "trajectory_duration_s": round(self.trajectory_duration_s, 3),
            "trajectory_valid": self.trajectory_valid,
            "safety_errors": self.safety_errors,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# MockColorDetector
# ---------------------------------------------------------------------------


class MockColorDetector:
    """Generates deterministic cube detections using a seeded RNG.

    Simulates an overhead RGB camera detecting colored cubes on a tabletop.
    In real hardware, this is replaced by a YOLOv8 / OpenCV HSV pipeline
    using the overhead board camera.

    Args:
        seed: RNG seed for deterministic output.
        n_cubes: Total number of cubes to detect.
    """

    def __init__(self, seed: int = 42, n_cubes: int = 6) -> None:
        self._rng = Random(seed)
        self._n_cubes = n_cubes

    def detect(self) -> list[CubeDetection]:
        """Return a list of synthetic cube detections."""
        detections: list[CubeDetection] = []
        positions = list(_BOARD_POSITIONS)
        self._rng.shuffle(positions)

        for idx in range(self._n_cubes):
            color = CUBE_COLORS[idx % len(CUBE_COLORS)]
            bx, by = positions[idx % len(positions)]
            # Add small position noise
            bx += self._rng.uniform(-0.005, 0.005)
            by += self._rng.uniform(-0.005, 0.005)
            confidence = 0.82 + self._rng.uniform(0.0, 0.17)
            detections.append(
                CubeDetection(
                    cube_idx=idx,
                    color=color,
                    position_x_m=round(bx, 5),
                    position_y_m=round(by, 5),
                    position_z_m=0.020,  # cube sits 20 mm above table
                    confidence=round(min(1.0, confidence), 4),
                )
            )
        return detections


# ---------------------------------------------------------------------------
# CubeSortPlanner
# ---------------------------------------------------------------------------


class CubeSortPlanner:
    """Maps a CubeDetection → JointTrajectory for the pick-place cycle.

    The trajectory is:  home → board_approach → staging_zone → home

    Uses ``arm_control.demo`` for named waypoints and
    ``arm_control.safety.ArmSafetyGate`` for limit validation.
    """

    def __init__(self) -> None:
        try:
            from arm_control.demo import (
                board_approach_trajectory,
                home_trajectory,
                staging_zone_trajectory,
            )
            from arm_control.safety import ArmSafetyGate

            self._home_traj = home_trajectory
            self._approach_traj = board_approach_trajectory
            self._staging_traj = staging_zone_trajectory
            self._gate = ArmSafetyGate()
            self._available = True
        except ImportError:  # pragma: no cover
            self._available = False

    def plan(self, detection: CubeDetection) -> CubeSortResult:
        """Plan and validate a full pick-sort cycle for one detected cube."""
        if not self._available:  # pragma: no cover
            return CubeSortResult(
                cube_idx=detection.cube_idx,
                color=detection.color,
                target_zone=detection.target_zone(),
                detection_confidence=detection.confidence,
                trajectory_steps=0,
                trajectory_duration_s=0.0,
                trajectory_valid=False,
                safety_errors=["arm_control not available"],
            )

        zone = detection.target_zone()
        trajectories = [
            self._home_traj(),
            self._approach_traj(),
            self._staging_traj(zone),
            self._home_traj(),
        ]

        all_errors: list[str] = []
        total_steps = 0
        total_duration = 0.0

        for traj in trajectories:
            errors = traj.validate()
            all_errors.extend(str(e) for e in errors)
            total_steps += len(traj.waypoints)
            total_duration += traj.duration_s()

        return CubeSortResult(
            cube_idx=detection.cube_idx,
            color=detection.color,
            target_zone=zone,
            detection_confidence=detection.confidence,
            trajectory_steps=total_steps,
            trajectory_duration_s=round(total_duration, 3),
            trajectory_valid=len(all_errors) == 0,
            safety_errors=all_errors,
        )


# ---------------------------------------------------------------------------
# CubeSortSimulation
# ---------------------------------------------------------------------------


@dataclass
class CubeSortSimulation:
    """End-to-end simulated cube sorting session.

    Runs ``n_cubes`` cubes through the full pipeline:
        detection → zone assignment → trajectory planning → safety validation

    Args:
        n_cubes: Number of cubes to sort (1–24).
        seed: RNG seed passed to MockColorDetector.
    """

    n_cubes: int = 6
    seed: int = 42
    _detector: MockColorDetector = field(init=False)
    _planner: CubeSortPlanner = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.n_cubes <= 24:
            raise ValueError(f"n_cubes must be in [1, 24], got {self.n_cubes}")
        self._detector = MockColorDetector(seed=self.seed, n_cubes=self.n_cubes)
        self._planner = CubeSortPlanner()

    def run(self) -> list[CubeSortResult]:
        """Run the simulation and return one result per cube."""
        detections = self._detector.detect()
        return [self._planner.plan(d) for d in detections]

    def summary(self, results: list[CubeSortResult]) -> dict[str, Any]:
        """Build a JSON-serialisable summary dict from a results list."""
        passed = sum(1 for r in results if r.passed)
        color_counts: dict[str, int] = {}
        zone_assignments: dict[str, list[int]] = {}

        for r in results:
            color_counts[r.color] = color_counts.get(r.color, 0) + 1
            zone_assignments.setdefault(r.target_zone, []).append(r.cube_idx)

        return {
            "n_cubes": self.n_cubes,
            "passed": passed,
            "failed": self.n_cubes - passed,
            "pass_rate": round(passed / max(self.n_cubes, 1), 4),
            "color_counts": color_counts,
            "zone_assignments": zone_assignments,
            "total_trajectory_duration_s": round(
                sum(r.trajectory_duration_s for r in results), 3
            ),
            "results": [r.to_dict() for r in results],
        }

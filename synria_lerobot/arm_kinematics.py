"""Forward and inverse kinematics of the Synria arm from its committed chain description.

The chain in ``config/synria_kinematics.json`` is the URDF's joint origins and axes
and the fixed joint to ``gripper_center``, the fingertip plane on the gripper's
centre line whose +Z is the approach axis and whose Y is the finger opening axis.
Joint angles the arm reports are in the driver's encoder convention; each lane
carries calibration offsets that turn a reported angle into the URDF angle, so
every function here takes and returns reported angles and applies the offsets
inside. Inverse kinematics is damped least squares on a position target plus an
approach-axis target, five constraints on six joints, so the rotation about the
approach axis stays free unless a finger axis is asked for as well; joint bounds
are honoured by clamping the iterate inside the operator limits less a margin, so
a solution never rests on a bound the policy client would hold at. Nothing here
touches hardware; it is plain numpy and is used by the staged executor and by the
dataset audit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

KINEMATICS_VERSION = "synria_kinematics_v1"
DEFAULT_MARGIN_RAD = math.radians(3.0)


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw as a rotation matrix (Rz * Ry * Rx)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return np.asarray(rz @ ry @ rx)


def _transform(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> np.ndarray:
    """A homogeneous transform from a URDF origin."""
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation_rpy(*rpy)
    matrix[:3, 3] = xyz
    return matrix


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis as a homogeneous transform."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ])
    return matrix


@dataclass(frozen=True)
class IkSolution:
    """A converged inverse-kinematics result in reported joint angles."""

    joints_rad: tuple[float, ...]
    position_error_m: float
    axis_error_rad: float
    iterations: int


class KinematicChain:
    """The six-joint chain plus the gripper centre, with per-lane calibration offsets."""

    def __init__(self, description: dict[str, Any]) -> None:
        """Validate the committed description; refuse a chain that is not six joints."""
        if description.get("version") != KINEMATICS_VERSION:
            raise ValueError("unsupported kinematics description version")
        chain = description["chain"]
        if len(chain) != 6 or len(description["joint_names"]) != 6:
            raise ValueError("the chain must describe exactly six joints")
        self.joint_names = tuple(str(name) for name in description["joint_names"])
        self._links = []
        for entry in chain:
            axis = np.asarray(entry["axis"], dtype=float)
            norm = float(np.linalg.norm(axis))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError("every joint axis must be a finite nonzero vector")
            self._links.append((_transform(tuple(entry["xyz"]), tuple(entry["rpy"])), axis / norm))
        tool = description["tool"]
        self._tool = _transform(tuple(tool["xyz"]), tuple(tool["rpy"]))
        base = description.get("base_in_world", {"xyz": [0.0, 0.0, 0.0], "yaw_deg": 0.0})
        self._base = _transform(tuple(base["xyz"]), (0.0, 0.0, math.radians(base["yaw_deg"])))
        self._offsets = {
            lane: np.array([math.radians(float(values.get(name, 0.0)))
                            for name in self.joint_names])
            for lane, values in description["calibration_offsets_deg"].items()
        }
        if not self._offsets:
            raise ValueError("calibration offsets must name at least one lane")

    @classmethod
    def load(cls, path: Path) -> KinematicChain:
        """Read the committed JSON description."""
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def lanes(self) -> tuple[str, ...]:
        """The lanes with calibration offsets."""
        return tuple(self._offsets)

    def offsets(self, lane: str) -> np.ndarray:
        """The six reported-to-URDF offsets of one lane, in radians."""
        if lane not in self._offsets:
            raise ValueError(f"no calibration offsets for lane {lane!r}")
        return self._offsets[lane]

    def forward(self, joints_rad: tuple[float, ...] | np.ndarray, lane: str) -> np.ndarray:
        """The gripper centre's world transform for six reported joint angles."""
        joints = np.asarray(joints_rad, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("forward kinematics needs six finite joint angles")
        urdf = joints + self.offsets(lane)
        matrix = self._base.copy()
        for (origin, axis), angle in zip(self._links, urdf, strict=True):
            matrix = matrix @ origin @ _rotation_about(axis, float(angle))
        return np.asarray(matrix @ self._tool)

    def position(self, joints_rad: tuple[float, ...] | np.ndarray, lane: str) -> np.ndarray:
        """The gripper centre's world position."""
        return np.array(self.forward(joints_rad, lane)[:3, 3], dtype=float)

    def approach_axis(self, joints_rad: tuple[float, ...] | np.ndarray, lane: str) -> np.ndarray:
        """The world direction of the gripper's approach axis (tool +Z)."""
        return np.array(self.forward(joints_rad, lane)[:3, 2], dtype=float)

    def finger_axis(self, joints_rad: tuple[float, ...] | np.ndarray, lane: str) -> np.ndarray:
        """The world direction along which the fingers open (tool +Y)."""
        return np.array(self.forward(joints_rad, lane)[:3, 1], dtype=float)

    def _residual(self, joints: np.ndarray, lane: str, target: np.ndarray,
                  approach: np.ndarray, finger: np.ndarray | None,
                  posture: np.ndarray | None, posture_weight: float) -> np.ndarray:
        matrix = self.forward(joints, lane)
        parts = [matrix[:3, 3] - target, np.cross(matrix[:3, 2], approach)]
        if finger is not None:
            parts.append(np.cross(matrix[:3, 1], finger))
        if posture is not None:
            parts.append(posture_weight * (joints - posture))
        return np.concatenate(parts)

    def solve_ik(
        self,
        target_position_m: tuple[float, float, float] | np.ndarray,
        approach_axis: tuple[float, float, float] | np.ndarray,
        seed_rad: tuple[float, ...] | np.ndarray,
        lane: str,
        bounds_rad: tuple[tuple[float, float], ...],
        *,
        finger_axis: tuple[float, float, float] | np.ndarray | None = None,
        posture_rad: tuple[float, ...] | np.ndarray | None = None,
        posture_weight: float = 0.02,
        margin_rad: float = DEFAULT_MARGIN_RAD,
        position_tolerance_m: float = 0.002,
        axis_tolerance_rad: float = math.radians(2.0),
        max_iterations: int = 200,
        damping: float = 0.02,
    ) -> IkSolution | None:
        """Damped least squares from the seed toward a position and an approach axis.

        ``bounds_rad`` are the six operator limits in reported angles; the iterate is
        clamped inside them less ``margin_rad`` after every step. ``posture_rad`` is a
        weak pull toward a reference configuration, weighted by ``posture_weight`` in
        metres per radian and faded out over the first half of the iterations, which
        selects the arm's branch (elbow up or down) when the five constraints leave it
        free without fighting the final millimetres; it never decides convergence. The result is
        None when the residual has not met both tolerances within ``max_iterations``,
        which is how an unreachable target is reported; the caller decides what that
        means.
        """
        target = np.asarray(target_position_m, dtype=float)
        approach = np.asarray(approach_axis, dtype=float)
        finger = None if finger_axis is None else np.asarray(finger_axis, dtype=float)
        if target.shape != (3,) or approach.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("inverse kinematics needs a finite target and approach axis")
        norm = float(np.linalg.norm(approach))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("the approach axis must be a nonzero vector")
        approach = approach / norm
        if finger is not None:
            if finger.shape != (3,) or float(np.linalg.norm(finger)) == 0.0:
                raise ValueError("the finger axis must be a nonzero vector")
            finger = finger / float(np.linalg.norm(finger))
        if len(bounds_rad) != 6 or any(hi - lo <= 2 * margin_rad for lo, hi in bounds_rad):
            raise ValueError("six operator bounds wider than twice the margin are required")
        lower = np.array([lo + margin_rad for lo, _ in bounds_rad])
        upper = np.array([hi - margin_rad for _, hi in bounds_rad])
        posture = None if posture_rad is None else np.asarray(posture_rad, dtype=float)[:6]
        if posture is not None and (posture.shape != (6,) or not np.isfinite(posture).all()):
            raise ValueError("the posture prior needs six finite joint angles")
        if not math.isfinite(posture_weight) or posture_weight < 0:
            raise ValueError("posture_weight must be a finite non-negative number")
        joints = np.clip(np.asarray(seed_rad, dtype=float)[:6], lower, upper)
        step = 1e-5
        for iteration in range(1, max_iterations + 1):
            weight = posture_weight * max(0.0, 1.0 - 2.0 * iteration / max_iterations)
            residual = self._residual(joints, lane, target, approach, finger, posture, weight)
            position_error = float(np.linalg.norm(residual[:3]))
            axis_error = float(math.asin(min(1.0, float(np.linalg.norm(residual[3:6])))))
            if position_error <= position_tolerance_m and axis_error <= axis_tolerance_rad:
                return IkSolution(tuple(float(v) for v in joints), position_error, axis_error,
                                  iteration)
            jacobian = np.empty((residual.size, 6))
            for index in range(6):
                bumped = joints.copy()
                bumped[index] += step
                jacobian[:, index] = (
                    self._residual(bumped, lane, target, approach, finger, posture, weight)
                    - residual
                ) / step
            normal = jacobian.T @ jacobian + (damping ** 2) * np.eye(6)
            delta = np.linalg.solve(normal, jacobian.T @ residual)
            largest = float(np.max(np.abs(delta)))
            if largest > 0.3:
                delta *= 0.3 / largest
            joints = np.clip(joints - delta, lower, upper)
        return None

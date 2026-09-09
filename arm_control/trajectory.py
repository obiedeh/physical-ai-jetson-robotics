"""Joint trajectory data structures for the Synria 6DOF arm.

These structures represent planned and executed trajectories. They are
framework-agnostic and carry no ROS or hardware dependencies so they can
be created, validated, and tested on any machine.

Typical workflow::

    # Plan a trajectory
    traj = JointTrajectory(
        name="reach_to_staging_zone_a",
        waypoints=[
            JointWaypoint({"joint_1": 0.0, "joint_2": 0.5, ...}, 0.0),
            JointWaypoint({"joint_1": 0.3, "joint_2": 0.8, ...}, 1.5),
        ],
    )
    errors = traj.validate()

    # Track execution
    record = TrajectoryExecutionRecord(trajectory=traj)
    record.status = TrajectoryStatus.executing
    # ... receive commanded/measured waypoints from the controller ...
    err = record.max_position_error_rad()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Joint name order matches the Synria 6DOF URDF.
JOINT_NAMES: tuple[str, ...] = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)


class TrajectoryStatus(str, Enum):
    pending = "pending"
    executing = "executing"
    success = "success"
    aborted = "aborted"
    safety_stop = "safety_stop"


@dataclass(frozen=True)
class JointWaypoint:
    """One joint-space waypoint with a time-from-start stamp."""

    positions_rad: dict[str, float]
    time_from_start_s: float

    def unknown_joints(self) -> list[str]:
        """Return joint names that are not in the Synria 6DOF set."""
        return [j for j in self.positions_rad if j not in JOINT_NAMES]

    def to_dict(self) -> dict[str, object]:
        return {
            "positions_rad": dict(self.positions_rad),
            "time_from_start_s": self.time_from_start_s,
        }


@dataclass(frozen=True)
class JointTrajectory:
    """Ordered list of joint-space waypoints."""

    name: str
    waypoints: list[JointWaypoint]

    def duration_s(self) -> float:
        """Total trajectory duration from first to last waypoint."""
        if not self.waypoints:
            return 0.0
        return max(wp.time_from_start_s for wp in self.waypoints)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty list = valid).

        Checks:
          - at least one waypoint
          - strictly increasing time stamps
          - no unrecognised joint names
        """
        errors: list[str] = []
        if not self.waypoints:
            errors.append("Trajectory has no waypoints.")
            return errors

        prev_t = -1.0
        for idx, wp in enumerate(self.waypoints):
            if wp.time_from_start_s <= prev_t:
                errors.append(
                    f"Waypoint {idx}: time_from_start_s "
                    f"{wp.time_from_start_s:.3f} s is not strictly "
                    f"increasing (previous={prev_t:.3f} s)."
                )
            prev_t = wp.time_from_start_s
            unknown = wp.unknown_joints()
            if unknown:
                errors.append(
                    f"Waypoint {idx}: unknown joint names {unknown}."
                )
        return errors

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "duration_s": self.duration_s(),
            "n_waypoints": len(self.waypoints),
            "waypoints": [wp.to_dict() for wp in self.waypoints],
        }


@dataclass
class TrajectoryExecutionRecord:
    """Mutable record of planned vs measured joint positions during execution.

    The controller appends waypoints to ``commanded_waypoints`` and
    ``measured_waypoints`` as execution proceeds. Call
    ``max_position_error_rad()`` afterwards to quantify tracking accuracy.
    """

    trajectory: JointTrajectory
    status: TrajectoryStatus = TrajectoryStatus.pending
    commanded_waypoints: list[JointWaypoint] = field(default_factory=list)
    measured_waypoints: list[JointWaypoint] = field(default_factory=list)
    abort_reason: str | None = None

    def max_position_error_rad(self) -> float | None:
        """Maximum absolute joint-position tracking error across all pairs.

        Returns ``None`` when there are no matched commanded/measured pairs.
        """
        pairs = list(zip(self.commanded_waypoints, self.measured_waypoints, strict=False))
        if not pairs:
            return None

        max_err = 0.0
        for cmd, meas in pairs:
            for joint, cmd_pos in cmd.positions_rad.items():
                if joint in meas.positions_rad:
                    err = abs(cmd_pos - meas.positions_rad[joint])
                    max_err = max(max_err, err)
        return round(max_err, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "trajectory_name": self.trajectory.name,
            "status": self.status.value,
            "abort_reason": self.abort_reason,
            "max_position_error_rad": self.max_position_error_rad(),
            "n_commanded": len(self.commanded_waypoints),
            "n_measured": len(self.measured_waypoints),
        }

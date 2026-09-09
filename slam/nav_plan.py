"""Navigation waypoint planning for the Yahboom ROSMASTER M3 Pro.

Provides pure-Python waypoint structures, plan construction, and validation.
All geometry is computed in 2D (XY floor plane) and is framework-agnostic —
no ROS environment required.

Typical workflow::

    from slam.nav_plan import Waypoint, WaypointType, build_nav_plan, validate_nav_plan

    waypoints = [
        Waypoint("home",    x_m=0.0, y_m=0.0, heading_rad=0.0),
        Waypoint("workcell", x_m=2.5, y_m=1.0, heading_rad=0.4),
        Waypoint("inspect", x_m=3.0, y_m=1.0, heading_rad=0.0,
                 waypoint_type=WaypointType.inspect),
        Waypoint("home",    x_m=0.0, y_m=0.0, heading_rad=0.0,
                 waypoint_type=WaypointType.dock),
    ]
    plan = build_nav_plan("factory_run_01", waypoints)
    issues = validate_nav_plan(plan)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum


class WaypointType(str, Enum):
    navigate = "navigate"   # standard drive-to point
    pause = "pause"         # stop and wait for operator clearance
    dock = "dock"           # docking / charging station
    inspect = "inspect"     # camera / sensor observation point


@dataclass(frozen=True)
class Waypoint:
    """A 2D navigation waypoint with heading and type metadata.

    Attributes:
        label: Unique human-readable name.
        x_m: X position in the map frame (metres).
        y_m: Y position in the map frame (metres).
        heading_rad: Target heading at arrival (radians, ROS REP-103 convention).
        waypoint_type: Semantic type — affects behaviour at the waypoint.
        tolerance_m: Arrival radius for position goal acceptance (metres).
    """

    label: str
    x_m: float
    y_m: float
    heading_rad: float = 0.0
    waypoint_type: WaypointType = WaypointType.navigate
    tolerance_m: float = 0.10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NavPlanSegment:
    """Geometric description of one leg between two waypoints."""

    from_label: str
    to_label: str
    distance_m: float
    heading_rad: float

    def heading_deg(self) -> float:
        return round(math.degrees(self.heading_rad), 2)


@dataclass(frozen=True)
class NavigationPlan:
    """An ordered waypoint sequence with precomputed path geometry.

    Attributes:
        name: Plan identifier.
        waypoints: Ordered waypoints.
        segments: Geometric segments connecting consecutive waypoints.
        total_distance_m: Sum of all segment distances.
    """

    name: str
    waypoints: list[Waypoint]
    segments: list[NavPlanSegment]
    total_distance_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_distance_m": self.total_distance_m,
            "n_waypoints": len(self.waypoints),
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "segments": [asdict(seg) for seg in self.segments],
        }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _distance(a: Waypoint, b: Waypoint) -> float:
    return math.sqrt((b.x_m - a.x_m) ** 2 + (b.y_m - a.y_m) ** 2)


def _heading(a: Waypoint, b: Waypoint) -> float:
    return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)


def build_nav_plan(
    name: str,
    waypoints: Sequence[Waypoint],
) -> NavigationPlan:
    """Build a NavigationPlan from an ordered sequence of Waypoints.

    Computes per-segment distance and heading automatically.

    Args:
        name: Human-readable plan name.
        waypoints: Ordered sequence of at least one Waypoint.

    Raises:
        ValueError: When ``waypoints`` is empty.
    """
    wps = list(waypoints)
    if not wps:
        raise ValueError("build_nav_plan requires at least one waypoint.")

    segments: list[NavPlanSegment] = []
    for a, b in zip(wps, wps[1:], strict=False):
        segments.append(
            NavPlanSegment(
                from_label=a.label,
                to_label=b.label,
                distance_m=round(_distance(a, b), 4),
                heading_rad=round(_heading(a, b), 5),
            )
        )

    total = round(sum(seg.distance_m for seg in segments), 4)
    return NavigationPlan(
        name=name,
        waypoints=wps,
        segments=segments,
        total_distance_m=total,
    )


def validate_nav_plan(
    plan: NavigationPlan,
    max_segment_m: float = 5.0,
) -> list[str]:
    """Validate a NavigationPlan. Returns a list of issues (empty = valid).

    Checks:
      - Plan has at least one waypoint.
      - No duplicate waypoint labels.
      - No segment exceeds ``max_segment_m``.
    """
    issues: list[str] = []
    if not plan.waypoints:
        issues.append("Plan has no waypoints.")
        return issues

    labels = [wp.label for wp in plan.waypoints]
    seen: set[str] = set()
    duplicates: list[str] = []
    for label in labels:
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        issues.append(f"Duplicate waypoint labels: {sorted(set(duplicates))}.")

    for seg in plan.segments:
        if seg.distance_m > max_segment_m:
            issues.append(
                f"Segment {seg.from_label}→{seg.to_label}: "
                f"{seg.distance_m:.2f} m exceeds max {max_segment_m:.1f} m."
            )

    return issues

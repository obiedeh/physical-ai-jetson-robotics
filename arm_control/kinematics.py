"""Forward kinematics and workspace validation for the Synria 6DOF arm.

Link geometry comes directly from the URDF xacro properties in
``ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro``::

    base_height = 0.090 m  (joint_1 Z offset)
    upper_arm   = 0.235 m  (joint_2 → joint_3 offset along Z)
    forearm     = 0.220 m  (joint_3 → joint_4 offset along X)
    wrist       = 0.105 m  (joint_5 Z offset)
    tool        = 0.090 m  (tool0 offset)

The planar FK helper provides a 2D projection useful for reachability
visualisation and pre-flight trajectory checks without running MoveIt 2.
Full 3D FK requires the complete DH table, which should be derived from
vendor hardware evidence before Milestone B bring-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Link lengths (metres) — from URDF xacro properties
# ---------------------------------------------------------------------------

LINK_BASE_HEIGHT_M: float = 0.090
LINK_UPPER_ARM_M: float = 0.235
LINK_FOREARM_M: float = 0.220
LINK_WRIST_M: float = 0.105
LINK_TOOL_M: float = 0.090

# Geometric max reach (all segments co-linear, fully extended).
SYNRIA_GEOMETRIC_MAX_REACH_M: float = (
    LINK_UPPER_ARM_M + LINK_FOREARM_M + LINK_WRIST_M + LINK_TOOL_M
)

# Vendor-stated max reach (650 mm from docs/HARDWARE.md).
SYNRIA_VENDOR_MAX_REACH_M: float = 0.650


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CartesianPose:
    """3D Cartesian pose: XYZ position (m) + Euler angles (rad)."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def distance_from_origin(self) -> float:
        """Euclidean distance from the arm base origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@dataclass(frozen=True)
class WorkspaceCheckResult:
    """Result of a workspace reachability check."""

    reachable: bool
    radial_distance_m: float
    max_reach_m: float
    message: str


@dataclass(frozen=True)
class PlanarFKResult:
    """2D FK result in the arm's sagittal plane (x forward, z up)."""

    x_m: float
    z_m: float
    q2_rad: float
    q3_rad: float
    reach_m: float  # distance from origin in the XZ plane


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def estimate_max_reach_m() -> float:
    """Return the geometric maximum reach from URDF link lengths (metres)."""
    return SYNRIA_GEOMETRIC_MAX_REACH_M


def check_workspace_reachability(
    target: CartesianPose,
    max_reach_m: float | None = None,
    min_reach_m: float = 0.05,
) -> WorkspaceCheckResult:
    """Check whether a Cartesian target is inside the reachable workspace.

    Uses a conservative spherical model: the target must lie within
    [min_reach_m, max_reach_m] of the arm base origin. This guards against
    full-extension singularities and near-base singularities.

    Args:
        target: Cartesian pose to evaluate.
        max_reach_m: Maximum reach override (defaults to geometric max from URDF).
        min_reach_m: Minimum reach (defaults to 0.05 m singularity buffer).
    """
    effective_max = max_reach_m if max_reach_m is not None else SYNRIA_GEOMETRIC_MAX_REACH_M
    r = target.distance_from_origin()

    if r > effective_max:
        return WorkspaceCheckResult(
            reachable=False,
            radial_distance_m=round(r, 4),
            max_reach_m=round(effective_max, 4),
            message=(
                f"Target is {r:.3f} m from base, exceeds max reach "
                f"{effective_max:.3f} m."
            ),
        )
    if r < min_reach_m:
        return WorkspaceCheckResult(
            reachable=False,
            radial_distance_m=round(r, 4),
            max_reach_m=round(effective_max, 4),
            message=(
                f"Target is {r:.3f} m from base, closer than minimum "
                f"reach {min_reach_m:.3f} m (singularity buffer)."
            ),
        )
    return WorkspaceCheckResult(
        reachable=True,
        radial_distance_m=round(r, 4),
        max_reach_m=round(effective_max, 4),
        message=(
            f"Target {r:.3f} m from base — within workspace "
            f"[{min_reach_m:.3f} m, {effective_max:.3f} m]."
        ),
    )


# ---------------------------------------------------------------------------
# Planar forward kinematics
# ---------------------------------------------------------------------------


def forward_kinematics_planar(
    q2_rad: float,
    q3_rad: float,
) -> PlanarFKResult:
    """Approximate 2D forward kinematics for the shoulder/elbow plane.

    Projects joint_2 (shoulder) and joint_3 (elbow) onto the arm's sagittal
    plane (x forward, z up) while treating joint_1 rotation, wrist, and tool
    as zero. Useful for pre-flight reachability checks and trajectory previews
    without a full 3D IK solver.

    Args:
        q2_rad: joint_2 angle (shoulder) in radians.
        q3_rad: joint_3 angle (elbow relative to upper arm) in radians.

    Returns:
        PlanarFKResult with the tool-centre-point XZ position.
    """
    z_shoulder = LINK_BASE_HEIGHT_M

    # Upper arm: joint_2 rotates around the Y axis in the sagittal plane.
    x1 = LINK_UPPER_ARM_M * math.sin(q2_rad)
    z1 = z_shoulder + LINK_UPPER_ARM_M * math.cos(q2_rad)

    # Forearm: joint_3 is cumulative.
    q_total = q2_rad + q3_rad
    x2 = x1 + LINK_FOREARM_M * math.sin(q_total)
    z2 = z1 + LINK_FOREARM_M * math.cos(q_total)

    reach = math.sqrt(x2**2 + z2**2)

    return PlanarFKResult(
        x_m=round(x2, 5),
        z_m=round(z2, 5),
        q2_rad=q2_rad,
        q3_rad=q3_rad,
        reach_m=round(reach, 5),
    )

"""Robotic arm control scaffolding for the Synria 6DOF arm.

Public surface:
    safety     — joint-limit and velocity safety gate
    kinematics — forward kinematics and workspace reachability
    trajectory — joint-space trajectory data structures
    demo       — named poses, canned trajectories, and pick-place sequences
"""

from arm_control.demo import (
    POSE_BOARD_APPROACH,
    POSE_CARRY,
    POSE_HOME,
    STAGING_POSES,
    board_approach_trajectory,
    full_pick_place_sequence,
    home_trajectory,
    staging_zone_trajectory,
    validate_all_demo_trajectories,
)
from arm_control.kinematics import (
    CartesianPose,
    WorkspaceCheckResult,
    check_workspace_reachability,
    estimate_max_reach_m,
    forward_kinematics_planar,
)
from arm_control.safety import (
    ArmSafetyConfig,
    ArmSafetyGate,
    SafetyCheckResult,
    SafetyViolation,
)
from arm_control.trajectory import (
    JointTrajectory,
    JointWaypoint,
    TrajectoryExecutionRecord,
    TrajectoryStatus,
)

__all__ = [
    "ArmSafetyConfig",
    "ArmSafetyGate",
    "SafetyCheckResult",
    "SafetyViolation",
    "CartesianPose",
    "WorkspaceCheckResult",
    "check_workspace_reachability",
    "estimate_max_reach_m",
    "forward_kinematics_planar",
    "JointTrajectory",
    "JointWaypoint",
    "TrajectoryExecutionRecord",
    "TrajectoryStatus",
    "POSE_HOME",
    "POSE_CARRY",
    "POSE_BOARD_APPROACH",
    "STAGING_POSES",
    "home_trajectory",
    "board_approach_trajectory",
    "staging_zone_trajectory",
    "full_pick_place_sequence",
    "validate_all_demo_trajectories",
]

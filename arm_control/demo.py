"""Demo trajectory sequences for the Synria 6DOF arm.

Provides deterministic, validated trajectory sequences useful for:
  - CLI demos and documentation
  - Pre-flight safety gate checks
  - Simulation smoke tests before real hardware motion
  - Isaac Lab policy evaluation reference poses

All trajectories stay within the URDF joint limits defined in
``arm_control/safety.py``. Durations are conservative (half max velocity).

Usage::

    from arm_control.demo import home_trajectory, staging_zone_reach_sequence

    traj = home_trajectory()
    errors = traj.validate()   # should be empty
    print(traj.duration_s())
"""

from __future__ import annotations

from arm_control.safety import ArmSafetyGate
from arm_control.trajectory import JointTrajectory, JointWaypoint

# ---------------------------------------------------------------------------
# Named joint poses (within URDF limits, conservative for bring-up)
# ---------------------------------------------------------------------------

#: Upright rest pose — all joints at zero.
POSE_HOME: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 0.0,
    "joint_5": 0.0,
    "joint_6": 0.0,
}

#: Safe carry pose — arm folded back, clear of table surface.
POSE_CARRY: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 1.0,
    "joint_3": -1.5,
    "joint_4": 0.0,
    "joint_5": -0.5,
    "joint_6": 0.0,
}

#: Pre-grasp approach — arm extended toward the board centre.
POSE_BOARD_APPROACH: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 0.6,
    "joint_3": -1.2,
    "joint_4": 0.0,
    "joint_5": -0.4,
    "joint_6": 0.0,
}

#: Staging zone left — arm swept to the left side of the table.
POSE_STAGING_LEFT: dict[str, float] = {
    "joint_1": 1.2,
    "joint_2": 0.5,
    "joint_3": -1.0,
    "joint_4": 0.0,
    "joint_5": -0.3,
    "joint_6": 0.0,
}

#: Staging zone right — arm swept to the right side of the table.
POSE_STAGING_RIGHT: dict[str, float] = {
    "joint_1": -1.2,
    "joint_2": 0.5,
    "joint_3": -1.0,
    "joint_4": 0.0,
    "joint_5": -0.3,
    "joint_6": 0.0,
}

#: Staging zone top — arm extended away from the robot base.
POSE_STAGING_TOP: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 0.3,
    "joint_3": -0.8,
    "joint_4": 0.0,
    "joint_5": -0.2,
    "joint_6": 0.0,
}

#: Staging zone bottom — arm retracted toward the robot base.
POSE_STAGING_BOTTOM: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 1.2,
    "joint_3": -1.5,
    "joint_4": 0.0,
    "joint_5": -0.6,
    "joint_6": 0.0,
}

# Convenience mapping used by CLI and tests.
STAGING_POSES: dict[str, dict[str, float]] = {
    "left": POSE_STAGING_LEFT,
    "right": POSE_STAGING_RIGHT,
    "top": POSE_STAGING_TOP,
    "bottom": POSE_STAGING_BOTTOM,
}


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------


def home_trajectory(duration_s: float = 3.0) -> JointTrajectory:
    """Return a single-waypoint trajectory that moves to the home pose.

    Used at start-up to establish a known arm configuration before any
    task-specific motion. Safe to run as a first command after e-stop clear.
    """
    return JointTrajectory(
        name="synria_home",
        waypoints=[
            JointWaypoint(
                positions_rad=dict(POSE_HOME),
                time_from_start_s=duration_s,
            )
        ],
    )


def board_approach_trajectory(
    start_pose: dict[str, float] | None = None,
    duration_s: float = 4.0,
) -> JointTrajectory:
    """Move from ``start_pose`` (default = home) to the board approach pose.

    The board approach pose positions the end-effector above the game board
    centre, ready for piece detection and pre-grasp alignment.
    """
    start = start_pose or POSE_HOME
    mid_t = duration_s / 2.0
    return JointTrajectory(
        name="synria_board_approach",
        waypoints=[
            JointWaypoint(
                positions_rad=dict(start),
                time_from_start_s=0.0,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_CARRY),
                time_from_start_s=mid_t,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_BOARD_APPROACH),
                time_from_start_s=duration_s,
            ),
        ],
    )


def staging_zone_trajectory(
    zone: str = "left",
    start_pose: dict[str, float] | None = None,
    duration_s: float = 5.0,
) -> JointTrajectory:
    """Move from ``start_pose`` to a staging zone via the carry pose.

    Args:
        zone: One of "left", "right", "top", "bottom".
        start_pose: Starting joint positions (default = board approach pose).
        duration_s: Total motion duration in seconds.

    Raises:
        ValueError: When ``zone`` is not a recognised staging zone name.
    """
    if zone not in STAGING_POSES:
        raise ValueError(
            f"Unknown staging zone '{zone}'. Choose from: {sorted(STAGING_POSES)}."
        )
    start = start_pose or POSE_BOARD_APPROACH
    target = STAGING_POSES[zone]
    mid_t = duration_s / 2.0
    return JointTrajectory(
        name=f"synria_staging_{zone}",
        waypoints=[
            JointWaypoint(
                positions_rad=dict(start),
                time_from_start_s=0.0,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_CARRY),
                time_from_start_s=mid_t,
            ),
            JointWaypoint(
                positions_rad=dict(target),
                time_from_start_s=duration_s,
            ),
        ],
    )


def full_pick_place_sequence(zone: str = "left") -> list[JointTrajectory]:
    """Return the four-trajectory sequence for a complete pick-place-return cycle.

    Sequence:
        1. Home → board approach
        2. Board approach → staging zone
        3. Staging zone → board approach (return)
        4. Board approach → home

    Args:
        zone: Target staging zone ("left", "right", "top", "bottom").

    Returns:
        List of four validated JointTrajectory objects.
    """
    t1 = board_approach_trajectory(start_pose=POSE_HOME, duration_s=4.0)
    t2 = staging_zone_trajectory(zone=zone, start_pose=POSE_BOARD_APPROACH, duration_s=5.0)
    # Move from staging zone back to board approach (reverse of t2).
    t3_return = JointTrajectory(
        name=f"synria_return_from_{zone}",
        waypoints=[
            JointWaypoint(
                positions_rad=dict(STAGING_POSES[zone]),
                time_from_start_s=0.0,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_CARRY),
                time_from_start_s=2.5,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_BOARD_APPROACH),
                time_from_start_s=5.0,
            ),
        ],
    )
    t4 = JointTrajectory(
        name="synria_board_to_home",
        waypoints=[
            JointWaypoint(
                positions_rad=dict(POSE_BOARD_APPROACH),
                time_from_start_s=0.0,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_CARRY),
                time_from_start_s=2.0,
            ),
            JointWaypoint(
                positions_rad=dict(POSE_HOME),
                time_from_start_s=4.0,
            ),
        ],
    )
    return [t1, t2, t3_return, t4]


def validate_all_demo_trajectories() -> dict[str, list[str]]:
    """Run safety + trajectory validation on all demo sequences.

    Returns a dict of trajectory_name → list of issues.
    Empty issue lists mean the trajectory is safe and structurally valid.
    """
    gate = ArmSafetyGate()
    results: dict[str, list[str]] = {}

    all_trajs: list[JointTrajectory] = [
        home_trajectory(),
        board_approach_trajectory(),
        staging_zone_trajectory("left"),
        staging_zone_trajectory("right"),
        staging_zone_trajectory("top"),
        staging_zone_trajectory("bottom"),
    ]
    all_trajs.extend(full_pick_place_sequence("left"))

    for traj in all_trajs:
        issues = list(traj.validate())
        for wp in traj.waypoints:
            safety_results = gate.check_joint_positions(wp.positions_rad)
            issues.extend(r.message for r in safety_results if not r.passed)
        results[traj.name] = issues

    return results

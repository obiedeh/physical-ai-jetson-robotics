"""Pure-Python MDP computation kernels for the Synria pick-and-place task.

These functions implement the mathematical core of every observation,
reward, and termination term.  They operate on plain Python scalars and
lists — no Isaac Lab, no torch, no GPU.  This makes them:

* **Unit-testable** on any machine (Windows, CI, Linux RTX).
* **Readable** — the algorithm is separated from the tensor plumbing.
* **Composable** — the Isaac Lab MDP wrappers in ``mdp.py`` call these
  functions after extracting tensors from the env.

Naming convention:
* ``obs_*`` — produces an observation vector (returns a list of floats).
* ``rew_*`` — returns a scalar reward signal (float).
* ``check_*`` — returns a boolean condition.

Usage::

    from isaac.isaaclab_tasks.synria_pickplace.mdp_logic import (
        obs_joint_state_flat,
        rew_ee_to_target_distance,
        check_piece_in_zone,
    )
    obs = obs_joint_state_flat(joint_pos=[0.0]*6, joint_vel=[0.0]*6, gripper_m=0.085)
    rew = rew_ee_to_target_distance(ee_xy=(0.1, 0.0), target_xy=(0.3, -0.3))
    in_zone = check_piece_in_zone(piece_xy=(0.3, -0.3), zone_xy=(0.3, -0.3))
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    GRASP_APPROACH_RADIUS_M,
    GRASP_CONFIRM_STEPS,
    GRIPPER_GRASP_WIDTH_M,
    MAX_PIECES,
    PIECE_DROP_THRESHOLD_M,
    PIECE_IN_ZONE_TOLERANCE_M,
    PIECE_RETURNED_TOLERANCE_M,
    TABLE_SURFACE_Z,
    ZONE_NAMES,
)

# ---------------------------------------------------------------------------
# Observation kernels
# ---------------------------------------------------------------------------


def obs_joint_state_flat(
    joint_pos: Sequence[float],
    joint_vel: Sequence[float],
    gripper_m: float,
) -> list[float]:
    """Return flat 13-float joint-state observation vector.

    Order: ``[pos_1, …, pos_6, vel_1, …, vel_6, gripper_width_m]``.

    Args:
        joint_pos: 6 joint positions in radians.
        joint_vel: 6 joint velocities in rad/s.
        gripper_m: Gripper open width in metres [0, 0.085].

    Returns:
        13-element float list matching the ``joint_state_synria`` obs shape.
    """
    return [*joint_pos, *joint_vel, gripper_m]


def obs_target_piece_onehot(piece_idx: int, n: int = MAX_PIECES) -> list[float]:
    """Return a one-hot vector of length ``n`` for the target piece index.

    Args:
        piece_idx: Zero-based index of the target piece (0 ≤ idx < n).
        n: Vector length (defaults to MAX_PIECES = 32).

    Returns:
        List of floats with exactly one ``1.0`` at position ``piece_idx``.

    Raises:
        ValueError: If piece_idx is outside [0, n).
    """
    if not (0 <= piece_idx < n):
        raise ValueError(f"piece_idx={piece_idx} out of [0, {n}).")
    vec = [0.0] * n
    vec[piece_idx] = 1.0
    return vec


def obs_target_zone_onehot(zone_idx: int) -> list[float]:
    """Return a 4-element one-hot vector for the target staging zone.

    Index order: left=0, right=1, top=2, bottom=3.

    Args:
        zone_idx: Zone index in [0, 3].

    Returns:
        4-element float list.

    Raises:
        ValueError: If zone_idx is outside [0, 3].
    """
    n = len(ZONE_NAMES)
    if not (0 <= zone_idx < n):
        raise ValueError(f"zone_idx={zone_idx} out of [0, {n}).")
    vec = [0.0] * n
    vec[zone_idx] = 1.0
    return vec


def obs_ee_pose_flat(
    pos_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> list[float]:
    """Return flat 7-float end-effector pose observation.

    Order: ``[x, y, z, qw, qx, qy, qz]``.

    Args:
        pos_xyz: EE position in world frame (metres).
        quat_wxyz: Unit quaternion orientation (w, x, y, z).
            Defaults to identity.

    Returns:
        7-element float list.
    """
    return [*pos_xyz, *quat_wxyz]


# ---------------------------------------------------------------------------
# Reward kernels
# ---------------------------------------------------------------------------


def rew_ee_to_target_distance(
    ee_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> float:
    """Negative 2-D Euclidean distance from EE to the current target.

    Weight is applied in env_cfg (``weight=-0.5``).  This term provides
    dense shaping toward the phase-dependent target XY.

    Args:
        ee_xy: End-effector world XY position (metres).
        target_xy: Phase-dependent target XY (board piece or zone centre).

    Returns:
        Negative distance in metres (always ≤ 0).
    """
    dx = ee_xy[0] - target_xy[0]
    dy = ee_xy[1] - target_xy[1]
    return -math.sqrt(dx * dx + dy * dy)


def rew_gripper_piece_penalty(
    gripper_xy: tuple[float, float],
    piece_xy: tuple[float, float],
    gripper_open_m: float,
    closed_threshold_m: float = GRIPPER_GRASP_WIDTH_M * 2.0,
) -> float:
    """Penalty proportional to gripper-piece distance when gripper is nearly closed.

    Returns ``0.0`` when the gripper is open (not attempting a grasp).
    Applied with a negative weight in env_cfg.

    Args:
        gripper_xy: Gripper centre world XY (metres).
        piece_xy: Tracked piece world XY (metres).
        gripper_open_m: Current gripper open width (metres).
        closed_threshold_m: Gripper widths at or below this are "grasping".

    Returns:
        Negative 2-D distance when grasping; 0.0 otherwise.
    """
    if gripper_open_m > closed_threshold_m:
        return 0.0
    dx = gripper_xy[0] - piece_xy[0]
    dy = gripper_xy[1] - piece_xy[1]
    return -math.sqrt(dx * dx + dy * dy)


def rew_action_norm(action: Sequence[float]) -> float:
    """L2 norm of the action vector.

    Applied with a small negative weight to discourage large joint jumps.

    Args:
        action: Action vector (e.g. 6 joint deltas + gripper command).

    Returns:
        Non-negative L2 norm.
    """
    return math.sqrt(sum(a * a for a in action))


def rew_grasp_confirmed(consec_grasp_steps: int) -> float:
    """Sparse +1 reward when grasp is confirmed for enough consecutive steps.

    Args:
        consec_grasp_steps: How many consecutive steps both grasp criteria
            (gripper width and piece-gripper distance) have been met.

    Returns:
        ``1.0`` on the exact step where the threshold is crossed; ``0.0``
        otherwise.  The MDP caller is responsible for advancing the phase.
    """
    return 1.0 if consec_grasp_steps == GRASP_CONFIRM_STEPS else 0.0


def rew_piece_in_zone(
    piece_xy: tuple[float, float],
    zone_xy: tuple[float, float],
    tolerance_m: float = PIECE_IN_ZONE_TOLERANCE_M,
) -> float:
    """Sparse +1 reward when the piece is centred within the staging zone.

    Args:
        piece_xy: Current piece world XY (metres).
        zone_xy: Target zone centre world XY (metres).
        tolerance_m: Acceptance radius (metres).

    Returns:
        ``1.0`` if within tolerance; ``0.0`` otherwise.
    """
    dx = piece_xy[0] - zone_xy[0]
    dy = piece_xy[1] - zone_xy[1]
    return 1.0 if math.sqrt(dx * dx + dy * dy) <= tolerance_m else 0.0


def rew_piece_returned_to_origin(
    piece_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    tolerance_m: float = PIECE_RETURNED_TOLERANCE_M,
) -> float:
    """Sparse +1 reward when the piece is returned within tolerance of its start.

    Args:
        piece_xy: Current piece world XY (metres).
        origin_xy: Original board square world XY (metres).
        tolerance_m: Acceptance radius (metres).

    Returns:
        ``1.0`` if within tolerance; ``0.0`` otherwise.
    """
    dx = piece_xy[0] - origin_xy[0]
    dy = piece_xy[1] - origin_xy[1]
    return 1.0 if math.sqrt(dx * dx + dy * dy) <= tolerance_m else 0.0


def rew_piece_dropped(
    piece_z: float,
    drop_threshold_m: float = PIECE_DROP_THRESHOLD_M,
) -> float:
    """``-1.0`` when the piece falls below the table surface.

    Args:
        piece_z: Current piece world Z (metres).
        drop_threshold_m: Distance below table surface to trigger (metres).

    Returns:
        ``-1.0`` if dropped; ``0.0`` otherwise.
    """
    return -1.0 if piece_z < TABLE_SURFACE_Z - drop_threshold_m else 0.0


def rew_arm_collision(is_colliding: bool) -> float:
    """``-1.0`` on any unsafe arm collision.

    Args:
        is_colliding: ``True`` when a contact sensor reports collision with
            anything other than the actively-tracked piece or its zone.

    Returns:
        ``-1.0`` if colliding; ``0.0`` otherwise.
    """
    return -1.0 if is_colliding else 0.0


# ---------------------------------------------------------------------------
# Check / termination kernels
# ---------------------------------------------------------------------------


def check_grasp_criteria(
    gripper_open_m: float,
    gripper_to_piece_dist_m: float,
    grasp_width_m: float = GRIPPER_GRASP_WIDTH_M,
    approach_radius_m: float = GRASP_APPROACH_RADIUS_M,
) -> bool:
    """Return ``True`` when both grasp criteria are simultaneously met.

    A grasp is considered in progress when:
    * Gripper width ≤ ``grasp_width_m`` (fingers nearly closed).
    * Gripper centre-to-piece distance ≤ ``approach_radius_m``.

    Args:
        gripper_open_m: Current gripper open width (metres).
        gripper_to_piece_dist_m: 3-D distance from gripper to piece (metres).
        grasp_width_m: Width threshold for "fingers closed" (metres).
        approach_radius_m: Distance threshold for "piece in gripper" (metres).

    Returns:
        ``True`` if both criteria are met.
    """
    return (
        gripper_open_m <= grasp_width_m
        and gripper_to_piece_dist_m <= approach_radius_m
    )


def check_piece_in_zone(
    piece_xy: tuple[float, float],
    zone_xy: tuple[float, float],
    tolerance_m: float = PIECE_IN_ZONE_TOLERANCE_M,
) -> bool:
    """``True`` when the piece is centred within the target zone."""
    dx = piece_xy[0] - zone_xy[0]
    dy = piece_xy[1] - zone_xy[1]
    return math.sqrt(dx * dx + dy * dy) <= tolerance_m


def check_piece_returned(
    piece_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    tolerance_m: float = PIECE_RETURNED_TOLERANCE_M,
) -> bool:
    """``True`` when the piece is back within tolerance of its start square."""
    dx = piece_xy[0] - origin_xy[0]
    dy = piece_xy[1] - origin_xy[1]
    return math.sqrt(dx * dx + dy * dy) <= tolerance_m


def check_piece_dropped(
    piece_z: float,
    drop_threshold_m: float = PIECE_DROP_THRESHOLD_M,
) -> bool:
    """``True`` when the piece has fallen below the table surface."""
    return piece_z < TABLE_SURFACE_Z - drop_threshold_m


def check_task_success_v1(
    phase_value: int,
    piece_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    tolerance_m: float = PIECE_RETURNED_TOLERANCE_M,
) -> bool:
    """``True`` when V1 task is complete: phase is RETURN_TO_BOARD AND piece is home.

    The calling code (``mdp.py``) should advance to ``TaskPhase.SUCCESS``
    after this returns ``True``.

    Args:
        phase_value: Integer value of the current ``TaskPhase``.
        piece_xy: Current piece world XY.
        origin_xy: Original board square world XY.
        tolerance_m: Acceptance radius.

    Returns:
        ``True`` when the task sequence is complete.
    """
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    return (
        phase_value == TaskPhase.RETURN_TO_BOARD
        and check_piece_returned(piece_xy, origin_xy, tolerance_m)
    )


# ---------------------------------------------------------------------------
# Batch helpers (operate on per-env lists — used by mdp.py wrappers)
# ---------------------------------------------------------------------------


def batch_ee_to_target_distance(
    ee_xys: list[tuple[float, float]],
    target_xys: list[tuple[float, float]],
) -> list[float]:
    """Vectorised ``rew_ee_to_target_distance`` over a list of envs."""
    return [rew_ee_to_target_distance(e, t) for e, t in zip(ee_xys, target_xys, strict=False)]


def batch_piece_in_zone(
    piece_xys: list[tuple[float, float]],
    zone_xys: list[tuple[float, float]],
    tolerance_m: float = PIECE_IN_ZONE_TOLERANCE_M,
) -> list[float]:
    """Vectorised ``rew_piece_in_zone``."""
    return [rew_piece_in_zone(p, z, tolerance_m) for p, z in zip(piece_xys, zone_xys, strict=False)]


def batch_piece_returned(
    piece_xys: list[tuple[float, float]],
    origin_xys: list[tuple[float, float]],
    tolerance_m: float = PIECE_RETURNED_TOLERANCE_M,
) -> list[float]:
    """Vectorised ``rew_piece_returned_to_origin``."""
    return [
        rew_piece_returned_to_origin(p, o, tolerance_m)
        for p, o in zip(piece_xys, origin_xys, strict=False)
    ]


def batch_piece_dropped(
    piece_zs: list[float],
    drop_threshold_m: float = PIECE_DROP_THRESHOLD_M,
) -> list[float]:
    """Vectorised ``rew_piece_dropped``."""
    return [rew_piece_dropped(z, drop_threshold_m) for z in piece_zs]

"""Board and staging-zone geometry for the Synria pick-and-place task.

All constants are derived directly from
``isaac/scripts/_synria_scene_common.py`` so that env configs and MDP
logic always agree with the actual USD scene layout.

This module is pure Python — no Isaac Lab or torch dependency.  It can
therefore be imported and tested on any machine, including Windows dev
and CI runners.

Usage::

    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        zone_center_xy, piece_start_xy, BOARD_SIZES, PIECE_COUNTS,
    )
    cx, cy = zone_center_xy("left", "chess")
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Scene layout (mirrors _synria_scene_common.py)
# ---------------------------------------------------------------------------

#: Height of the table surface in world Z (metres).
TABLE_SURFACE_Z: float = 0.79           # TABLE_HEIGHT (0.75) + TABLE_THICKNESS (0.04)

#: Arm base XYZ in world frame.
ARM_BASE_X: float = -0.50
ARM_BASE_Y: float = 0.0
ARM_BASE_Z: float = TABLE_SURFACE_Z

#: Board centre in world XY.
BOARD_CENTER_X: float = 0.10
BOARD_CENTER_Y: float = 0.0

#: Staging pad geometry (metres).
STAGING_ZONE_OFFSET: float = 0.06      # inner edge distance from board edge
STAGING_ZONE_DEPTH: float = 0.10       # pad depth perpendicular to board edge
STAGING_ZONE_PAD_Z: float = TABLE_SURFACE_Z + 0.0006   # centre of 1.2 mm pad

#: Training piece proxy dimensions (metres) — a pawn-sized cylinder used as
#: the physical target piece until per-game piece USDs are wired in.
PIECE_RADIUS_M: float = 0.02
PIECE_HEIGHT_M: float = 0.05

#: Zone index order (matches one-hot encoding in the observation space).
ZONE_NAMES: tuple[str, ...] = ("left", "right", "top", "bottom")
ZONE_INDEX: dict[str, int] = {name: i for i, name in enumerate(ZONE_NAMES)}

# ---------------------------------------------------------------------------
# Per-game constants
# ---------------------------------------------------------------------------

#: Square (symmetrical) board side lengths in metres.
BOARD_SIZES: dict[str, float] = {
    "chess":    0.44,   # 8×8 squares, ~55 mm each
    "checkers": 0.44,   # 8×8 squares
    "ludo":     0.44,   # 15×15 grid with 44 mm squares (outer edge ~0.44 m)
}

#: Total number of movable pieces per game.
PIECE_COUNTS: dict[str, int] = {
    "chess":    32,   # 16 white + 16 black
    "checkers": 24,   # 12 white + 12 black
    "ludo":     16,   # 4 tokens × 4 colours
}

#: Number of squares per side (determines square-centre spacing).
BOARD_GRID: dict[str, int] = {
    "chess":    8,
    "checkers": 8,
    "ludo":     15,
}

#: Maximum piece count across all games (sets the one-hot vector width).
MAX_PIECES: int = max(PIECE_COUNTS.values())  # 32

# ---------------------------------------------------------------------------
# Task acceptance tolerances
# ---------------------------------------------------------------------------

#: Trial is aborted when the piece slides farther than this from the board
#: centre (m) — the proxy-world equivalent of "piece fell off the table".
#: Covers the board (0.44 m) plus staging zones with margin.
#: 0.80, was 0.60: pieces dropped at the zones (0.35 m out) bounce and
#: roll; at 0.60 the roll-out killed 73% of episodes right after the first
#: successful set-down — the policy could never cycle twice.
PIECE_OUT_OF_BOUNDS_RADIUS_M: float = 0.80

#: Set-down acceptance radius (m): wider than the in-zone tolerance because
#: a piece released from grasp height bounces/rolls a little — demanding a
#: 3 cm landing made most genuine drops score nothing.
#: 0.08, was 0.05: funnel telemetry (A/B run, both arms) showed docking at
#: 0.3-0.7% of episodes but set-downs stuck at 0.01% — the dock funnel
#: attracts within 15 cm but the +30 and release shaping only paid inside
#: 5 cm, a precision ask beyond exploration noise (std 0.26). Re-tighten
#: during the queued "clean drop" goal once cycling is established.
SETDOWN_TOLERANCE_M: float = 0.08

# ---------------------------------------------------------------------------
# Carry-around sequence task (user spec 2026-07-28): pick -> carry the cup
# around for 5 s -> gentle set-down -> release -> return to reset pose.
# ---------------------------------------------------------------------------

#: Steps the piece must be held aloft in the CARRY phase (5 s at 30 Hz).
CARRY_DURATION_STEPS: int = 150

#: E15 graduated placement gate: the E14 strict gate (0.08 m + 15 deg
#: upright, all-at-once) was unreachable — training setdowns fell 2.2%
#: -> 0.5% and the eval read 0.000. Start wide/lenient so the skill can
#: form, then TIGHTEN toward spec (0.08 m / 0.966) once the eval shows
#: target placements above ~20%.
PLACEMENT_RADIUS_M: float = 0.15
UPRIGHT_COS_GATE: float = 0.87   # ~30 deg

#: Cumulative XY path the held piece must travel during the carry ("move it
#: around", not just hover): 20 cm of wandering.
MIN_CARRY_PATH_M: float = 0.20

#: Max piece speed (m/s) at the moment of the resting set-down for it to
#: count as GENTLE — placed, not dropped.
GENTLE_SETDOWN_MAX_VEL: float = 0.10

#: Arm joint-space L2 distance (rad, over the 6 arm joints) to the default
#: pose that counts as "returned to reset".
#: 1.2, was 0.5: with exploration noise std ~1.05 the arm's joint L2
#: fluctuates ~0.5-1.0 rad around any pose — the 0.5 gate sat below the
#: noise floor and the return event never fired once in 4k iterations
#: (return income negative: pure drift). Tighten during the polish goal
#: as the entropy anneal collapses the noise.
RETURN_POSE_TOL_RAD: float = 1.2

#: Fraction of resets staged mid-carry (piece in hand, carry clock nearly
#: complete) — mass-produces set-down/return practice, the same curriculum
#: trick that installed the release.
CARRY_ELAPSED_FRACTION: float = 0.15  # E9: reliable under attach — release practice

#: E3: fraction of resets staged POST-PLACEMENT (cup resting at its
#: origin, phase RETURN) — the episode opens one homing motion (zero
#: action under use_default_offset) from the +30 sequence bonus. Only
#: ~10% of natural episodes ever reach RETURN, each briefly, so the
#: terminal leg gets almost no credit mass; staging manufactures it,
#: the same trick that installed the release.
# E9: retry under attach (E3's failure context was pre-attach physics)
RETURN_START_FRACTION: float = 0.10

#: Piece must be within this radius of zone centre to count as "placed" (m).
#: 3 cm ≈ half a board square — achievable; the old 1 cm demanded sub-square
#: precision from a 40 mm piece.
PIECE_IN_ZONE_TOLERANCE_M: float = 0.03

#: Piece must be within this radius of its origin to count as "returned" (m).
PIECE_RETURNED_TOLERANCE_M: float = 0.03

#: Piece Z below this threshold → dropped (m, relative to table surface).
PIECE_DROP_THRESHOLD_M: float = 0.05

#: Gripper width at or below this value is considered "grasping" (m).
#: MUST exceed the piece diameter (2 × PIECE_RADIUS_M = 0.04): fingers
#: closing onto the piece stall at piece diameter, so a threshold below it
#: (the old 0.015) makes grasp confirmation physically impossible.
GRIPPER_GRASP_WIDTH_M: float = 0.045

#: Fully-open gripper width (m). The URDF fingers are PRISMATIC with 0.025 m
#: travel each (left 0..0.025, right -0.025..0) → max true width 0.05 m.
#: (The old 0.085 came from a servo-gripper spec that does not match the URDF.)
GRIPPER_OPEN_M: float = 0.05

#: DEPRECATED — the fingers are prismatic, not angular. Kept only for legacy
#: callers; mdp.py computes width directly from the two finger joints.
GRIPPER_JOINT_RANGE_RAD: float = 0.5

#: EE must be within this distance of the piece to count a grasp approach (m).
#: Measured from the PAD CENTRE (gripper_geometry.pad_center_world) since
#: 2026-08-04. It was previously measured from the tool0 FLANGE, which sits
#: ~10-13 cm above the pads: a piece held mid-pad read 0.10-0.13 m away, so
#: 0.08 rejected genuine in-hand pieces and grasp income flatlined at zero
#: even in staged in-hand curriculum episodes. 0.15 was chosen to absorb that
#: flange offset.
#:
#: Deliberately NOT re-tightened when the reference point moved. From the pad
#: centre 0.15 is loose, but the radius is redundant safety on top of the
#: unfakeable airborne + holding-band criteria, and it is not the binding
#: term — the lift check is. Tightening it would silently redefine the
#: grasp-success metric mid-programme and break comparability with the E1-E22
#: numbers in SYNRIA_EXPERIMENT_LEDGER.md. Tighten it only as a deliberate,
#: recorded gate change with a fresh baseline eval.
GRASP_APPROACH_RADIUS_M: float = 0.15

#: Gripper width BELOW this is an empty pinch (m). Grasp confirmation
#: requires width in [this, GRIPPER_GRASP_WIDTH_M] — the "holding band" —
#: so pinching air can never satisfy the criteria. Calibrated from
#: diag_grasp_feasibility.py: a REAL squeezed hold reads ~10 mm (the box
#: pads penetrate the piece under sustained drive force), while an empty
#: pinch closes to ~0.1 mm — so the floor sits between the two.
GRIPPER_HOLDING_WIDTH_MIN_M: float = 0.004

#: Piece must be at least this far above its rest height to count as
#: lifted (m). Unfakeable grasp evidence: the piece only leaves the table
#: if the gripper is actually holding it.
PIECE_LIFT_THRESHOLD_M: float = 0.02

#: Pad-collider centre in each finger LINK-LOCAL frame (m), read from the
#: USD (`reports/gripper_phase1_composition.json`, prim
#: `.../[left|right]_gripper/pad_collider`, xformOp:translate).
#:
#: The fingers extend DIAGONALLY from their link frames, so the pad centre
#: is NOT the link origin: measured at the authored pose the pad centre sits
#: 21.2 mm behind (-x) and 21.2 mm above (+z) the finger frame — a **30.0 mm**
#: offset in the approach plane. Averaged over both fingers the y components
#: cancel, so the frame midpoint is a correct CLOSING-AXIS centre but is
#: wrong by (-21.2, 0, +21.2) mm in the approach plane.
#:
#: That error is larger than the whole clearance budget for the 40 mm cup
#: (5 mm per side), so any reward or gate that treats the finger-frame
#: midpoint as the grasp point aims the hand at a pose where the piece is
#: not between the pads. Apply these with the finger link's world rotation
#: (see mdp._grasp_center_local).
#: Lateral gate for the grasp-height reward (m). Height error only shapes
#: behaviour once the hand is actually over the cup; outside this radius the
#: term pays nothing, so it cannot be farmed on the way in. 6 cm is a little
#: wider than the cup radius (2 cm) plus the pad half-face (2.17 cm), so the
#: gate opens just as a grasp becomes geometrically possible.
GRASP_HEIGHT_XY_GATE_M: float = 0.06

PAD_LOCAL_OFFSET_M: dict = {
    "left": (0.0, -0.030, -0.01375),
    "right": (0.0, 0.030, 0.01375),
}

# ---------------------------------------------------------------------------
# Pre-grasped curriculum (reset-time)
# ---------------------------------------------------------------------------

#: Fraction of episode resets that start with the piece already in the hand
#: (phase = PLACE_ON_ZONE). Discovering descend+squeeze+lift from scratch is
#: a needle-in-a-haystack for PPO; starting some episodes post-grasp lets the
#: carry/place behaviour train while the value function learns that held
#: states are valuable, pulling the pick behaviour in from the other side.
PREGRASP_FRACTION: float = 0.20  # E9: reliable under attach — carry practice

#: Fraction of resets that start with the piece standing ON its assigned
#: staging zone with the trial in PICK_FROM_ZONE — trains the second half of
#: the cycle (re-pick at zone -> carry home) directly.
ZONE_START_FRACTION: float = 0.0  # E5

#: Fraction of resets that start with the piece in hand at the rendezvous
#: and the trial zone sampled ALONG THE LINE from the rendezvous toward the
#: real zone plate (t ~ U(0,1)): t~0 is docked release practice (the trick
#: that installed the let-go — v16d: setdowns 0.01% -> 13% within 2k
#: iterations), t~1 is a full carry, and the continuum in between builds
#: the transport leg incrementally — the last unlearned skill (v16d
#: plateau: grasp 58% but zone arrivals ~13% ~= the docked freebies).
#: After the cycle fires, advance_cycle flips to a real zone plate.
#: 0.25, was 0.12: this slice now teaches the critical remaining skill.
DOCKED_START_FRACTION: float = 0.25

#: Zones reachable from the training arm mount at (-0.22, 0): "top" sits
#: 2 cm from the arm's own base column and "bottom" is 0.66 m away (beyond
#: the ~0.6 m reach) — trials targeting them are unwinnable by geometry.
FEASIBLE_ZONE_IDS: tuple[int, ...] = (0, 1)  # left, right

#: Env-local grasp centre (midpoint between the finger pads) at the DEFAULT
#: reset pose — measured by `diag_grasp_feasibility.py --no-raise`.
PREGRASP_CENTER_LOCAL: tuple[float, float, float] = (0.0683, 0.0002, 0.8557)

#: Pre-grasped finger joint position (m): ±19.5 mm -> 39 mm width around the
#: 40 mm piece — 0.5 mm interference per pad, a mild initial grip the policy
#: must learn to maintain by continuing to squeeze.
PREGRASP_FINGER_POS_M: float = 0.0195

#: Consecutive steps of criteria to trigger "grasp confirmed". 1, not 3:
#: the criteria now include the piece being physically AIRBORNE in the
#: holding band — unfakeable in a single step — and the 3-step window lost
#: a race at curriculum resets (the zeroed action buffer commands open
#: fingers, so a staged in-hand piece slips out before step 3).
GRASP_CONFIRM_STEPS: int = 1

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def zone_center_xy(zone: str, game: str) -> tuple[float, float]:
    """Return world (x, y) of a staging zone centre.

    Derived from ``_synria_scene_common.add_staging_zones`` with::

        half_board = board_size / 2
        edge_clearance = half_board + STAGING_ZONE_OFFSET + STAGING_ZONE_DEPTH / 2

    Args:
        zone: One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
        game: One of ``"chess"``, ``"checkers"``, ``"ludo"``.

    Returns:
        ``(x_world, y_world)`` in metres.

    Raises:
        ValueError: For unknown zone or game names.
    """
    if zone not in ZONE_INDEX:
        raise ValueError(f"Unknown zone={zone!r}. Valid: {ZONE_NAMES}")
    if game not in BOARD_SIZES:
        raise ValueError(f"Unknown game={game!r}. Valid: {list(BOARD_SIZES)}")
    half_board = BOARD_SIZES[game] / 2.0
    edge_clearance = half_board + STAGING_ZONE_OFFSET + STAGING_ZONE_DEPTH / 2.0
    centers: dict[str, tuple[float, float]] = {
        "left":   (BOARD_CENTER_X,                  BOARD_CENTER_Y - edge_clearance),
        "right":  (BOARD_CENTER_X,                  BOARD_CENTER_Y + edge_clearance),
        "top":    (BOARD_CENTER_X - edge_clearance, BOARD_CENTER_Y),
        "bottom": (BOARD_CENTER_X + edge_clearance, BOARD_CENTER_Y),
    }
    return centers[zone]


def square_center_xy(game: str, row: int, col: int) -> tuple[float, float]:
    """Return world (x, y) of a board square centre.

    The coordinate frame:
    * ``row = 0`` is the row closest to the arm base (−X of board centre).
    * ``col = 0`` is the operator's left edge of the board (−Y of board centre).

    Args:
        game: Board game name.
        row: 0-indexed row (0 = near arm).
        col: 0-indexed column (0 = operator's left).

    Returns:
        ``(x_world, y_world)`` in metres.
    """
    board_size = BOARD_SIZES[game]
    n = BOARD_GRID[game]
    sq = board_size / n
    # Row 0 is at -X of board centre (toward arm); increasing row → +X.
    x = BOARD_CENTER_X - board_size / 2.0 + sq * (row + 0.5)
    y = BOARD_CENTER_Y - board_size / 2.0 + sq * (col + 0.5)
    return (round(x, 5), round(y, 5))


def l2_distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """2-D Euclidean distance (metres)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def l2_distance_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """3-D Euclidean distance (metres)."""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


# ---------------------------------------------------------------------------
# Initial piece positions per game
# ---------------------------------------------------------------------------
#
# Each entry is a list of (row, col) tuples in the _square_center_xy_ frame
# (row 0 = near arm, col 0 = operator's left). Index in the list is the
# piece's `target_piece_idx` used by the TrialStateManager.
#


def _chess_initial_positions() -> list[tuple[int, int]]:
    """All 32 chess pieces in standard opening order.

    Indices 0-15 = white, 16-31 = black.
    White is positioned on rows closest to the arm (rows 0-1).
    """
    positions: list[tuple[int, int]] = []
    # White back rank (row 0): R N B Q K B N R
    for col in range(8):
        positions.append((0, col))
    # White pawns (row 1)
    for col in range(8):
        positions.append((1, col))
    # Black pawns (row 6)
    for col in range(8):
        positions.append((6, col))
    # Black back rank (row 7): R N B Q K B N R
    for col in range(8):
        positions.append((7, col))
    return positions


def _checkers_initial_positions() -> list[tuple[int, int]]:
    """24 checker discs on dark squares of their starting rows.

    Indices 0-11 = white (rows 0-2, near arm), 12-23 = black (rows 5-7).
    Dark squares: squares where (row + col) is odd.
    """
    positions: list[tuple[int, int]] = []
    for row in range(3):
        for col in range(8):
            if (row + col) % 2 == 1:
                positions.append((row, col))
    for row in range(5, 8):
        for col in range(8):
            if (row + col) % 2 == 1:
                positions.append((row, col))
    return positions


def _ludo_initial_positions() -> list[tuple[int, int]]:
    """16 Ludo tokens in their home quadrants.

    Standard Ludo home positions (simplified to a 15×15 grid).
    Indices 0-3 = red, 4-7 = blue, 8-11 = yellow, 12-15 = green.
    """
    # Approximate home quadrant centres on a 15×15 grid:
    # Red (near arm, left): rows 1-2, cols 1-2
    # Blue (near arm, right): rows 1-2, cols 12-13
    # Yellow (far arm, left): rows 12-13, cols 1-2
    # Green (far arm, right): rows 12-13, cols 12-13
    home_squares = [
        (1, 1), (1, 2), (2, 1), (2, 2),          # red
        (1, 12), (1, 13), (2, 12), (2, 13),       # blue
        (12, 1), (12, 2), (13, 1), (13, 2),       # yellow
        (12, 12), (12, 13), (13, 12), (13, 13),   # green
    ]
    return home_squares


#: Initial (row, col) positions for all pieces in each game.
INITIAL_POSITIONS: dict[str, list[tuple[int, int]]] = {
    "chess":    _chess_initial_positions(),
    "checkers": _checkers_initial_positions(),
    "ludo":     _ludo_initial_positions(),
}


def piece_start_xy(game: str, piece_idx: int) -> tuple[float, float]:
    """Return world (x, y) of a piece at its episode-start position.

    Args:
        game: Board game name.
        piece_idx: Zero-based index into the game's piece list.

    Returns:
        ``(x_world, y_world)`` in metres.

    Raises:
        IndexError: If piece_idx is out of range for the game.
    """
    positions = INITIAL_POSITIONS[game]
    if not (0 <= piece_idx < len(positions)):
        raise IndexError(
            f"piece_idx={piece_idx} out of range [0, {len(positions)}) "
            f"for game={game!r}."
        )
    row, col = positions[piece_idx]
    return square_center_xy(game, row, col)


# ---------------------------------------------------------------------------
# Graspable-workspace spawn feasibility (spawn-bin verification 2026-07-28)
# ---------------------------------------------------------------------------

#: Training arm mount in env-local XY (matches env_cfg).
TRAIN_ARM_BASE_XY: tuple[float, float] = (-0.22, 0.0)

#: Spawns beyond this radius from the arm base are outside the TOP-DOWN
#: GRASP envelope: the fixed harness measured grasp 83.7% for spawns at
#: 0.13-0.27 m vs 1.4% at 0.46-0.55 m (E5_final, 264 episodes). Half of
#: all trials were geometrically unwinnable — same failure class as the
#: infeasible zones (FEASIBLE_ZONE_IDS). Revisit if the arm is
#: re-mounted closer to the board.
GRASPABLE_SPAWN_RADIUS_M: float = 0.35

#: Piece indices whose start square lies inside the graspable envelope.
FEASIBLE_PIECE_IDS: dict[str, tuple[int, ...]] = {
    _g: tuple(
        _i
        for _i in range(PIECE_COUNTS[_g])
        if math.hypot(
            piece_start_xy(_g, _i)[0] - TRAIN_ARM_BASE_XY[0],
            piece_start_xy(_g, _i)[1] - TRAIN_ARM_BASE_XY[1],
        )
        <= GRASPABLE_SPAWN_RADIUS_M
    )
    for _g in PIECE_COUNTS
}

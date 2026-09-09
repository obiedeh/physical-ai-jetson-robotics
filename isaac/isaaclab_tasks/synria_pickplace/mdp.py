"""MDP term functions for the Synria board ↔ staging pick-and-place task.

These functions are called by Isaac Lab's manager system every step.
Each function has the correct Isaac Lab signature and a full
implementation that calls the pure-Python kernels in ``mdp_logic.py``.

Isaac Lab runtime path (RTX 5090):

    env.scene["robot"]   → ArticulationView  → joint states, body positions
    env.scene["table"]   → RigidObject       → table surface Z confirmation
    env.extras["trial_state"]  → TrialStateManager  → per-env phase + targets
    env.extras["piece_pos_w"]  → (num_envs, 3) tensor → tracked piece position

Env setup checklist (see README.md for the full list):

1. Wire ``TrialStateManager`` in the env's ``__post_init__`` and store in
   ``env.extras["trial_state"]``.
2. At episode reset, update ``env.extras["piece_pos_w"]`` with the
   initial piece position for each reset env.
3. Every sim step, update ``env.extras["piece_pos_w"]`` by reading the
   piece rigid-body position from the scene.
4. Confirm the arm URDF joint names (``joint_1``..``joint_6``) and body
   name (``tool0``) against the imported USD before first run.

The module is import-safe without Isaac Lab: when torch is absent the
term function bodies are unreachable but the module still parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import-not-found]

from isaac.isaaclab_tasks.synria_pickplace import gripper_geometry
from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    BOARD_CENTER_X,
    BOARD_CENTER_Y,
    CARRY_DURATION_STEPS,
    CARRY_ELAPSED_FRACTION,
    FEASIBLE_PIECE_IDS,
    FEASIBLE_ZONE_IDS,
    GENTLE_SETDOWN_MAX_VEL,
    GRASP_APPROACH_RADIUS_M,
    GRASP_CONFIRM_STEPS,
    GRASP_HEIGHT_XY_GATE_M,
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_HOLDING_WIDTH_MIN_M,
    GRIPPER_OPEN_M,
    MAX_PIECES,
    MIN_CARRY_PATH_M,
    PIECE_COUNTS,
    PIECE_DROP_THRESHOLD_M,
    PIECE_HEIGHT_M,
    PIECE_LIFT_THRESHOLD_M,
    PIECE_OUT_OF_BOUNDS_RADIUS_M,
    PIECE_RETURNED_TOLERANCE_M,
    PLACEMENT_RADIUS_M,
    PREGRASP_CENTER_LOCAL,
    PREGRASP_FINGER_POS_M,
    PREGRASP_FRACTION,
    RETURN_POSE_TOL_RAD,
    RETURN_START_FRACTION,
    SETDOWN_TOLERANCE_M,
    TABLE_SURFACE_Z,
    UPRIGHT_COS_GATE,
    ZONE_NAMES,
    ZONE_START_FRACTION,
    piece_start_xy,
    zone_center_xy,
)
from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    PAD_LOCAL_OFFSET_M as PAD_LOCAL_OFFSET_M,
)
from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TOOL0_BODY_NAME = "tool0"     # body name in the Synria 6DOF arm USD


class VecTrialState:
    """GPU-tensor trial state for all envs.

    Replaces the per-env Python ``TrialStateManager`` on the hot path: the
    CPU manager needed per-env loops with a device sync per element
    (``float(tensor[i])``) in every reward/obs term — unusable at 4096 envs.
    (``trial_state.TrialStateManager`` remains for pure-Python unit tests.)

    Phase values match :class:`TaskPhase` (0 PICK_FROM_BOARD, 1 PLACE_ON_ZONE,
    2 PICK_FROM_ZONE, 3 RETURN_TO_BOARD, 4 SUCCESS).
    """

    def __init__(self, num_envs: int, game: str, device: str | torch.device) -> None:
        self.game = game
        piece_count = PIECE_COUNTS[game]
        self.piece_origins = torch.tensor(
            [piece_start_xy(game, i) for i in range(piece_count)],
            dtype=torch.float32, device=device,
        )  # (P, 2)
        self.zone_centers = torch.tensor(
            [zone_center_xy(z, game) for z in ZONE_NAMES],
            dtype=torch.float32, device=device,
        )  # (4, 2)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.piece_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.zone_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.origin_xy = torch.zeros((num_envs, 2), device=device)
        self.zone_xy = torch.zeros((num_envs, 2), device=device)
        self.consec_grasp = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.carried_aloft = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.carry_prev_dist = torch.full((num_envs,), -1.0, device=device)
        self.pin_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pin_pos = torch.zeros((num_envs, 3), device=device)
        self.ee_prev_dist = torch.full((num_envs,), -1.0, device=device)
        # E27c: previous |pad-centre z - cup z| during PICK, for the
        # grasp-height progress payment. -1.0 = no reading yet this episode.
        self.gz_prev_err = torch.full((num_envs,), -1.0, device=device)
        # E30: previous cup tilt (rad) during SETDOWN, for the
        # upright-maintenance progress payment. -1.0 = no reading yet.
        self.tilt_prev = torch.full((num_envs,), -1.0, device=device)
        self.prev_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Funnel telemetry: per-episode "ever reached this stage" flags.
        # Logged as Funnel/* rates at reset so the conversion chain
        # (grasp -> hold 2s -> reach zone -> dock low -> set down) is
        # readable straight off the training log instead of inferred
        # from reward incomes.
        self.f_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.f_hold2s = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.f_zone = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.f_dock = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.funnel_rates: dict = {}
        # Dwell clock for the docking attractor (monotone per trial — an
        # oscillating in-and-out of the dock state cannot re-farm it) and
        # the previous-step holding-band flag for the one-shot release event.
        self.dock_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prev_width_band = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Release payment latch. release_at_zone pays ONCE per trial: the
        # event is a deliberate let-go, and there is exactly one of those per
        # cycle. Without this, seed 44 (2026-08-05) farmed it ~9x per episode
        # by oscillating the gripper in the band where the piece is both
        # "lifted" and "low", earning 55x E22's release income while placing
        # the FEWEST pieces of any run on record.
        self.release_paid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Carry-around sequence state: cumulative XY path of the held piece
        # ("move it around"), previous piece XY for the path integral, and
        # the carry-complete funnel flag.
        self.move_dist = torch.zeros(num_envs, device=device)
        self.prev_piece_xy = torch.full((num_envs, 2), float("nan"), device=device)
        self.f_carry5s = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Kinematic attach-on-grasp (E8): while attached, the cup is glued
        # to the hand at its attach-time tool-frame offset and orientation.
        self.attached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.attach_off = torch.zeros((num_envs, 3), device=device)
        self.attach_quat = torch.zeros((num_envs, 4), device=device)
        # E16: previous piece->plate distance for SETDOWN approach progress.
        self.setdown_prev_dist = torch.full((num_envs,), -1.0, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample a fresh trial (random target piece + zone) for env_ids."""
        m = len(env_ids)
        dev = self.phase.device
        # Sample only spawns inside the top-down grasp envelope — the fixed
        # harness measured grasp 84% inside vs 1.4% outside; spawns beyond
        # the envelope are geometrically unwinnable (see FEASIBLE_PIECE_IDS).
        feasible_pieces = torch.tensor(
            FEASIBLE_PIECE_IDS[self.game], dtype=torch.long, device=dev
        )
        self.piece_idx[env_ids] = feasible_pieces[
            torch.randint(len(feasible_pieces), (m,), device=dev)
        ]
        # Sample only geometrically feasible zones — see FEASIBLE_ZONE_IDS.
        feasible = torch.tensor(FEASIBLE_ZONE_IDS, dtype=torch.long, device=dev)
        self.zone_idx[env_ids] = feasible[
            torch.randint(len(FEASIBLE_ZONE_IDS), (m,), device=dev)
        ]
        self.origin_xy[env_ids] = self.piece_origins[self.piece_idx[env_ids]]
        self.zone_xy[env_ids] = self.zone_centers[self.zone_idx[env_ids]]
        self.phase[env_ids] = int(TaskPhase.PICK_FROM_BOARD)
        self.consec_grasp[env_ids] = 0
        self.cycles[env_ids] = 0
        self.carried_aloft[env_ids] = False
        self.hold_steps[env_ids] = 0
        self.carry_prev_dist[env_ids] = -1.0
        self.pin_steps[env_ids] = 0
        self.ee_prev_dist[env_ids] = -1.0
        self.gz_prev_err[env_ids] = -1.0
        self.tilt_prev[env_ids] = -1.0
        self.prev_lifted[env_ids] = False
        self.release_paid[env_ids] = False
        self.f_grasp[env_ids] = False
        self.f_hold2s[env_ids] = False
        self.f_zone[env_ids] = False
        self.f_dock[env_ids] = False
        self.dock_steps[env_ids] = 0
        self.prev_width_band[env_ids] = False
        self.move_dist[env_ids] = 0.0
        self.prev_piece_xy[env_ids] = float("nan")
        self.f_carry5s[env_ids] = False
        self.attached[env_ids] = False
        self.setdown_prev_dist[env_ids] = -1.0

    def advance_cycle(self, fired: torch.Tensor) -> None:
        """Set-down completed for `fired` envs: the drop point becomes the
        new origin, the target flips to the other feasible zone, and the
        machine returns to PICK for the next round."""
        self.cycles = torch.where(fired, self.cycles + 1, self.cycles)
        self.origin_xy = torch.where(
            fired.unsqueeze(-1), self.zone_xy, self.origin_xy
        )
        a, b = FEASIBLE_ZONE_IDS[0], FEASIBLE_ZONE_IDS[1]
        flipped = torch.where(
            self.zone_idx == a,
            torch.full_like(self.zone_idx, b),
            torch.full_like(self.zone_idx, a),
        )
        self.zone_idx = torch.where(fired, flipped, self.zone_idx)
        self.zone_xy = torch.where(
            fired.unsqueeze(-1), self.zone_centers[self.zone_idx], self.zone_xy
        )
        self.phase = torch.where(
            fired, torch.full_like(self.phase, int(TaskPhase.PICK_FROM_BOARD)), self.phase
        )
        self.consec_grasp = torch.where(
            fired, torch.zeros_like(self.consec_grasp), self.consec_grasp
        )
        self.carried_aloft = torch.where(
            fired, torch.zeros_like(self.carried_aloft), self.carried_aloft
        )
        self.release_paid = torch.where(
            fired, torch.zeros_like(self.release_paid), self.release_paid
        )
        self.hold_steps = torch.where(
            fired, torch.zeros_like(self.hold_steps), self.hold_steps
        )
        self.carry_prev_dist = torch.where(
            fired, torch.full_like(self.carry_prev_dist, -1.0), self.carry_prev_dist
        )
        self.dock_steps = torch.where(
            fired, torch.zeros_like(self.dock_steps), self.dock_steps
        )
        self.move_dist = torch.where(
            fired, torch.zeros_like(self.move_dist), self.move_dist
        )

    def target_xy(self) -> torch.Tensor:
        """(num_envs, 2) phase-dependent target (E14: defined placement).

        PICK/RETURN -> the piece's rest/origin position (pick target);
        CARRY/SETDOWN -> the trial's zone plate (the DEFINED placement
        target of the goal spec). Feeds trial_target_pos, so the policy
        SEES where the cup must go.
        """
        board_phases = (self.phase == int(TaskPhase.PICK_FROM_BOARD)) | (
            self.phase == int(TaskPhase.RETURN_TO_BOARD)
        )
        return torch.where(board_phases.unsqueeze(-1), self.origin_xy, self.zone_xy)


def _get_trial_mgr(env: ManagerBasedRLEnv) -> VecTrialState:
    """Return the VecTrialState, lazy-initialising it on first call."""
    mgr = env.extras.get("trial_state")  # type: ignore[attr-defined]
    if mgr is None:
        game = getattr(env.cfg, "_game", "chess")
        mgr = VecTrialState(num_envs=env.num_envs, game=game, device=env.device)  # type: ignore[attr-defined]
        env.extras["trial_state"] = mgr  # type: ignore[attr-defined]
    return mgr  # type: ignore[return-value, no-any-return]


def _robot_ids(env: ManagerBasedRLEnv) -> dict:
    """Resolve and cache joint/body indices BY NAME.

    Positional assumptions (arm = joints 0-5, gripper = joint 6) are fragile:
    Isaac Lab orders joints by USD traversal, not URDF declaration order.
    """
    ids = env.extras.get("synria_robot_ids")  # type: ignore[attr-defined]
    if ids is None:
        robot = env.scene["robot"]  # type: ignore[attr-defined]
        arm_ids = robot.find_joints([f"Joint{i}" for i in range(1, 7)])[0]
        left_id = robot.find_joints(["left_finger"])[0][0]
        right_id = robot.find_joints(["right_finger"])[0][0]
        tool_id = robot.find_bodies(_TOOL0_BODY_NAME)[0][0]
        left_body = robot.find_bodies(["left_gripper"])[0][0]
        right_body = robot.find_bodies(["right_gripper"])[0][0]
        ids = {
            "arm": arm_ids,
            "left": left_id,
            "right": right_id,
            "tool0": tool_id,
            "left_body": left_body,
            "right_body": right_body,
        }
        env.extras["synria_robot_ids"] = ids  # type: ignore[attr-defined]
        # Once per env, on the first MDP call: confirm the pad offsets the
        # reward is about to use still describe the asset that just loaded.
        # A mismatch here does not crash — it trains for hours against geometry
        # that does not describe the robot, and the result reads as a bad
        # reward design. That is how the 30 mm error survived E17-E22 while
        # five separate diagnoses were written for its symptoms.
        gripper_geometry.verify_env_geometry(env)
    return ids  # type: ignore[return-value, no-any-return]


def _tool0_pos_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 3) env-local tool0 (EE) position.

    All task geometry (board squares, staging zones, trial targets) is
    expressed in env-local coordinates, but ``body_pos_w`` is world-frame:
    envs are spaced ``env_spacing`` apart, so comparing world positions to
    task targets is wrong for every env except the one at the world origin.
    """
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    tool_idx = _robot_ids(env)["tool0"]
    return robot.data.body_pos_w[:, tool_idx, :] - env.scene.env_origins  # type: ignore[attr-defined, no-any-return]


def _grasp_center_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 3) env-local GRASP CENTER: midpoint of the two finger PADS.

    The approach-shaping streams originally measured distance from tool0
    (the wrist flange), whose zero point sits 10-13 cm PAST the true grasp
    pose — the pads hang below the flange. Paying tool0-dist -> 0 taught the
    policy to drive the flange down onto the piece and stall at contact
    (traced at 12k iters: hover at ~50 mm flange-to-piece, fingers closing
    on air 74 mm away).

    CORRECTED 2026-08-04. That first fix moved the target from the flange to
    the midpoint of the two finger LINK FRAMES and called it "midpoint of the
    pads" — but those are not the same point. The fingers extend diagonally,
    so each pad collider centre sits 21.2 mm behind (-x) and 21.2 mm above
    (+z) its link frame: a **30.0 mm** residual error in the approach plane,
    verified against the USD (task_geometry.PAD_LOCAL_OFFSET_M) and against
    the runtime articulation.

    That residual is larger than the entire clearance budget for the 40 mm cup
    (5 mm per side). Driving the frame midpoint onto the piece parks the true
    pads 21.2 mm above and 21.2 mm behind it, so the piece is never between
    them — the policy is rewarded for a pose it cannot grasp from. This now
    transforms the measured pad offsets by each finger's world rotation, which
    is the same computation the Phase 5 gauge uses
    (isaac/scripts/synria_grasp_feasibility.py::pad_center_world).
    """
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    ids = _robot_ids(env)
    pad_mid = gripper_geometry.pad_center_world(  # (N, 3) world
        robot, body_index={"left": ids["left_body"], "right": ids["right_body"]}
    )
    return pad_mid - env.scene.env_origins  # type: ignore[attr-defined, no-any-return]


def _get_piece_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 3) env-local piece position.

    Sources, in priority order:
    1. The ``piece`` rigid object in the scene (world pose minus per-env
       origin, so all task-geometry comparisons stay env-local).
    2. ``env.extras["piece_pos_w"]`` — legacy hook for tests/manual driving.
    3. A fixed table-height placeholder. It must NOT be all-zeros — z=0 sits
       below TABLE_SURFACE_Z minus the drop threshold, so
       piece_dropped_below_table_term would fire for every env on step 1.
    """
    try:
        piece = env.scene["piece"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        piece = None
    if piece is not None:
        return piece.data.root_pos_w - env.scene.env_origins  # type: ignore[attr-defined, no-any-return]

    pos = env.extras.get("piece_pos_w")  # type: ignore[attr-defined]
    if pos is None:
        pos = torch.zeros((env.num_envs, 3), device=env.device)  # type: ignore[attr-defined]
        pos[:, 2] = TABLE_SURFACE_Z
        return pos
    return pos  # type: ignore[return-value, no-any-return]


def _gripper_width_m(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 1) gripper open-width tensor in metres.

    The fingers are PRISMATIC: left ∈ [0, 0.025], right ∈ [-0.025, 0], so
    true width = left - right ∈ [0, 0.05]. (The previous angular-servo
    formula ``pos / 0.5 rad × 0.085`` produced a max "width" of ~4 mm and
    made the gripper read as permanently grasping.)
    """
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    ids = _robot_ids(env)
    width = robot.data.joint_pos[:, ids["left"]] - robot.data.joint_pos[:, ids["right"]]
    return width.clamp(0.0, GRIPPER_OPEN_M).unsqueeze(-1)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------


def joint_state_synria(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 13) tensor: 6 joint pos + 6 joint vel + gripper width.

    Tensor layout::

        [joint_1_pos, …, joint_6_pos,
         joint_1_vel, …, joint_6_vel,
         gripper_open_m]

    Isaac Lab API used:
        - ``env.scene["robot"].data.joint_pos``  → (num_envs, n_joints)
        - ``env.scene["robot"].data.joint_vel``  → (num_envs, n_joints)
    """
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    arm_ids = _robot_ids(env)["arm"]
    joint_pos = robot.data.joint_pos[:, arm_ids]  # (N, 6)
    joint_vel = robot.data.joint_vel[:, arm_ids]  # (N, 6)
    gripper = _gripper_width_m(env)                # (N, 1)
    return torch.cat([joint_pos, joint_vel, gripper], dim=-1)  # type: ignore[attr-defined]


def ee_pose_synria(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 7) tensor: EE position (3) + quaternion (4, w first).

    Looks up the ``tool0`` body in the arm articulation.

    DELIBERATELY still tool0, unlike the reward and gate streams (2026-08-04).
    Those were moved onto the pad centre because they encode task criteria that
    are only meaningful at the contact surface. An observation has no such
    obligation — it only has to be sufficient, and it is: pad centre is a rigid
    function of (tool0 pose, finger joint position), and the finger width is
    already observed by ``arm_joint_state``. Rewriting this to the pad centre
    would shift three input dims by the 10-13 cm flange offset and invalidate
    every preserved checkpoint (E15/E16/E18/E22) for transfer, buying a
    representation the network can already compute.

    Isaac Lab API used:
        - ``robot.find_bodies("tool0")``        → body index list
        - ``robot.data.body_pos_w``              → (num_envs, n_bodies, 3)
        - ``robot.data.body_quat_w``             → (num_envs, n_bodies, 4) w,x,y,z
    """
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    tool_idx = _robot_ids(env)["tool0"]
    ee_pos = _tool0_pos_local(env)                     # (N, 3) env-local
    ee_quat = robot.data.body_quat_w[:, tool_idx, :]  # (N, 4)
    return torch.cat([ee_pos, ee_quat], dim=-1)        # type: ignore[attr-defined]


def piece_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 3) env-local piece position observation."""
    return _get_piece_pos(env)


def trial_phase_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 5) one-hot of the trial phase — PICK/PLACE/PICKZ/RETURN/DONE.

    Without this the task is partially unobservable: "piece between my open
    fingers near a surface" is the SAME observation whether the right move
    is close-and-lift (pick) or stay-open-and-withdraw (just set down), so
    the policy hedged between opposite actions in identical states.
    """
    trial_mgr = _get_trial_mgr(env)
    return torch.nn.functional.one_hot(  # type: ignore[attr-defined, no-any-return]
        trial_mgr.phase.clamp(0, 4), num_classes=5
    ).float()


def target_piece_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, MAX_PIECES) one-hot tensor for the target piece.

    Reads ``env.extras["trial_state"]`` for per-env target piece indices.
    """
    trial_mgr = _get_trial_mgr(env)
    return torch.nn.functional.one_hot(  # type: ignore[attr-defined, no-any-return]
        trial_mgr.piece_idx, num_classes=MAX_PIECES
    ).float()


def trial_target_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 4): the ACTUAL metric target the rewards pay toward.

    Layout: [target_x, target_y, dx, dy] — the phase-dependent target
    (piece origin in pick phases, zone centre in carry phases) in env-local
    coordinates, plus the piece->target delta.

    The zone was previously observed only as a one-hot INDEX: the policy
    had to memorize label->coordinates with no direct teaching signal, and
    the carry-gradient curriculum (zone sampled along a line) made the
    label actively wrong. The piece was always observed as a position —
    which is why picking became strong while carrying stayed aimless: the
    destination was literally invisible.
    """
    trial_mgr = _get_trial_mgr(env)
    target = trial_mgr.target_xy()                 # (N, 2)
    piece_xy = _get_piece_pos(env)[:, :2]          # (N, 2)
    return torch.cat([target, target - piece_xy], dim=-1)  # type: ignore[attr-defined]


def target_zone_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs, 4) one-hot tensor for the target staging zone.

    Zone order: left=0, right=1, top=2, bottom=3.
    """
    trial_mgr = _get_trial_mgr(env)
    return torch.nn.functional.one_hot(  # type: ignore[attr-defined, no-any-return]
        trial_mgr.zone_idx, num_classes=len(ZONE_NAMES)
    ).float()


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def piece_to_target_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) closeness reward in (0, 1]: 1 - tanh(xy_dist / 0.15).

    Measures the PIECE (not the EE) against the phase-dependent target:
    * PICK_FROM_BOARD / RETURN_TO_BOARD → board piece origin XY
    * PLACE_ON_ZONE / PICK_FROM_ZONE    → staging zone centre XY

    EE-based carry shaping was exploitable: the policy air-pinched near the
    piece to fake a grasp, then walked its EMPTY hand to the zone and kept
    collecting. Only transporting the piece pays this term now.

    POSITIVE closeness kernel (1 - tanh), not a distance penalty: with
    negative-only shaping, ending the episode early stops the penalty
    clock, so the policy learned to knock the piece out of bounds ASAP
    (94% of episodes). Positive per-step income makes early termination
    forfeit future reward instead.
    """
    # CARRY-AROUND SEQUENCE (user spec 2026-07-28): there is no destination
    # any more — the carry requirement is DURATION (5 s aloft) plus
    # MOVEMENT (>= MIN_CARRY_PATH_M of cumulative XY wandering). This term
    # pays the path integral of the HELD piece's XY motion, capped per
    # step, and advances CARRY -> SETDOWN when both requirements are met.
    # Path-integral income is farm-resistant here because the carry clock
    # (hold-decay in hold_maintenance) and the phase advance bound how long
    # wandering can pay within one cycle.
    trial_mgr = _get_trial_mgr(env)
    piece_xy = _get_piece_pos(env)[:, :2]  # (N, 2) env-local

    in_carry = trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE)
    lifted = _piece_lifted_mask(env)
    active = in_carry & lifted & trial_mgr.carried_aloft
    prev_ok = ~torch.isnan(trial_mgr.prev_piece_xy[:, 0])
    step_dist = torch.where(  # type: ignore[attr-defined]
        active & prev_ok,
        torch.norm(piece_xy - torch.nan_to_num(trial_mgr.prev_piece_xy), dim=-1).clamp(
            0.0, 0.05
        ),
        torch.zeros_like(piece_xy[:, 0]),
    )
    trial_mgr.move_dist = trial_mgr.move_dist + step_dist  # type: ignore[attr-defined]
    trial_mgr.prev_piece_xy = torch.where(  # type: ignore[attr-defined]
        active.unsqueeze(-1), piece_xy, torch.full_like(piece_xy, float("nan"))
    )
    # Carry complete: held 5 s AND wandered enough -> SETDOWN phase.
    carry_done = (
        active
        & (trial_mgr.hold_steps >= CARRY_DURATION_STEPS)
        & (trial_mgr.move_dist >= MIN_CARRY_PATH_M)
    )
    trial_mgr.f_carry5s = trial_mgr.f_carry5s | carry_done  # type: ignore[attr-defined]
    trial_mgr.phase = torch.where(  # type: ignore[attr-defined]
        carry_done,
        torch.full_like(trial_mgr.phase, int(TaskPhase.PICK_FROM_ZONE)),
        trial_mgr.phase,
    )
    return step_dist


def _piece_lifted_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) bool — piece at least PIECE_LIFT_THRESHOLD_M above rest."""
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    return _get_piece_pos(env)[:, 2] >= rest_z + PIECE_LIFT_THRESHOLD_M


def approach_guidance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) small closeness kernel, pick phases — newborn guidance.

    The fully progress/event-gated economy gives a RANDOM policy almost no
    gradient anywhere (v13 scratch flatlined at the catch stage). This
    low-weight positional kernel restores dense direction for beginners;
    at weight ~0.15 its ~4/episode ceiling is negligible against the +49
    cycle chain, so it cannot re-open the loiter farm for a competent
    policy.
    """
    trial_mgr = _get_trial_mgr(env)
    # PICK_FROM_ZONE now means SETDOWN in the carry-around sequence — only
    # the true pick phase gets approach guidance.
    pick = trial_mgr.phase == int(TaskPhase.PICK_FROM_BOARD)
    dist = torch.norm(_grasp_center_local(env) - _get_piece_pos(env), dim=-1)  # type: ignore[attr-defined]
    kernel = 1.0 - torch.tanh(dist / 0.15)  # type: ignore[attr-defined]
    return torch.where(pick, kernel, torch.zeros_like(kernel))  # type: ignore[attr-defined, no-any-return]


def carry_guidance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) small closeness kernel toward the target while carrying
    aloft (not yet arrived) — newborn guidance for the transport leg."""
    # Carry-around sequence: no destination — a small flat aloft bonus
    # while genuinely carrying keeps the newborn gradient alive; the real
    # movement income is the path integral in piece_to_target_distance.
    trial_mgr = _get_trial_mgr(env)
    in_carry = trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE)
    active = in_carry & _piece_lifted_mask(env) & trial_mgr.carried_aloft
    return active.float()  # type: ignore[no-any-return]


def release_at_zone(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) one-shot closeness payment on the RELEASE TRANSITION.

    Fires ONCE per trial, on the step the fingers leave the holding band while
    the carried piece was aloft, low, and inside the placement radius — the
    deliberate let-go the cycle bonus requires, scaled toward the zone centre.
    """
    # ONE-SHOT RELEASE EVENT (v16b redesign): the state-based payment was
    # never visited — funnel showed docking as a momentary swing-through
    # (dock 6.5% of episodes, income ~0), so a payment on the sustained
    # released-at-zone STATE had nothing to bite on. Pay the TRANSITION
    # instead: fingers leave the holding band while the carried piece was
    # aloft, low, near the zone.
    #
    # EXPLOIT FIX 2026-08-05. As shipped this term had neither of the two
    # things its own name and docstring claimed. There was NO zone test at
    # all (an inline comment even conceded "no zone"), and no closeness
    # scaling — it returned a bare 0/1 anywhere in the setdown phase, at
    # weight 90. The stated anti-farming argument was that prev_lifted forces
    # a re-lift, "and doing that inside the set-down radius just completes
    # the cycle" — which only holds if there IS a radius test, and there
    # wasn't. Worse, the gates overlap: a piece is "lifted" above rest+2 cm
    # and "low" below rest+5 cm, so in that 3 cm band it is both at once and
    # the hand can simply oscillate open/closed to re-fire the payment.
    #
    # Seed 44 (synria_chess_pickplace_v4_graspfix_s44) found it: release
    # income 9.335 against E22's 0.170 — 55x — while posting the LOWEST
    # piece_in_zone (0.015) and cycle_complete (0.169) of any run on record,
    # and more than double the mean reward of its sibling seeds. Paid
    # enormously for letting go; placed almost nothing.
    #
    # Three changes: gate on the placement radius (the same one piece_in_zone
    # uses, so "released at the zone" and "in the zone" cannot disagree),
    # scale by closeness as the docstring always promised, and latch to one
    # payment per trial. The latch is what actually closes the farm — there
    # is exactly one deliberate let-go per cycle, so a second payment in the
    # same trial was never anything but an exploit.
    trial_mgr = _get_trial_mgr(env)
    width = _gripper_width_m(env).squeeze(-1)
    released = width >= 0.04
    in_setdown = trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    piece = _get_piece_pos(env)
    piece_z = piece[:, 2]
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    # LOW release only: paying for opening the hand at altitude taught
    # bombing runs — pieces released high bounce and roll out of bounds.
    low = piece_z <= rest_z + 0.05
    dist = torch.norm(piece[:, :2] - trial_mgr.zone_xy, dim=-1)  # type: ignore[attr-defined]
    at_zone = dist <= PLACEMENT_RADIUS_M
    release_event = (
        in_setdown
        & trial_mgr.carried_aloft
        & trial_mgr.prev_width_band
        & trial_mgr.prev_lifted
        & released
        & low
        & at_zone
        & ~trial_mgr.release_paid
    )
    trial_mgr.release_paid = trial_mgr.release_paid | release_event  # type: ignore[attr-defined]
    trial_mgr.prev_width_band = (  # type: ignore[attr-defined]
        (width >= GRIPPER_HOLDING_WIDTH_MIN_M) & (width <= GRIPPER_GRASP_WIDTH_M)
    )
    # Closeness scaling: releasing nearer the zone centre pays more, so the
    # gradient keeps pointing inward instead of flattening at the radius.
    funnel = (1.0 - dist / PLACEMENT_RADIUS_M).clamp(0.0, 1.0)
    return release_event.float() * funnel  # type: ignore[no-any-return]


def lower_at_zone(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) reward for DESCENDING the held piece over the zone.

    Bridges carry-at-altitude to low-release: pays 0 at 10 cm+ altitude and
    1.0 when the held piece is at rest height over the zone. Without this
    breadcrumb the policy's only paying release action (lower-then-open)
    was unreachable — release_at_zone never fired once in 400 iterations.
    """
    # SETDOWN-phase descent shaping. E14: the funnel toward the DEFINED
    # target returns — descending pays most low AND centred over the
    # trial's zone plate.
    trial_mgr = _get_trial_mgr(env)
    in_setdown = trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    held = _piece_lifted_mask(env)
    piece = _get_piece_pos(env)
    piece_z = piece[:, 2]
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    closeness = (1.0 - ((piece_z - rest_z) / 0.10)).clamp(0.0, 1.0)
    dist = torch.norm(piece[:, :2] - trial_mgr.zone_xy, dim=-1)  # type: ignore[attr-defined]
    funnel = (1.0 - dist / 0.30).clamp(0.0, 1.0)
    active = in_setdown & held & trial_mgr.carried_aloft
    # DWELL CLOCK (v16b): funnel telemetry showed docking as a momentary
    # swing-through — nothing paid for STOPPING over the zone, so the held
    # piece transited the dock region without ever dwelling long enough to
    # sample the release. Full pay for the first ~1 s of accumulated dock
    # time, fading to zero by ~2 s. The clock is monotone per trial (reset
    # only on cycle completion or trial reset), so oscillating in and out
    # of the dock state cannot re-farm it: the parking budget is ~+15 per
    # trial at weight 15 against +30 for completing the cycle.
    trial_mgr.dock_steps = torch.where(  # type: ignore[attr-defined]
        active, trial_mgr.dock_steps + 1, trial_mgr.dock_steps
    )
    dwell_decay = (1.0 - (trial_mgr.dock_steps.float() - 30.0) / 30.0).clamp(0.0, 1.0)
    return torch.where(  # type: ignore[attr-defined, no-any-return]
        active,
        funnel * closeness * dwell_decay,
        torch.zeros_like(closeness),
    )


def hold_at_zone_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) -1 while hovering the HELD piece over the zone.

    The arrival-nulled carry income made hovering unpaid; this makes it
    actively costly, so "just keep holding it" loses to "drop it".
    """
    trial_mgr = _get_trial_mgr(env)
    piece_xy = _get_piece_pos(env)[:, :2]
    dist = torch.norm(piece_xy - trial_mgr.zone_xy, dim=-1)  # type: ignore[attr-defined]
    hovering = (
        (trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE))
        & (dist <= SETDOWN_TOLERANCE_M)
        & _piece_lifted_mask(env)
    )
    return -hovering.float()  # type: ignore[no-any-return]


def piece_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) dense lift reward in [0, 1]: piece height above rest.

    Bridges grasp → carry: unfakeable positive income for physically raising
    the piece (scaled to saturate at 10 cm above rest height). An empty hand
    earns nothing; throwing the piece ends the episode via out-of-bounds.

    DECAYS WITH HOLD DURATION: full pay for the first ~1.5 s of a hold,
    linear fade to zero by ~3 s (90 steps at 30 Hz). Unconditional lift
    income taught the policy to LATCH — hold the piece aloft indefinitely
    farming ~45/episode. Lifting must start the clock, not stop it.
    """
    trial_mgr = _get_trial_mgr(env)
    lifted = _piece_lifted_mask(env)
    trial_mgr.hold_steps = torch.where(  # type: ignore[attr-defined]
        lifted, trial_mgr.hold_steps + 1, torch.zeros_like(trial_mgr.hold_steps)
    )
    trial_mgr.f_hold2s = trial_mgr.f_hold2s | (trial_mgr.hold_steps >= 60)  # type: ignore[attr-defined]
    # 5 s full pay, fade by 10 s (was 1.5 s/3 s): the aggressive anti-latch
    # decay starved LEARNERS — a policy that cannot yet complete a carry
    # earns ~nothing from holding, so it drops the piece and parks (traced:
    # handed piece dropped at pin release, arm motionless for 80+ steps).
    # A latcher still caps at ~1/3-episode income vs ~50 for one cycle.
    decay = (1.0 - (trial_mgr.hold_steps.float() - 150.0) / 150.0).clamp(0.0, 1.0)
    piece_z = _get_piece_pos(env)[:, 2]
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    height = ((piece_z - rest_z) / 0.10).clamp(0.0, 1.0)
    return height * decay  # type: ignore[no-any-return]


def hold_maintenance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) +decay while HOLDING the carried piece with fingers in the
    holding band during the carry phase — the direct "do not open your hand"
    gradient.

    Traced at v14c-10k: pregrasp envs opened the hand during the pin and the
    piece fell within ~2 steps of release — far too fast for the lift/carry
    streams to credit-assign "keep the fingers closed". This pays every step
    the grip persists and stops the step it fails, so the atomic action of
    staying closed carries its own advantage. Reuses the piece_lifted
    hold-decay clock: a latcher caps at ~7/episode vs ~30 for one cycle,
    and hold_at_zone_penalty still nets hovering AT the zone to <= 0.
    """
    trial_mgr = _get_trial_mgr(env)
    width = _gripper_width_m(env).squeeze(-1)
    in_band = (width >= GRIPPER_HOLDING_WIDTH_MIN_M) & (width <= GRIPPER_GRASP_WIDTH_M)
    # Grip must persist through CARRY and the SETDOWN lowering alike.
    in_carry = (trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE)) | (
        trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    )
    active = in_carry & in_band & _piece_lifted_mask(env) & trial_mgr.carried_aloft
    decay = (1.0 - (trial_mgr.hold_steps.float() - 150.0) / 150.0).clamp(0.0, 1.0)
    return torch.where(active, decay, torch.zeros_like(decay))  # type: ignore[attr-defined, no-any-return]


def carry_drop_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) -1 one-shot when a carried piece is LOST away from the zone.

    v14d block trend: hold_maintenance income COLLAPSED 0.012 -> 0.002 —
    the policy was learning to drop the handed piece at pin release,
    because dropping cost zero while an attempted carry risked the piece
    slipping, rolling out of bounds, and ending the episode (forfeiting
    all remaining income). Pricing the loss itself flips the ordering:
    instant drop = certain penalty, attempted carry = chance of the +30
    cycle against the same penalty. Set-downs near the zone are exempt
    (that IS the task), and the transition latch means one charge per
    fall, not per airborne step.
    """
    # Carry-around sequence: a drop is losing the piece DURING THE CARRY
    # (PLACE_ON_ZONE). Lowering in the SETDOWN phase is intentional and is
    # never charged — the phase gate replaces the old zone-radius
    # exemption.
    trial_mgr = _get_trial_mgr(env)
    lifted = _piece_lifted_mask(env)
    in_carry = trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE)
    dropped = in_carry & trial_mgr.carried_aloft & trial_mgr.prev_lifted & ~lifted
    trial_mgr.prev_lifted = lifted.clone()  # type: ignore[attr-defined]
    return -dropped.float()  # type: ignore[no-any-return]


def ee_to_piece_distance_3d(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) closeness reward in (0, 1]: 1 - tanh(3d_dist / 0.15).

    Complements ``ee_to_target_distance`` (XY-only, phase-target based) with
    a Z-aware gradient toward the physical piece — without it the policy has
    no dense signal to *descend* to table height.
    """
    ee_pos = _grasp_center_local(env)   # (N, 3) — pad midpoint, not flange
    piece_pos = _get_piece_pos(env)     # (N, 3)
    dist = torch.norm(ee_pos - piece_pos, dim=-1)  # type: ignore[attr-defined]
    # PROGRESS PAYMENTS (the last continuous positional stream converted):
    # closeness income paid ~18/episode for hovering the empty hand beside
    # the piece — risk-free versus a risky full cycle, so the policy slowly
    # drifted back to loitering (v11 decay 0.36 -> 0.24 cycles). Paying per
    # metre of approach CLOSED bounds income at the distance itself.
    trial_mgr = _get_trial_mgr(env)
    pick = trial_mgr.phase == int(TaskPhase.PICK_FROM_BOARD)
    prev = trial_mgr.ee_prev_dist
    progress = torch.where(  # type: ignore[attr-defined]
        pick & (prev >= 0.0),
        (prev - dist).clamp(-0.05, 0.05),  # symmetric — see piece_to_target
        torch.zeros_like(dist),
    )
    trial_mgr.ee_prev_dist = torch.where(  # type: ignore[attr-defined]
        pick, dist, torch.full_like(dist, -1.0)
    )
    return progress


def grasp_height_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) progress payment for bringing the PADS to the cup's mid-height.

    *** MEASURED HARMFUL — NOT REGISTERED. E29 ran this against a matched
    control (3 seeds, 5000 iters, fresh): mean grasp_upright 0.157 -> 0.039,
    i.e. 0.25x where the pre-registered rule required >=1.25x, and grasp rate
    0.61x, failing in the opposite direction on every seed. Kept for the
    record, and because the DIAGNOSIS it came from (E27c) still stands — it
    is the intervention that failed. Do not re-register without re-weighting
    and a fresh pre-registration. ***

    E27c measured the cause of the full-cycle cap: at the grasp instant the
    pad-vs-cup height is 9.9 mm (median) for grasps that come out upright and
    **43.3 mm** for grasps that come out tilted — effect size -1.46, the only
    variable of seven that separates them. The cup's rim sits 25 mm above its
    centre, so a +43 mm grasp closes 18 mm ABOVE the rim, catches the cup by
    its top edge and tips it. A tipped cup can never satisfy
    ``release_at_zone``'s upright clause, and orientation is decided here and
    then persists (P(upright at plate) = 0.861 given an upright grasp, 0.065
    given a tilted one). Lateral alignment is NOT the problem: min XY miss is
    3.4 mm vs 3.2 mm between the two populations.

    PAID AS PROGRESS, not closeness. Absolute-closeness income is the
    loitering exploit this project already paid for twice (see
    ``ee_to_piece_distance_3d`` and the v11 0.36 -> 0.24 decay): a policy that
    is paid for *being* at the right height hovers there and never commits.
    Paying per metre of height error CLOSED bounds total income at the initial
    error, so there is nothing to farm.

    Active during PICK only, and only inside a lateral gate — height error far
    from the cup is meaningless and would just be free income on the way in.
    """
    gc = _grasp_center_local(env)          # (N, 3) — pad midpoint
    piece = _get_piece_pos(env)            # (N, 3)
    err = (gc[:, 2] - piece[:, 2]).abs()   # (N,)

    trial_mgr = _get_trial_mgr(env)
    pick = trial_mgr.phase == int(TaskPhase.PICK_FROM_BOARD)
    # Lateral gate: only shape height once the hand is actually over the cup.
    near = torch.norm(gc[:, :2] - piece[:, :2], dim=-1) <= GRASP_HEIGHT_XY_GATE_M
    active = pick & near

    prev = trial_mgr.gz_prev_err
    progress = torch.where(  # type: ignore[attr-defined]
        active & (prev >= 0.0),
        (prev - err).clamp(-0.02, 0.02),   # symmetric, same as the other streams
        torch.zeros_like(err),
    )
    trial_mgr.gz_prev_err = torch.where(  # type: ignore[attr-defined]
        active, err, torch.full_like(err, -1.0)
    )
    return progress


def upright_maintenance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) progress payment for keeping the cup UPRIGHT while setting down.

    E30, measured before it was written. ``release_at_zone()`` requires the
    cup upright at the release, but nothing asked it to STAY upright while
    being lowered — and the cup is kinematically attached, so it inherits
    every degree of wrist rotation on the way down. The descent probe over
    149 upright plate arrivals found:

        placed (58): tilt 8.5 deg at arrival -> 9.0 deg at the lowest cup-z
        failed (91): tilt 11.2 deg at arrival -> **90.0 deg** — flat on its side

    and partitioned the failures as TIPPED 89.0%, NEVER_LOWERED 4.4%,
    OTHER 4.4%, NEVER_RELEASED 2.2%. Episodes fail after arriving in a
    placeable state, by falling over during the descent.

    Gated on ``attached`` as well as the SETDOWN phase: without that, a
    policy could farm the term by rocking an EMPTY hand, and orientation of
    a cup it is not holding is not its business.

    Paid as PROGRESS with a symmetric clamp, so introducing tilt is CHARGED
    at the same rate correcting it is paid and total income is bounded by
    the initial error. Holding the cup steady earns ~0 and costs ~0; tipping
    it costs. That is the incentive that was missing.

    WEIGHT NOTE: deliberately smaller than the shaping streams it sits
    beside. E29's failure is the reason — `grasp_height` used weight 300 over
    a ~0.1 m error range and distorted the approach badly enough to cost
    grasps outright. Tilt spans ~1.5 rad, an order of magnitude more range,
    so the same weight would dominate the entire reward.
    """
    trial_mgr = _get_trial_mgr(env)
    # Only while genuinely holding the cup, in the set-down phase.
    active = (trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)) & trial_mgr.attached  # type: ignore[attr-defined]

    try:
        quat = env.scene["piece"].data.root_quat_w  # type: ignore[attr-defined]
        up_z = (1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1.0, 1.0)
        tilt = torch.arccos(up_z)                   # radians off vertical
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

    prev = trial_mgr.tilt_prev
    progress = torch.where(  # type: ignore[attr-defined]
        active & (prev >= 0.0),
        (prev - tilt).clamp(-0.02, 0.02),
        torch.zeros_like(tilt),
    )
    trial_mgr.tilt_prev = torch.where(  # type: ignore[attr-defined]
        active, tilt, torch.full_like(tilt, -1.0)
    )
    return progress


def gripper_to_piece_when_grasping(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) penalty for EMPTY pinching: distance to the piece
    while the gripper is closed below the holding band.

    Width < GRIPPER_HOLDING_WIDTH_MIN_M means the fingers closed on nothing
    (a hand around the 40 mm piece stalls near piece diameter). Air-pinching
    was the policy's grasp-faking exploit — make it directly costly.
    Returns ``0.0`` per env when the gripper is open or actually holding.
    """
    gripper_m = _gripper_width_m(env).squeeze(-1)   # (N,)
    piece_pos_w = _get_piece_pos(env)               # (N, 3)
    # Pad centre, not the flange: the penalty asks "did the fingers close on
    # nothing NEAR the piece", and the fingers are 10-13 cm from tool0. Measured
    # from the flange the penalty grew while the pads were closing correctly.
    gripper_pos = _grasp_center_local(env)[:, :2]   # (N, 2) env-local

    piece_xy = piece_pos_w[:, :2]                   # (N, 2)
    dist = torch.norm(gripper_pos - piece_xy, dim=-1)  # (N,)  type: ignore[attr-defined]

    empty_pinch = gripper_m < GRIPPER_HOLDING_WIDTH_MIN_M   # (N,) bool
    return torch.where(empty_pinch, -dist, torch.zeros_like(dist))  # type: ignore[attr-defined]


def action_norm_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) L2 norm of the last action vector.

    Reads ``env.action_manager.action`` which is shaped (num_envs, action_dim).
    Applied with a small negative weight (−0.01) to discourage large motions.
    """
    action = env.action_manager.action  # type: ignore[attr-defined]
    return torch.norm(action, dim=-1)   # type: ignore[attr-defined, no-any-return]


def grasp_confirmed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) sparse +1 on the step where grasp is confirmed.

    Grasp is confirmed when, for GRASP_CONFIRM_STEPS consecutive steps, ALL of:
    * Gripper width inside the HOLDING BAND [GRIPPER_HOLDING_WIDTH_MIN_M,
      GRIPPER_GRASP_WIDTH_M] — an empty pinch closes to ~0 and fails; an
      open hand fails. Only fingers stalled on the piece pass.
    * Gripper-to-piece 3-D distance ≤ GRASP_APPROACH_RADIUS_M.
    * Piece lifted ≥ PIECE_LIFT_THRESHOLD_M above rest — unfakeable: the
      piece only leaves the table if it is actually held. (The v1 criteria
      without this were satisfied by pinching air next to the piece — 80%
      "grasp" rate with zero real pickups.)

    Side-effects:
    * Increments ``trial_state.consec_grasp`` for passing envs.
    * Resets the counter for failing envs.
    * Advances the phase for envs that just hit the confirmation threshold.
    """
    trial_mgr = _get_trial_mgr(env)
    gripper_m = _gripper_width_m(env).squeeze(-1)   # (N,)
    piece_pos_w = _get_piece_pos(env)               # (N, 3)
    # Pad centre, not the flange — the criterion is "is the piece between the
    # pads", which is only meaningful measured from the pads. The radius term
    # is not the binding constraint here (``lifted`` is unfakeable and
    # dominates), so moving the reference point does not loosen the gate or
    # break metric comparability with the E22 numbers.
    gripper_pos_3d = _grasp_center_local(env)       # (N, 3) env-local

    dist_3d = torch.norm(gripper_pos_3d - piece_pos_w, dim=-1)  # (N,)  type: ignore[attr-defined]
    holding_width = (gripper_m >= GRIPPER_HOLDING_WIDTH_MIN_M) & (
        gripper_m <= GRIPPER_GRASP_WIDTH_M
    )
    lifted = piece_pos_w[:, 2] >= (
        TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0 + PIECE_LIFT_THRESHOLD_M
    )
    criteria = holding_width & (dist_3d <= GRASP_APPROACH_RADIUS_M) & lifted
    trial_mgr.f_grasp = trial_mgr.f_grasp | criteria  # type: ignore[attr-defined]
    trial_mgr.consec_grasp = torch.where(  # type: ignore[attr-defined]
        criteria, trial_mgr.consec_grasp + 1, torch.zeros_like(trial_mgr.consec_grasp)
    )

    # PHASE GATE: grasp confirmation only advances the TRUE pick phase.
    # PICK_FROM_ZONE is the SETDOWN phase in the carry-around sequence —
    # gripping there is normal and must NOT advance the machine.
    in_pick_phase = trial_mgr.phase == int(TaskPhase.PICK_FROM_BOARD)
    # >= not ==: the counter keeps climbing while the criteria hold in
    # non-pick phases; an exact-equality check could be stepped over and
    # never fire again for the rest of the episode.
    fired = (trial_mgr.consec_grasp >= GRASP_CONFIRM_STEPS) & in_pick_phase
    trial_mgr.phase = torch.where(fired, trial_mgr.phase + 1, trial_mgr.phase)  # type: ignore[attr-defined]
    trial_mgr.consec_grasp = torch.where(  # type: ignore[attr-defined]
        fired, torch.zeros_like(trial_mgr.consec_grasp), trial_mgr.consec_grasp
    )
    return fired.float()  # type: ignore[no-any-return]


def piece_in_zone(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) sparse +1 on a GENTLE completed set-down (SETDOWN phase).

    Carry-around sequence: fires when the carried piece is RESTING on the
    surface (anywhere), the gripper has RELEASED it, and the touchdown was
    GENTLE (piece speed below GENTLE_SETDOWN_MAX_VEL) — placed, not
    dropped. Advances SETDOWN -> RETURN; the full-cycle bonus fires in
    return_to_reset when the arm is back home.
    """
    trial_mgr = _get_trial_mgr(env)
    piece_xy = _get_piece_pos(env)[:, :2]   # (N, 2)
    piece_z = _get_piece_pos(env)[:, 2]
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    resting = (piece_z - rest_z).abs() <= 0.02
    # E19: RELEASED = the cup is physically free of the hand (attach
    # state), NOT a width test. With finger exploration std ~0.75 the
    # width flickers every step, so the conjunction resting & width-open
    # & gentle almost never aligned — the +43 release chain went
    # uncredited even after real drops at the plate (eval at_plate 0.69
    # vs place 0.004). The attach event is authoritative: once detached
    # and resting, the placement counts regardless of the instantaneous
    # finger width. carried_aloft still blocks shuffleboard.
    released = ~trial_mgr.attached
    # GENTLE: piece speed at rest below threshold — a flung or bounced
    # piece is still moving when it first reads "resting".
    try:
        piece_vel = env.scene["piece"].data.root_lin_vel_w  # type: ignore[attr-defined]
        gentle = torch.norm(piece_vel, dim=-1) <= GENTLE_SETDOWN_MAX_VEL  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        gentle = torch.ones_like(resting)
    # PROVENANCE: the carried flag is latched during CARRY — a set-down
    # only counts for a piece that was genuinely carried.
    in_carry = trial_mgr.phase == int(TaskPhase.PLACE_ON_ZONE)
    trial_mgr.carried_aloft = trial_mgr.carried_aloft | (  # type: ignore[attr-defined]
        in_carry & _piece_lifted_mask(env)
    )
    # Funnel publication (extras["log"] is rebuilt AFTER reset events run;
    # the runner reads it every step, so publishing here lands).
    if trial_mgr.funnel_rates:
        env.extras.setdefault("log", {}).update(trial_mgr.funnel_rates)  # type: ignore[attr-defined]
    in_setdown = trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    # E14 — DEFINED TARGET + UPRIGHT: the set-down only counts at the
    # trial's zone plate, with the cup standing upright (axis within
    # ~15 deg of vertical; attach preserves carry orientation, so an
    # upright pick carries through to an upright placement).
    at_target = (
        torch.norm(piece_xy - trial_mgr.zone_xy, dim=-1) <= PLACEMENT_RADIUS_M  # type: ignore[attr-defined]
    )
    try:
        quat = env.scene["piece"].data.root_quat_w  # type: ignore[attr-defined]
        up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        upright = up_z >= UPRIGHT_COS_GATE
    except (KeyError, AttributeError):
        upright = torch.ones_like(resting)
    fired = (
        in_setdown
        & resting
        & released
        & gentle
        & at_target
        & upright
        & trial_mgr.carried_aloft
    )
    # The set-down point becomes the piece's new origin (next pick target);
    # phase advances to RETURN — the cycle pays out when the arm is home.
    trial_mgr.origin_xy = torch.where(  # type: ignore[attr-defined]
        fired.unsqueeze(-1), piece_xy, trial_mgr.origin_xy
    )
    trial_mgr.f_dock = trial_mgr.f_dock | fired  # type: ignore[attr-defined]
    trial_mgr.phase = torch.where(  # type: ignore[attr-defined]
        fired,
        torch.full_like(trial_mgr.phase, int(TaskPhase.RETURN_TO_BOARD)),
        trial_mgr.phase,
    )
    trial_mgr.carried_aloft = torch.where(  # type: ignore[attr-defined]
        fired, torch.zeros_like(trial_mgr.carried_aloft), trial_mgr.carried_aloft
    )
    trial_mgr.release_paid = torch.where(  # type: ignore[attr-defined]
        fired, torch.zeros_like(trial_mgr.release_paid), trial_mgr.release_paid
    )
    return fired.float()  # type: ignore[no-any-return]


def setdown_approach(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) E16: symmetric approach progress of the HELD cup toward
    the placement plate during the SETDOWN phase.

    E15b flatlined (setdown ~5% for 5k iterations): carries end wherever
    the wander finishes, and nothing paid for approaching the plate —
    the 30 cm docking funnel is out of reach from most carry endpoints.
    Telescoping symmetric progress (the design that taught the original
    carousel transport) closes the gap: net metres toward the plate pay,
    oscillation nets zero.
    """
    trial_mgr = _get_trial_mgr(env)
    in_setdown = trial_mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    active = in_setdown & _piece_lifted_mask(env) & trial_mgr.carried_aloft
    dist = torch.norm(  # type: ignore[attr-defined]
        _get_piece_pos(env)[:, :2] - trial_mgr.zone_xy, dim=-1
    )
    prev = trial_mgr.setdown_prev_dist
    progress = torch.where(  # type: ignore[attr-defined]
        active & (prev >= 0.0),
        (prev - dist).clamp(-0.05, 0.05),
        torch.zeros_like(dist),
    )
    trial_mgr.setdown_prev_dist = torch.where(  # type: ignore[attr-defined]
        active, dist, torch.full_like(dist, -1.0)
    )
    return progress


def return_to_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) RETURN-phase shaping: symmetric joint-space progress of
    the arm toward its default (reset) pose after a completed set-down."""
    trial_mgr = _get_trial_mgr(env)
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    arm_ids = _robot_ids(env)["arm"]
    jdist = torch.norm(  # type: ignore[attr-defined]
        robot.data.joint_pos[:, arm_ids] - robot.data.default_joint_pos[:, arm_ids],
        dim=-1,
    )
    in_return = trial_mgr.phase == int(TaskPhase.RETURN_TO_BOARD)
    prev = trial_mgr.carry_prev_dist
    progress = torch.where(  # type: ignore[attr-defined]
        in_return & (prev >= 0.0),
        (prev - jdist).clamp(-0.05, 0.05),
        torch.zeros_like(jdist),
    )
    trial_mgr.carry_prev_dist = torch.where(  # type: ignore[attr-defined]
        in_return, jdist, torch.full_like(jdist, -1.0)
    )
    return progress


def cycle_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) +1 when the arm reaches home in RETURN — the full
    user-specified sequence is done: pick -> carry 5 s -> gentle set-down
    -> release -> return to reset. advance_cycle restarts at PICK."""
    trial_mgr = _get_trial_mgr(env)
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    arm_ids = _robot_ids(env)["arm"]
    jdist = torch.norm(  # type: ignore[attr-defined]
        robot.data.joint_pos[:, arm_ids] - robot.data.default_joint_pos[:, arm_ids],
        dim=-1,
    )
    in_return = trial_mgr.phase == int(TaskPhase.RETURN_TO_BOARD)
    fired = in_return & (jdist <= RETURN_POSE_TOL_RAD)
    trial_mgr.advance_cycle(fired)  # type: ignore[attr-defined]
    return fired.float()  # type: ignore[no-any-return]


def piece_returned_to_origin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) sparse +1 when piece is returned to its origin.

    Only non-zero in RETURN_TO_BOARD phase.  Advances phase to SUCCESS.
    """
    trial_mgr = _get_trial_mgr(env)
    piece_xy = _get_piece_pos(env)[:, :2]   # (N, 2)
    dist = torch.norm(piece_xy - trial_mgr.origin_xy, dim=-1)  # type: ignore[attr-defined]
    fired = (
        (trial_mgr.phase == int(TaskPhase.RETURN_TO_BOARD))
        & (dist <= PIECE_RETURNED_TOLERANCE_M)
        & _piece_lifted_mask(env)
    )
    # fired → SUCCESS
    trial_mgr.phase = torch.where(fired, trial_mgr.phase + 1, trial_mgr.phase)  # type: ignore[attr-defined]
    return fired.float()  # type: ignore[no-any-return]


def piece_dropped_below_table(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) -1 when piece falls below the table surface."""
    piece_pos_w = _get_piece_pos(env)
    piece_z = piece_pos_w[:, 2]                    # (N,)
    dropped = piece_z < (TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M)
    reward = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
    reward[dropped] = -1.0
    return reward


def arm_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) -1 when the arm collides with non-target objects.

    Reads contact forces from ``env.scene["contact_sensor_arm"]`` if
    present.  Falls back to zero if the sensor is not configured yet.

    Isaac Lab API:
        ``env.scene["contact_sensor_arm"].data.net_forces_w``
        → (num_envs, n_sensor_bodies, 3) contact forces in world frame.
    """
    try:
        sensor = env.scene["contact_sensor_arm"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        sensor = None
    if sensor is None:
        # Sensor not wired yet — return zero reward.
        return torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

    # Non-zero net force on any arm link = collision.
    forces = sensor.data.net_forces_w   # (N, n_links, 3)
    force_mag = torch.norm(forces, dim=-1).max(dim=-1).values  # (N,)  type: ignore[attr-defined]
    collision_mask = force_mag > 0.1    # 0.1 N threshold to filter noise
    reward = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]
    reward[collision_mask] = -1.0
    return reward


# ---------------------------------------------------------------------------
# Termination terms
# ---------------------------------------------------------------------------


def task_success_v1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) bool — True when phase reached SUCCESS.

    Phase is advanced to SUCCESS by ``piece_returned_to_origin`` on the
    step where the piece is within tolerance of its origin.
    """
    trial_mgr = _get_trial_mgr(env)
    return trial_mgr.phase == int(TaskPhase.SUCCESS)  # type: ignore[no-any-return]


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Standard time-out termination via Isaac Lab's episode length buffer.

    Isaac Lab's DoneTerm with ``time_out=True`` handles this automatically;
    this function is a no-op wrapper kept for schema completeness.
    """
    return env.episode_length_buf >= env.max_episode_length  # type: ignore[attr-defined, no-any-return]


def piece_dropped_below_table_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) bool — True when piece has dropped below table."""
    piece_pos_w = _get_piece_pos(env)
    piece_z = piece_pos_w[:, 2]
    return piece_z < (TABLE_SURFACE_Z - PIECE_DROP_THRESHOLD_M)


def piece_out_of_bounds_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) bool — True when the piece leaves the play area.

    Proxy-world equivalent of "the piece fell off the table": the training
    scene's ground plane is infinite, so a knocked-over piece can roll for
    metres and the below-table drop termination never fires. End the trial
    instead when the piece is farther than PIECE_OUT_OF_BOUNDS_RADIUS_M from
    the board centre.
    """
    piece_xy = _get_piece_pos(env)[:, :2]
    center = torch.tensor(  # type: ignore[attr-defined]
        [BOARD_CENTER_X, BOARD_CENTER_Y], device=env.device  # type: ignore[attr-defined]
    )
    return torch.norm(piece_xy - center, dim=-1) > PIECE_OUT_OF_BOUNDS_RADIUS_M  # type: ignore[attr-defined, no-any-return]


def arm_collision_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (num_envs,) bool — True on any unsafe arm collision."""
    try:
        sensor = env.scene["contact_sensor_arm"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        sensor = None
    if sensor is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # type: ignore[attr-defined]
    forces = sensor.data.net_forces_w
    force_mag = torch.norm(forces, dim=-1).max(dim=-1).values  # type: ignore[attr-defined]
    return force_mag > 0.1  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Event terms
# ---------------------------------------------------------------------------


def attach_carried_pieces(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Interval event (every step): kinematic attach-on-grasp (E8).

    The measured blocker (ledger, 2026-07-28): the gripper/table contact
    model cannot sustain a carry (lateral slip at 5 N and 20 N) nor open
    under table-crush loads — capping RL, scripted experts, and any
    future policy alike. Until the gripper model is overhauled, the cup
    is GLUED to the hand while carried:

    * ATTACH: phase is carry/setdown, fingers in the holding band, cup
      within 9 cm of the hand — the PICK itself stays fully physical
      (grasp confirmation still requires a real squeeze-and-lift).
    * While attached: the cup tracks the hand at its attach-time
      tool-frame offset, keeping its attach-time orientation (upright
      carries stay upright); velocities zeroed.
    * DETACH: fingers commanded/read open (width >= 42 mm) or the trial
      left the carry phases — the release DECISION remains learned; only
      the grip physics is bypassed.
    """
    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # type: ignore[import-not-found]

    mgr = _get_trial_mgr(env)
    try:
        piece = env.scene["piece"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    tool_idx = _robot_ids(env)["tool0"]
    tool_pos = robot.data.body_pos_w[:, tool_idx]
    tool_quat = robot.data.body_quat_w[:, tool_idx]
    piece_pos = piece.data.root_pos_w
    width = _gripper_width_m(env).squeeze(-1)

    carryish = (mgr.phase == int(TaskPhase.PLACE_ON_ZONE)) | (
        mgr.phase == int(TaskPhase.PICK_FROM_ZONE)
    )
    in_band = (width >= GRIPPER_HOLDING_WIDTH_MIN_M) & (width <= GRIPPER_GRASP_WIDTH_M)
    near = torch.norm(piece_pos - tool_pos, dim=-1) < 0.09
    new_attach = carryish & in_band & near & ~mgr.attached & (mgr.pin_steps == 0)
    if bool(new_attach.any()):
        mgr.attach_off[new_attach] = quat_apply_inverse(
            tool_quat[new_attach], piece_pos[new_attach] - tool_pos[new_attach]
        )
        mgr.attach_quat[new_attach] = piece.data.root_quat_w[new_attach]
        mgr.attached = mgr.attached | new_attach

    detach = mgr.attached & ((width >= 0.042) | ~carryish)
    mgr.attached = mgr.attached & ~detach

    if not bool(mgr.attached.any()):
        return
    ids = torch.nonzero(mgr.attached).squeeze(-1)
    root_state = piece.data.default_root_state[ids].clone()
    root_state[:, 0:3] = tool_pos[ids] + quat_apply(tool_quat[ids], mgr.attach_off[ids])
    root_state[:, 3:7] = mgr.attach_quat[ids]
    root_state[:, 7:] = 0.0
    piece.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    piece.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)


def hold_staged_pieces(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Interval event (every step): keep curriculum-staged pieces pinned at
    the rendezvous point while the arm settles out of its reset transient.

    Runs for all envs; only envs with pin_steps > 0 are touched.
    """
    mgr = _get_trial_mgr(env)
    active = mgr.pin_steps > 0
    if not bool(active.any()):
        return
    try:
        piece = env.scene["piece"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return
    ids = torch.nonzero(active).squeeze(-1)
    root_state = piece.data.default_root_state[ids].clone()
    root_state[:, 0:3] = mgr.pin_pos[ids] + env.scene.env_origins[ids]  # type: ignore[attr-defined]
    root_state[:, 7:] = 0.0
    piece.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    piece.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)

    # KEEP FINGERS AT PIECE-SURFACE WIDTH while pinned: the pinned piece is
    # effectively infinite-mass, so closing drives push the pads straight
    # THROUGH it (width read 9 mm inside a 40 mm piece). At pin release the
    # stored interpenetration ejected the piece violently and every staged
    # carry died within ~10 steps. Clamping the finger joints each pin step
    # hands over a zero-stored-energy, surface-contact grip.
    robot = env.scene["robot"]  # type: ignore[attr-defined]
    rids = _robot_ids(env)
    jpos = robot.data.joint_pos[ids].clone()
    jvel = robot.data.joint_vel[ids].clone()
    jpos[:, rids["left"]] = jpos[:, rids["left"]].clamp(min=PREGRASP_FINGER_POS_M)
    jpos[:, rids["right"]] = jpos[:, rids["right"]].clamp(max=-PREGRASP_FINGER_POS_M)
    jvel[:, rids["left"]] = 0.0
    jvel[:, rids["right"]] = 0.0
    robot.write_joint_state_to_sim(jpos, jvel, env_ids=ids)

    mgr.pin_steps = torch.where(  # type: ignore[attr-defined]
        active, mgr.pin_steps - 1, mgr.pin_steps
    )


def reset_trial_and_piece(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Episode-reset event: start a new trial and place the piece at its origin.

    1. Resets the ``TrialStateManager`` for the given envs (fresh phase,
       new random target piece + staging zone).
    2. Teleports the ``piece`` rigid object to the new trial's origin board
       square at table height with zero velocity. Positions from the trial
       manager are env-local; per-env origins are added before writing the
       world-frame pose to sim.
    """
    mgr = _get_trial_mgr(env)
    # Episode-end funnel rates over the resetting batch, BEFORE the flags
    # are cleared. Published to extras["log"] each step by piece_in_zone.
    if len(env_ids) > 0:
        mgr.funnel_rates = {
            "Funnel/grasp": float(mgr.f_grasp[env_ids].float().mean()),
            "Funnel/hold2s": float(mgr.f_hold2s[env_ids].float().mean()),
            "Funnel/carry5s": float(mgr.f_carry5s[env_ids].float().mean()),
            "Funnel/setdown": float(mgr.f_dock[env_ids].float().mean()),
            "Funnel/cycle": float((mgr.cycles[env_ids] > 0).float().mean()),
        }
    mgr.reset(env_ids)

    try:
        piece = env.scene["piece"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return  # no physical piece in the scene — trial reset alone is fine

    root_state = piece.data.default_root_state[env_ids].clone()  # (M, 13)
    root_state[:, 0:2] = mgr.origin_xy[env_ids]
    root_state[:, 2] = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0

    # Curriculum split per reset:
    #   [0, PREGRASP_FRACTION)                      -> piece in hand at the
    #       default pose, phase PLACE_ON_ZONE (trains hold + carry-out).
    #   [PREGRASP_FRACTION, +ZONE_START_FRACTION)   -> piece standing ON its
    #       assigned zone, phase PICK_FROM_ZONE (trains re-pick + carry home).
    #   rest                                        -> scratch (full task).
    u = torch.rand(len(env_ids), device=root_state.device)  # type: ignore[attr-defined]
    pre_mask = u < PREGRASP_FRACTION
    zone_mask = (u >= PREGRASP_FRACTION) & (u < PREGRASP_FRACTION + ZONE_START_FRACTION)
    # CARRY-ELAPSED start: staged like pregrasp (piece in hand at the
    # rendezvous) with the carry clock nearly complete — the episode is a
    # few steps from the SETDOWN phase, mass-producing gentle-set-down and
    # return practice (the curriculum trick that installed the release).
    dock_mask = (u >= PREGRASP_FRACTION + ZONE_START_FRACTION) & (
        u < PREGRASP_FRACTION + ZONE_START_FRACTION + CARRY_ELAPSED_FRACTION
    )
    # E3 RETURN-start slice: cup resting at its origin, phase RETURN — the
    # episode opens one homing motion from the sequence bonus.
    _r0 = PREGRASP_FRACTION + ZONE_START_FRACTION + CARRY_ELAPSED_FRACTION
    return_mask = (u >= _r0) & (u < _r0 + RETURN_START_FRACTION)
    if return_mask.any():
        mgr.phase[env_ids[return_mask]] = int(TaskPhase.RETURN_TO_BOARD)
    staged_mask = pre_mask | dock_mask

    if staged_mask.any():
        staged_ids = env_ids[staged_mask]
        root_state[staged_mask, 0] = PREGRASP_CENTER_LOCAL[0]
        root_state[staged_mask, 1] = PREGRASP_CENTER_LOCAL[1]
        root_state[staged_mask, 2] = PREGRASP_CENTER_LOCAL[2]
        mgr.phase[staged_ids] = int(TaskPhase.PLACE_ON_ZONE)

        robot = env.scene["robot"]  # type: ignore[attr-defined]
        rids = _robot_ids(env)
        jpos = robot.data.default_joint_pos[staged_ids].clone()
        jvel = robot.data.default_joint_vel[staged_ids].clone()
        jpos[:, rids["left"]] = PREGRASP_FINGER_POS_M
        jpos[:, rids["right"]] = -PREGRASP_FINGER_POS_M
        robot.write_joint_state_to_sim(jpos, jvel, env_ids=staged_ids)
        # RESET-POSE TRANSIENT: the arm physically starts at its spawn
        # configuration and takes ~25 steps to settle into the default
        # pose — pieces staged at the settled grasp centre fall through
        # empty air long before the hand arrives. Pin staged pieces at
        # the rendezvous point until the hand settles onto them.
        mgr.pin_steps[staged_ids] = 30
        mgr.pin_pos[staged_ids, 0] = PREGRASP_CENTER_LOCAL[0]
        mgr.pin_pos[staged_ids, 1] = PREGRASP_CENTER_LOCAL[1]
        mgr.pin_pos[staged_ids, 2] = PREGRASP_CENTER_LOCAL[2]

    if dock_mask.any():
        # Pre-credit the carry requirements so the SETDOWN phase arrives a
        # few steps after the staging pin releases (pin adds ~30 held
        # steps on top of this credit).
        dock_ids = env_ids[dock_mask]
        mgr.hold_steps[dock_ids] = CARRY_DURATION_STEPS - 20
        mgr.move_dist[dock_ids] = MIN_CARRY_PATH_M
        # E18: PROGRESSIVE PLACEMENT TARGET. Since the defined-target gate
        # (E14), staged release practice at the rendezvous earned ZERO
        # placement credit (~30 cm from any plate) and setdowns regressed
        # (E17: 4.2% -> 2.2%). Lerp this slice's target from the
        # rendezvous toward the real plate: t~0 pays release practice
        # immediately, t~1 is a true plate placement — graduation under
        # attach, where staged holds are reliable.
        t = torch.rand((len(dock_ids), 1), device=root_state.device)  # type: ignore[attr-defined]
        start = torch.tensor(
            PREGRASP_CENTER_LOCAL[:2], device=root_state.device
        ).unsqueeze(0)
        real_zone = mgr.zone_centers[mgr.zone_idx[dock_ids]]
        mgr.zone_xy[dock_ids] = start + t * (real_zone - start)

    if zone_mask.any():
        # SURFACE-PICK stage (evolution of the grasp-value stage, whose
        # value-association job is done — jackpots flow). The remaining
        # wall is descending onto a SURFACE piece: only ~3% of episodes
        # perform a from-surface grasp. Pin the piece ON THE TABLE directly
        # beneath the settling hand (which arrives ~4 cm above it), fingers
        # open — close+lift becomes a one-move discovery, bridging the
        # catch skill to true surface picking. Phase stays PICK.
        zone_ids = env_ids[zone_mask]
        rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
        root_state[zone_mask, 0] = PREGRASP_CENTER_LOCAL[0]
        root_state[zone_mask, 1] = PREGRASP_CENTER_LOCAL[1]
        root_state[zone_mask, 2] = rest_z
        mgr.pin_steps[zone_ids] = 30
        mgr.pin_pos[zone_ids, 0] = PREGRASP_CENTER_LOCAL[0]
        mgr.pin_pos[zone_ids, 1] = PREGRASP_CENTER_LOCAL[1]
        mgr.pin_pos[zone_ids, 2] = rest_z

    root_state[:, :3] += env.scene.env_origins[env_ids]  # type: ignore[attr-defined]
    root_state[:, 7:] = 0.0

    piece.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    piece.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


# ---------------------------------------------------------------------------
# Action terms
# ---------------------------------------------------------------------------


def joint_position_action(env: ManagerBasedRLEnv) -> None:  # type: ignore[return]
    """Apply absolute joint position targets to the arm joints.

    This function is referenced by ``ActionsCfg.arm_joint_pos`` and called
    each step by Isaac Lab's ActionManager.  The actual PhysX drive application
    is handled by the framework; this stub satisfies the function-reference
    requirement in ``ActionTermCfg``.

    The policy outputs 6 target joint angles (radians).  Values are clipped to
    the per-joint URDF limits configured in ``ActionsCfg.arm_joint_pos.clip``
    before being passed to the articulation drives.

    Isaac Lab API (called internally by framework):
        ``env.scene["robot"].set_joint_position_target(targets, joint_ids)``

    Args:
        env: The Isaac Lab manager-based RL env instance.
    """
    # Isaac Lab's ActionManager calls this function and uses its return value
    # for clipping/scaling before passing to the asset drives.  The actual
    # drive application is handled by the framework post-clip.
    # This stub intentionally returns None; Isaac Lab's ActionTermCfg.func
    # contract allows pure-side-effect functions — the framework reads
    # env.action_manager.action for the processed tensor.
    pass  # pragma: no cover - action application is framework-handled


def gripper_position_action(env: ManagerBasedRLEnv) -> None:  # type: ignore[return]
    """Apply symmetric gripper finger position targets.

    This function is referenced by ``ActionsCfg.gripper`` and called each
    step by Isaac Lab's ActionManager.  The policy outputs a scalar in
    [-1.0, +1.0]:

        +1.0 → fully open  (left_finger = +0.025 m, right_finger = -0.025 m)
        -1.0 → fully closed (left_finger = 0.0 m,   right_finger =  0.0 m)

    The ``scale=0.025`` in ``ActionTermCfg`` maps the policy output to
    metres before the clip is applied by the framework.

    Args:
        env: The Isaac Lab manager-based RL env instance.
    """
    pass  # pragma: no cover - action application is framework-handled

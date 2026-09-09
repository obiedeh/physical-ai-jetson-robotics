"""Env config classes for the Synria board ↔ staging pick-and-place task.

One base config (`SynriaTabletopPickPlaceEnvCfg`) carries everything shared
across the three games (Ludo, chess, checkers): the arm articulation, the
table + floor scene, the four staging zones, the observation/action/reward/
termination managers. Three per-game subclasses (`SynriaLudoPickPlaceEnvCfg`,
`SynriaChessPickPlaceEnvCfg`, `SynriaCheckersPickPlaceEnvCfg`) bind a
specific scene USD and game-specific initial state.

Per the task spec in ``docs/SYNRIA_PICK_AND_PLACE_TASK.md``, the V1 task is:

    1. Pick a target piece off the board.
    2. Place it on one of the four staging zones (left / right / top / bottom).
    3. Pick it back up.
    4. Return it to its original square.

Runtime requires Isaac Lab on a Linux RTX 5090 box. On Windows the module
imports as a no-op (the Isaac Lab imports raise ImportError at module load
time and the package surface is empty until import-time fix).
"""

from __future__ import annotations

from pathlib import Path

from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    PIECE_HEIGHT_M,
    PIECE_RADIUS_M,
    TABLE_SURFACE_Z,
    piece_start_xy,
)

# Resolve once at module load — repo-root paths used in scene USD references.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# v2 (2026-07-25): re-imported with Isaac Sim 5.1's classic URDF importer.
# The v1 export (synria_6dof_arm/synria_6dof_arm.usda) shipped a collision
# payload from a DIFFERENT robot (UR-style link names), so no collider ever
# bound to the Alicia links — the gripper closed straight through the piece
# (diag_grasp_feasibility.py). v2 has convex-hull colliders on all 9 links
# and a well-formed fixed-base root joint.
_SYNRIA_ARM_USD = (
    _REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm_v2" / "synria_6dof_arm.usd"
)
# Scene USDs — board-mirrored ("left") variants of the Alicia-D no-robot
# scenes. Each is a thin override that references the original Alicia-D scene
# and moves /World/Game to the +X table end (mirrored across world X = 0), so
# the board sits on the same camera-left side as the interactive viewer for
# every game. Regenerate with:
#   isaac/usd/scenes/synria_mirrored/generate_mirrored_scenes.py
# The robot articulation is layered on top by SynriaTabletopSceneCfg.robot.
_MIRRORED_SCENES_DIR = _REPO_ROOT / "isaac" / "usd" / "scenes" / "synria_mirrored"
_SCENE_USD = {
    "ludo": _MIRRORED_SCENES_DIR / "ludotable_norobot_left.usda",
    "chess": _MIRRORED_SCENES_DIR / "chesstable_norobot_left.usda",
    "checkers": _MIRRORED_SCENES_DIR / "checkerstable_norobot_left.usda",
}
# Legacy in-repo scene USDs (kept here for reference / rollback):
# _SCENE_USD = {
#     "ludo": ... / "scenes" / "synria_ludo" / "synria_ludo_v0.usda",
#     "chess": ... / "scenes" / "synria_chess" / "synria_chess_v0.usda",
#     "checkers": ... / "scenes" / "synria_checkers" / "synria_checkers_v0.usda",
# }


# ---------------------------------------------------------------------------
# Isaac Lab imports — gated so the module parses on Windows without Isaac Lab.
# ---------------------------------------------------------------------------

try:
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
    from isaaclab.assets import (  # type: ignore[import-not-found]
        ArticulationCfg,
        AssetBaseCfg,
        RigidObjectCfg,
    )
    from isaaclab.controllers import DifferentialIKControllerCfg  # type: ignore[import-not-found]
    from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg  # type: ignore[import-not-found]
    from isaaclab.envs.mdp.actions import (  # type: ignore[import-not-found]
        BinaryJointPositionActionCfg,
        JointPositionActionCfg,
    )
    from isaaclab.managers import (  # type: ignore[import-not-found]
        ActionTermCfg,
        SceneEntityCfg,
    )
    from isaaclab.managers import (
        EventTermCfg as EventTerm,
    )
    from isaaclab.managers import (
        ObservationGroupCfg as ObsGroup,
    )
    from isaaclab.managers import (
        ObservationTermCfg as ObsTerm,
    )
    from isaaclab.managers import (
        RewardTermCfg as RewTerm,
    )
    from isaaclab.managers import (
        TerminationTermCfg as DoneTerm,
    )
    from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import-not-found]
    from isaaclab.utils.configclass import configclass  # type: ignore[import-not-found]

    _ISAACLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware-gated import
    _ISAACLAB_AVAILABLE = False

    # Provide minimal stand-ins so the rest of this module parses. These are
    # ONLY for static parsing; instantiating them raises.
    def configclass(cls: type) -> type:  # type: ignore[no-redef]
        return cls

    class _StubBase:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "Isaac Lab is not available in this environment. "
                "Run this task config on the Linux RTX 5090 workstation "
                "with Isaac Sim 5.1 + Isaac Lab installed."
            )

    ManagerBasedRLEnvCfg = _StubBase  # type: ignore[assignment, misc]
    ViewerCfg = _StubBase  # type: ignore[assignment, misc]
    InteractiveSceneCfg = _StubBase  # type: ignore[assignment, misc]
    ArticulationCfg = _StubBase  # type: ignore[assignment, misc]
    AssetBaseCfg = _StubBase  # type: ignore[assignment, misc]
    ImplicitActuatorCfg = _StubBase  # type: ignore[assignment, misc]
    ActionTermCfg = _StubBase  # type: ignore[assignment, misc]
    JointPositionActionCfg = _StubBase  # type: ignore[assignment, misc]
    BinaryJointPositionActionCfg = _StubBase  # type: ignore[assignment, misc]
    DifferentialIKControllerCfg = _StubBase  # type: ignore[assignment, misc]
    ObsGroup = _StubBase  # type: ignore[assignment, misc]
    ObsTerm = _StubBase  # type: ignore[assignment, misc]
    RewTerm = _StubBase  # type: ignore[assignment, misc]
    DoneTerm = _StubBase  # type: ignore[assignment, misc]
    EventTerm = _StubBase  # type: ignore[assignment, misc]
    RigidObjectCfg = _StubBase  # type: ignore[assignment, misc]
    SceneEntityCfg = _StubBase  # type: ignore[assignment, misc]
    sim_utils = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Local MDP terms (observation/action/reward/termination function stubs)
# ---------------------------------------------------------------------------

if _ISAACLAB_AVAILABLE:
    from . import mdp


# ---------------------------------------------------------------------------
# Scene config — arm articulation + scene USD as a base asset
# ---------------------------------------------------------------------------


@configclass
class SynriaTabletopSceneCfg(InteractiveSceneCfg):
    """Synria arm + tabletop scene. Subclasses bind a specific scene USD."""

    if _ISAACLAB_AVAILABLE:
        # Synria 6DOF arm — loaded from the SolidWorks-export USD.
        # The arm USD provides articulation, joint limits, mesh references.
        #
        # Actuator limits sourced from synria_6dof_arm.urdf:
        #   Joint1: ±2.749 rad, effort=5 Nm, velocity=12 rad/s
        #   Joint2: ±2.000 rad, effort=5 Nm, velocity=12 rad/s
        #   Joint3: -0.5..+π rad, effort=5 Nm, velocity=12 rad/s
        #   Joint4: ±2.790 rad, effort=5 Nm, velocity=12 rad/s
        #   Joint5: ±1.570 rad, effort=5 Nm, velocity=12 rad/s
        #   Joint6: ±π rad, effort=5 Nm, velocity=12 rad/s
        #   left_finger:  0..+0.025 m (prismatic), effort=5 N, velocity=12 m/s
        #   right_finger: -0.025..0 m (prismatic), effort=5 N, velocity=12 m/s
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/SynriaArm",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_SYNRIA_ARM_USD),
                # copy_from_source=False uses USD reference mechanism which properly
                # remaps absolute body paths (</RobotName/Geometry/base_link/...>)
                # to the instantiated prim path, fixing joint body resolution.
                copy_from_source=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                # TRAINING frame = task_geometry frame (board centre at
                # (0.10, 0), squares x ∈ [-0.09, 0.29], y ∈ ±0.19, surface at
                # TABLE_SURFACE_Z). The old viewer-scene placement (0.1524, 0,
                # 0.8382) put the arm base 8.9 cm from the nearest chess
                # square — the reach reward pulled the gripper straight into
                # the arm's own base column. At x = -0.22 the 32 squares sit
                # in a 0.19–0.55 m annulus: clear of the base, inside the
                # arm's ~0.6 m reach (URDF link sum). Identity rotation faces
                # the arm's URDF +X forward at the board.
                pos=(-0.22, 0.0, TABLE_SURFACE_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
                # Task-ready rest pose: elbow lifted (Joint3) with the gripper
                # open, so the arm boots posed over the table rather than
                # folded flat.
                joint_pos={
                    "Joint1": 0.0,
                    "Joint2": 0.0,
                    "Joint3": 1.3,
                    "Joint4": 0.0,
                    "Joint5": 0.0,
                    "Joint6": 0.0,
                    "left_finger": 0.025,
                    "right_finger": -0.025,
                },
            ),
            actuators={
                # *_sim variants: plain effort_limit is deprecated and plain
                # velocity_limit is silently IGNORED by implicit actuators
                # (Isaac Lab warns at startup) — joints could snap at
                # unbounded speed toward noisy exploration targets, which is
                # what made the wrist/gripper flicker violently during
                # training. 3 rad/s is a sane physical cap for this arm.
                "arm_joints": ImplicitActuatorCfg(
                    joint_names_expr=["Joint[1-6]"],
                    effort_limit_sim=5.0,
                    velocity_limit_sim=3.0,
                    stiffness=400.0,
                    damping=40.0,
                ),
                "gripper_fingers": ImplicitActuatorCfg(
                    joint_names_expr=["left_finger", "right_finger"],
                    # 20 N, was 5 N (E7): six controller designs (RL policy,
                    # IK expert, joint-space expert, assist variants) all
                    # failed at release/carry the same way — at 5 N the grip
                    # has ~1-2 N lateral capacity (cup slips in any real
                    # carry) and the fingers jam shut under table-contact
                    # crush loads, physically unable to open (measured:
                    # width 10 mm vs open targets for hundreds of steps).
                    # Real parallel grippers of this class exert 20-100 N;
                    # the force limit stays BOUNDED and modest — this is a
                    # calibration correction, not a disabled safety limit.
                    effort_limit_sim=20.0,
                    velocity_limit_sim=0.05,
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
        )

        # Per-scene USD — overridden in each subclass via __post_init__.
        tabletop: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tabletop",
            spawn=sim_utils.UsdFileCfg(usd_path=""),  # filled by subclass
        )

        # Scene illumination — the training scene has no scene-USD lights
        # (the tabletop USD is swapped for a bare ground plane), so without
        # this the GUI viewport renders near-black.
        dome_light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
        )

        # The physical target piece — a pawn-sized cylinder proxy until
        # per-game piece USDs are wired in. Spawn XY is the chess piece-0
        # start square; the reset event re-places it at each trial's origin
        # square, so the static spawn only needs to be somewhere sane.
        piece: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Piece",
            spawn=sim_utils.CylinderCfg(
                radius=PIECE_RADIUS_M,
                height=PIECE_HEIGHT_M,
                # Heavy damping: the cylinder tips onto its side on any
                # imperfect landing and ROLLS — 78% of episodes still died
                # out-of-bounds even at 0.8 m. Wider bounds risk pieces
                # physically entering neighbour envs (2 m spacing), so kill
                # the roll at the source: land, wobble, stop.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    angular_damping=5.0,
                    linear_damping=0.5,
                ),
                # 50 g + high friction: at 20 g with default material, a
                # glancing blow from the arm launched the piece rolling for
                # metres across the ground plane.
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.6,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    piece_start_xy("chess", 0)[0],
                    piece_start_xy("chess", 0)[1],
                    TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0,
                )
            ),
        )


# ---------------------------------------------------------------------------
# Observation / action / reward / termination managers (group configs)
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation manager — see docs/SYNRIA_PICK_AND_PLACE_TASK.md."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: state + target + cameras."""

        if _ISAACLAB_AVAILABLE:
            # Joint state (6 positions + 6 velocities + gripper open width)
            joint_state = ObsTerm(func=mdp.joint_state_synria)
            # End-effector pose (3 position + 4 quaternion, env-local)
            ee_pose = ObsTerm(func=mdp.ee_pose_synria)
            # Piece position (3, env-local) — the policy must see the piece
            piece_pos = ObsTerm(func=mdp.piece_pos_rel)
            # Target encoding: which piece, which zone
            target_piece_id = ObsTerm(func=mdp.target_piece_id)
            target_zone_id = ObsTerm(func=mdp.target_zone_id)
            # The ACTUAL metric target position + piece->target delta (4).
            # The one-hot zone index above forced the policy to memorize
            # label->coordinates unaided, and the carry-gradient curriculum
            # made the label wrong — the destination must be OBSERVED.
            target_pos = ObsTerm(func=mdp.trial_target_pos)
            # Trial phase (5) — without it "piece in my open fingers" is the
            # same observation whether the job is to grab or to let go
            trial_phase = ObsTerm(func=mdp.trial_phase_onehot)
            # TODO(RTX): add overhead RGB + wrist RGB camera observations
            # once camera sensor configs are wired in the scene USD.

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    if _ISAACLAB_AVAILABLE:
        policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action manager — joint position targets + gripper command.

    Action vector layout (8 floats total):
        [0..5]  Target joint positions in radians for Joint1..Joint6.
                Clipped to URDF joint limits before being applied.
        [6]     Left-finger target position (0.0 = closed, 0.025 = open).
        [7]     Right-finger target position (0.0 = closed, -0.025 = open).

    The policy outputs absolute joint position targets.  The actuator PD
    controller (stiffness 400 Nm/rad, damping 40 Nm·s/rad) tracks them
    with implicit PhysX drives.
    """

    if _ISAACLAB_AVAILABLE:
        arm_joint_pos: JointPositionActionCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"],
            scale=1.0,
            clip={
                "Joint1": (-2.749, 2.749),
                "Joint2": (-2.000, 2.000),
                "Joint3": (-0.500, 3.14159),
                "Joint4": (-2.790, 2.790),
                "Joint5": (-1.570, 1.570),
                "Joint6": (-3.14159, 3.14159),
            },
        )

        # E22 — BINARY GRIPPER (diagnosis #2): a clipped Gaussian head
        # cannot learn the gripper's effectively binary open/close
        # decision (large std saturates the 10 mm clip -> no mean
        # gradient; small std destroys stochastic-grasp robustness).
        # One signed action: > 0 = open (width 50 mm), < 0 = close
        # (width 31 mm — gentle <=5 mm/pad squeeze on the 40 mm cup;
        # the earlier deep-squeeze ejection lesson is preserved in the
        # close targets). Same solution as Isaac Lab's Franka tasks.
        gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={"left_finger": 0.025, "right_finger": -0.025},
            close_command_expr={"left_finger": 0.0155, "right_finger": -0.0155},
        )


@configclass
class RewardsCfg:
    """Reward manager — see docs/SYNRIA_PICK_AND_PLACE_TASK.md reward sketch."""

    if _ISAACLAB_AVAILABLE:
        # SIGN CONVENTION: the two closeness terms return POSITIVE rewards
        # in (0, 1] (1 - tanh kernels) so that staying engaged with the piece
        # EARNS income each step — with penalty-only shaping, early
        # termination stops the penalty clock and the policy learns to knock
        # the piece out of bounds ASAP. The remaining terms return NEGATIVE
        # values for undesirable states (-dist, -1 on drop/collision), so
        # their weights must also be positive (negative weights inverted
        # them; found + fixed 2026-07-25).

        # Dense shaping. piece_to_target (not EE-to-target): carrying the
        # PIECE is what pays — walking an empty hand to the zone earns
        # nothing (that was the air-pinch exploit's payoff). piece_lift
        # bridges grasp → carry with unfakeable income for raising the piece.
        # ECONOMY (v6 rebalance): carrying must out-pay hovering. At lift=3.0
        # vs carry=0.5 the policy farmed altitude at the origin forever —
        # holding still paid 6x more than progressing. Now the carry term
        # dominates dense income and the terminal milestones (below) out-pay
        # any farming strategy, so finishing the task is the best policy.
        # weight 600 with progress payments: a full 0.45 m carry earns
        # ~600 x 0.45 x dt = +9 total, spread over the actual approach.
        piece_to_target_distance = RewTerm(func=mdp.piece_to_target_distance, weight=600.0)
        # progress payments: ~0.4 m approach x 300 x dt = +4 total per pick
        ee_to_piece_distance = RewTerm(func=mdp.ee_to_piece_distance_3d, weight=300.0)

        # E27c/E28's `grasp_height` term lived here and is REVERTED. E29
        # measured it against a matched control: mean grasp_upright 0.157 ->
        # 0.039 (0.25x, where the pre-registered rule needed >=1.25x) and
        # grasp rate 0.61x, on all three seeds. Driving the hand toward the
        # cup's mid-height appears to cost grasps outright. The FUNCTION is
        # kept in mdp.py with its result recorded; do not re-register it
        # without re-weighting and a fresh pre-registration.
        # 1.5, not 0.8: at 0.8 the v7 policy learned to PUSH the piece to the
        # zone without ever grasping (shuffleboard) — but the state machine
        # requires a real lift-grasp at PICK_FROM_ZONE, so pushing alone can
        # never finish. Lift must pay enough to be worth discovering, while
        # staying below the carry term so hover-farming doesn't return.
        piece_lift = RewTerm(func=mdp.piece_lifted, weight=1.5)
        gripper_to_piece_when_grasping = RewTerm(
            func=mdp.gripper_to_piece_when_grasping, weight=0.3
        )
        # Release shaping: pay for an OPEN hand at the destination, and tax
        # hovering the held piece over it — "drop it" must beat "hold it".
        hold_maintenance = RewTerm(func=mdp.hold_maintenance, weight=1.0)
        # 15, was 60: with the carry gradient putting 25% of episodes into
        # real carries, -2 per drop made GRASPING itself net-negative — the
        # policy suppressed picks to avoid ever entering the risky carry
        # phase (v16e: grasp funnel fell monotonically 52% -> 17% while
        # zone/setdown held). The penalty was scaffolding for the era when
        # completions were impossible and dropping was otherwise free; now
        # a drop already forfeits a reachable +30, so pricing stays mild.
        carry_drop = RewTerm(func=mdp.carry_drop_penalty, weight=15.0)
        # 90, was 3: at weight 3 the release breadcrumb paid +0.1 per event —
        # logged income rounded to 0.0000 over 7k iterations while the
        # dock->setdown funnel conversion sat flat at ~7%. The docked-and-
        # holding state earns ~nothing, so opening the hand was guided only
        # by the diluted sparse +30. +3/step of released-low-at-zone is
        # visible; no farm window because the payment region equals the
        # cycle radius and the cycle fire flips the phase within ~2 steps.
        release_at_zone = RewTerm(func=mdp.release_at_zone, weight=90.0)

        # E30 — UPRIGHT ON DESCENT. 89% of failed placements arrive upright
        # (11.2 deg) and then fall flat during the descent (90 deg at the
        # lowest cup-z), while successful ones hold within 9 deg. Nothing
        # priced cup orientation while lowering. Symmetric progress payment,
        # so tipping is charged at the rate correcting is paid.
        #
        # Weight 100, not 300: tilt spans ~1.5 rad against grasp_height's
        # ~0.1 m, so matching that weight would let this term dominate the
        # whole reward. E29 is the cautionary case — a 300-weight shaping
        # term over the wrong range cost grasps outright.
        upright_descent = RewTerm(func=mdp.upright_maintenance, weight=100.0)
        # 15, was 1.5: with the dwell clock in lower_at_zone this is now a
        # capped parking attractor (~+15/trial max) rather than an
        # unbounded hover stream — strong enough to teach STOPPING over
        # the zone, still dominated by the +30 cycle bonus.
        lower_at_zone = RewTerm(func=mdp.lower_at_zone, weight=15.0)
        # Newborn guidance: small dense kernels so a fresh network has a
        # gradient before it can earn progress/event income. Ceilings are
        # negligible for a competent policy (see mdp docstrings).
        approach_guidance = RewTerm(func=mdp.approach_guidance, weight=0.15)
        carry_guidance = RewTerm(func=mdp.carry_guidance, weight=0.2)
        # DE-FANGED (weight 0, term kept for its metric stream): the hover
        # tax was added to kill arrival-null loitering, but hold-decay and
        # carry_drop now do that job. Left active it taxed the DOCKING
        # APPROACH itself — every path to the +30 set-down passes through
        # "at zone & lifted", so each fumbled docking attempt collected
        # -1/step and the policy repeatedly learned then UNLEARNED zone
        # approach (lower_at_zone peaked and decayed to ~0 three times:
        # v14e blk3, v14f blk3-6).
        hold_at_zone = RewTerm(func=mdp.hold_at_zone_penalty, weight=0.0)
        action_norm_penalty = RewTerm(func=mdp.action_norm_penalty, weight=-0.01)

        # Sparse milestones. Isaac Lab multiplies every term by step dt
        # (1/30 s), so a one-shot milestone needs weight ≈ 30× its intended
        # payout: weight 30 → +1 actual, 150 → +5. The old weights of 1-5
        # paid 0.03-0.17 — noise next to ~27/episode of dense income, so the
        # optimum was to hover forever and never risk the grasp.
        # 300 (+10 one-shot): farm-proofing all the continuous incomes also
        # starved grasp DISCOVERY — v10 plateaued at the curriculum ceiling
        # (~0.2 cycles/ep = pre-grasped envs banking one free cycle each)
        # because from-surface re-picks never got found. A one-shot bonus is
        # structurally unfarmable (fires once, phase advances immediately),
        # so it can be as loud as discovery needs.
        grasp_confirmed = RewTerm(func=mdp.grasp_confirmed, weight=300.0)
        # RETURN-phase shaping (joint-space progress home) and the full
        # carry-around sequence bonus (+30 when the arm reaches home after
        # a gentle set-down). piece_in_zone is the intermediate set-down
        # event at +10.
        # E2 (weight 180) REVERTED: tripling the terminal stream regressed
        # transport (0.223->0.168) and place (0.117->0.063) on the fixed
        # harness with full flat — the bigger terminal gradient starved the
        # stages that feed it.
        return_home = RewTerm(func=mdp.return_to_reset, weight=60.0)
        # E16: telescoping approach income toward the placement plate in
        # SETDOWN — a full 0.3 m approach pays ~+6 total.
        setdown_approach = RewTerm(func=mdp.setdown_approach, weight=600.0)
        cycle_complete = RewTerm(func=mdp.cycle_complete, weight=900.0)
        # Carousel cycle bonus: pick -> carry aloft -> SET DOWN in zone
        # (resting + released). Weight 900 = +30 per cycle; the logged
        # Episode_Reward value reads ~1.0 per completed cycle.
        piece_in_zone = RewTerm(func=mdp.piece_in_zone, weight=300.0)

        piece_dropped = RewTerm(func=mdp.piece_dropped_below_table, weight=30.0)
        collision = RewTerm(func=mdp.arm_collision, weight=30.0)


@configclass
class TerminationsCfg:
    """Termination manager — success, truncation, drop, collision."""

    if _ISAACLAB_AVAILABLE:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        piece_dropped = DoneTerm(func=mdp.piece_dropped_below_table_term)
        piece_out_of_bounds = DoneTerm(func=mdp.piece_out_of_bounds_term)
        collision = DoneTerm(func=mdp.arm_collision_term)


@configclass
class EventsCfg:
    """Event manager — new trial + piece placement on every episode reset."""

    if _ISAACLAB_AVAILABLE:
        reset_trial = EventTerm(func=mdp.reset_trial_and_piece, mode="reset")
        # every-step pin keeps staged pieces at the rendezvous point while
        # the arm settles out of the reset-pose transient (~1 s)
        hold_staged = EventTerm(
            func=mdp.hold_staged_pieces,
            mode="interval",
            interval_range_s=(0.03, 0.04),
        )
        # E8 kinematic attach-on-grasp: the cup is glued to the hand while
        # carried (pick stays physical; release decision stays learned) —
        # see attach_carried_pieces docstring and the experiment ledger's
        # amended diagnosis (gripper/table contact fidelity).
        attach_carried = EventTerm(
            func=mdp.attach_carried_pieces,
            mode="interval",
            interval_range_s=(0.03, 0.04),
        )


# ---------------------------------------------------------------------------
# Base env config
# ---------------------------------------------------------------------------


@configclass
class SynriaTabletopPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config for the Synria board ↔ staging pick-and-place task.

    Three per-game subclasses bind the scene USD and game-specific initial
    state. The shared MDP structure (observations, actions, rewards,
    terminations) is defined once here.
    """

    if _ISAACLAB_AVAILABLE:
        # GUI viewport: start the camera framed on env_0's arm + piece
        # instead of the world-origin default (eye 7.5 m out — the dark-gray
        # arms are invisible from there).
        viewer: ViewerCfg = ViewerCfg(
            eye=(1.3, 1.3, 1.7),
            lookat=(0.1, 0.0, 0.85),
            origin_type="env",
            env_index=0,
        )

        # Scene
        # replicate_physics=False: the arm USD bakes a world-anchored fixed
        # Physics/root_joint with no body0 rel. The physics cloner cannot
        # update that joint's localPose per env ("Cloning joints ... without
        # a body rel may cause issues"), so clones spawn offset from their
        # root-joint anchor and the corrective forces explode the sim within
        # seconds (links at ~1e13 m). Parsing each env's physics individually
        # avoids the stale anchors at the cost of slower startup.
        scene: SynriaTabletopSceneCfg = SynriaTabletopSceneCfg(
            num_envs=4096, env_spacing=2.0, replicate_physics=False
        )

        # MDP
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventsCfg = EventsCfg()

        # Episode timing
        episode_length_s: float = 30.0
        decimation: int = 2

    def __post_init__(self) -> None:
        if not _ISAACLAB_AVAILABLE:
            return
        # Per-subclass scene USD wired in by `_bind_scene` in the subclass.
        if hasattr(self, "_bind_scene"):
            self._bind_scene()


# ---------------------------------------------------------------------------
# Per-game subclasses — bind scene USDs and initial-state seeding
# ---------------------------------------------------------------------------


@configclass
class SynriaLudoPickPlaceEnvCfg(SynriaTabletopPickPlaceEnvCfg):
    """Synria + Ludo tabletop pick-and-place env."""
    _game: str = "ludo"

    def _bind_scene(self) -> None:
        self.scene.tabletop.spawn.usd_path = str(_SCENE_USD["ludo"])
        # TODO(RTX): seed initial state with 16 tokens in home quadrants
        # + 1 die adjacent to the board. Use _synria_scene_common's
        # add_staging_zones return value to know zone XY targets.


@configclass
class SynriaChessPickPlaceEnvCfg(SynriaTabletopPickPlaceEnvCfg):
    """Synria + chess tabletop pick-and-place env."""
    _game: str = "chess"

    def _bind_scene(self) -> None:
        self.scene.tabletop.spawn.usd_path = str(_SCENE_USD["chess"])
        # TODO(RTX): seed initial state with 32 pieces in standard opening.


@configclass
class SynriaCheckersPickPlaceEnvCfg(SynriaTabletopPickPlaceEnvCfg):
    """Synria + checkers tabletop pick-and-place env."""
    _game: str = "checkers"

    def _bind_scene(self) -> None:
        self.scene.tabletop.spawn.usd_path = str(_SCENE_USD["checkers"])
        # TODO(RTX): seed initial state with 24 discs on dark squares of
        # ranks 1-3 (white) and 6-8 (black).

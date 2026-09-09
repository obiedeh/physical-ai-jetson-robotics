"""Fixed-conditions evaluation harness for the Synria cup task.

Runs a checkpoint DETERMINISTICALLY (inference policy, no exploration
noise) on scratch-only episodes (curriculum staging disabled), fixed seed,
fixed horizon, and reports the eight user-specified stage success rates:

    reach      : grasp centre came within REACH_R of the cup this episode
    align      : grasp centre XY-aligned over the cup, low, hand open
    grasp      : PICK -> CARRY phase transition (grasp confirmed)
    lift       : cup held >= 2 cm for >= 10 consecutive steps
    transport  : CARRY -> SETDOWN transition (carry requirement met)
    place      : SETDOWN -> RETURN transition (gentle resting release)
    upright    : placement with cup axis within 15 deg of vertical
    full       : completed sequence (returned home; cycle counter)

Per-episode flags are latched between resets; totals are normalised by
completed episode count. This harness is the ONLY metric source for
keep/revise/revert decisions in the experiment ledger.

Usage:
    $ISAAC_PYTHON isaac/scripts/eval_synria_sequence.py --headless \
        --checkpoint <model.pt> [--num_envs 256] [--steps 1700] [--seed 123]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Sibling module: pure Python, no torch/Isaac, so importing it before
# AppLauncher is safe (unlike anything under isaaclab_tasks, whose package
# __init__ pulls in isaaclab and segfaults Kit if imported too early).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stats import spread, wilson_interval  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REACH_R = 0.10          # m: reach = grasp centre within this of the cup
ALIGN_XY = 0.025        # m: XY alignment tolerance over the cup
# E27c: 0.05 was a 50 mm band on a 50 mm cup — it could not tell a body
# grasp from a rim grasp, which is why `align` read 0.118 and was dismissed
# as soft colour while the height error it should have caught was capping
# the whole task. Upright grasps sit at 9.9 mm median, tilted at 43.3 mm;
# 0.02 separates them and is inside the cup's 25 mm half-height.
# NOTE: `align` numbers before 2026-08-06 are NOT comparable to later ones.
# `align_legacy` preserves the old 0.05 band so the historical series stays
# readable.
ALIGN_Z = 0.02          # m: grasp centre height band for alignment
ALIGN_Z_LEGACY = 0.05   # m: pre-E27c band, reported for continuity only
UPRIGHT_COS = 0.966     # cos(15 deg)
LIFT_STEPS = 10         # consecutive lifted steps that count as "lift"


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval_synria_sequence")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1700)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the JSON report (default: "
        "reports/eval/<checkpoint-dir-name>.json).",
    )
    parser.add_argument(
        "--takeover_stage",
        choices=("none", "release"),
        default="none",
        help="'release': once the policy reaches SETDOWN carrying the cup, "
             "hand the arm+gripper to a scripted release (lower to the plate, "
             "open). Measures the full-cycle ceiling if the ONLY thing fixed "
             "is the release — E26 found the policy arrives at_plate 0.754 but "
             "places 0.132, and that compute buys nothing at the release.",
    )
    parser.add_argument(
        "--takeover_gain", type=float, default=0.6,
        help="proportional gain on the scripted descent (fraction of the "
             "remaining error commanded per step)",
    )
    parser.add_argument(
        "--takeover_tol", type=float, default=0.012,
        help="m: cup-to-target distance at which the scripted release opens "
             "the gripper",
    )
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    # SCRATCH-ONLY: disable curriculum staging so the eval measures the
    # real task from reset every episode.
    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg(
        "Synria-Chess-PickPlace-v0", device="cuda", num_envs=args.num_envs
    )
    env_cfg.seed = args.seed
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    wrapper = RslRlVecEnvWrapper(raw)
    rc = TASK_TRAIN_CFG["Synria-Chess-PickPlace-v0"]()
    runner = OnPolicyRunner(wrapper, rc.to_dict(), log_dir=None, device="cuda")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda")
    print(f"[eval] checkpoint: {args.checkpoint}")
    print(f"[eval] num_envs={args.num_envs} steps={args.steps} seed={args.seed}")

    env = raw.unwrapped
    mgr = mdp._get_trial_mgr(env)
    dev = env.device
    n = env.num_envs
    zeros_b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731

    # Per-episode latched stage flags.
    f = {k: zeros_b() for k in
         ("reach", "align", "grasp", "grasp_upright", "lift", "lift_upright",
          "transport", "transport_upright", "at_plate",
          "at_plate_upright", "place", "upright", "full", "align_legacy")}
    totals = {k: 0 for k in f}
    episodes_done = 0
    # SPAWN-BIN VERIFICATION: cup spawn distance from the training arm
    # base (-0.22, 0); near/far split at 0.35 m (spawns are bimodal:
    # 0.13-0.27 m vs 0.46-0.55 m).
    base_xy = torch.tensor([-0.22, 0.0], device=dev)
    ep_spawn_dist = torch.norm(mgr.origin_xy - base_xy, dim=-1)
    bin_tot = {"near": 0, "far": 0}
    bin_grasp = {"near": 0, "far": 0}
    bin_full = {"near": 0, "far": 0}
    # E27c — APPROACH GEOMETRY vs TILT. E27b showed the cup is already
    # tilted at the grasp transition, but measuring alignment AT that
    # transition is too late: grasp_confirmed requires a lift, so by then
    # the cup has been held (and possibly tipped) for many steps. Instead
    # track approach quality CONTINUOUSLY while the hand is still open, and
    # snapshot it at the moment the grasp fires. Then split the population
    # by whether that grasp came out upright.
    # E27c v2. The first version tracked the closest approach ONLY while the
    # hand was open (width >= 0.04) and produced medians of 113-193 mm --
    # impossible for a successful grasp on a 40 mm cup. That is the finding,
    # not a glitch: the hand is largely NOT OPEN during the approach, so the
    # open-hand series sampled far-away moments and the comparison was
    # meaningless. Track BOTH series now, plus the gripper width at closest
    # approach, and emit one row PER EPISODE so every conditional can be
    # computed offline without another sim run -- which also removes the
    # OR-latching caveat that limited E27/E27b.
    best_dxy = torch.full((n,), 1e9, device=dev)   # closest approach, ANY width
    best_dz = torch.zeros(n, device=dev)
    best_w = torch.zeros(n, device=dev)            # gripper width at that moment
    best_dxy_open = torch.full((n,), 1e9, device=dev)   # closest with hand OPEN
    g_dxy = torch.full((n,), float("nan"), device=dev)  # state AT the grasp instant
    g_dz = torch.full((n,), float("nan"), device=dev)
    g_w = torch.full((n,), float("nan"), device=dev)
    ep_rows: list = []
    lift_run = torch.zeros(n, dtype=torch.long, device=dev)
    prev_phase = mgr.phase.clone()
    prev_cycles = mgr.cycles.clone()
    prev_ep_len = env.episode_length_buf.clone()
    rest_z = TABLE_SURFACE_Z + 0.05 / 2.0

    # ------------------------------------------------------------------
    # SCRIPTED RELEASE TAKEOVER
    # ------------------------------------------------------------------
    # E26 concluded that compute buys the whole approach-grasp-carry chain
    # and buys NOTHING at the release (at_plate 0.754 -> place 0.132, and
    # place did not move with 10k more iterations). The recorded successor
    # programme infers from that "replace the release with the verified
    # scripted expert" -- an inference nothing has tested. This measures
    # the ceiling of that idea before anyone trains for it.
    #
    # Method: while the policy is in SETDOWN still holding the cup, drive
    # the CUP (not the hand -- the cup is kinematically attached, and the
    # cup's pose is what the placement test reads) toward the zone plate at
    # resting height with a damped-least-squares Jacobian step, then open
    # the gripper. Detach fires at width >= 42 mm and the binary OPEN
    # command is 50 mm, so opening IS the release.
    robot = env.scene["robot"]
    tool_idx = mdp._robot_ids(env)["tool0"]
    n_arm = 6
    takeover_on = args.takeover_stage == "release"
    ep_takeover = zeros_b()          # latched: takeover engaged this episode
    totals_takeover = 0
    diag = {"shape_logged": False, "steps_engaged": 0, "drop_fired": 0,
            "reached_tol": zeros_b(), "opened": zeros_b()}
    # Post-open forensics: for envs where the scripted release actually
    # opened, which clause of release_at_zone() ever held?
    post = {k: zeros_b() for k in
            ("resting", "released", "gentle", "at_target", "upright", "carried")}
    min_err = torch.full((n,), 1e9, device=dev)
    hold_quat = torch.zeros((n, 4), device=dev)   # wrist orientation to hold
    hold_set = zeros_b()
    open_latched = zeros_b()
    rest_z_cup = TABLE_SURFACE_Z + 0.05 / 2.0

    def _scripted_release(act: "torch.Tensor") -> "torch.Tensor":
        """Overwrite `act` for envs in SETDOWN that are still holding the cup."""
        engaged = (
            (mgr.phase == int(TaskPhase.PICK_FROM_ZONE))
            & mgr.carried_aloft
            & mgr.attached
        )
        if not bool(engaged.any()):
            return act
        ep_takeover.__ior__(engaged)
        # Latch the wrist orientation the carry ended with. The cup is rigidly
        # attached, so holding this keeps it upright; the first version
        # controlled POSITION ONLY (3 Jacobian rows) and let the redundant
        # DOFs rotate the wrist freely during the descent, tipping the cup.
        # Forensics: upright held in only 58 of 134 envs that opened, and it
        # was the single dominant failure clause.
        newly = engaged & ~hold_set
        if bool(newly.any()):
            hold_quat[newly] = robot.data.body_quat_w[newly, tool_idx]
            hold_set.__ior__(newly)

        piece = mdp._get_piece_pos(env)                      # (N, 3) env-local
        # Target the ACTUAL success criteria, not a harder problem. A
        # placement counts anywhere within PLACEMENT_RADIUS_M (0.15) of the
        # zone with the cup resting (|z - rest| <= 0.02) and gentle. The
        # first version of this controller drove to the exact zone centre and
        # refused to open until within 12 mm of it -- over-constrained by an
        # order of magnitude, which is why only 19/204 envs ever opened and
        # the median closest approach was 86 mm. So: hold xy where the policy
        # already delivered the cup, and only pull laterally if it sits near
        # the edge of the radius. The descent in z is the real job.
        d_zone = torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1)
        want_xy = torch.where(
            (d_zone > 0.11).unsqueeze(-1), mgr.zone_xy, piece[:, :2]
        )
        target = torch.cat(
            [want_xy, torch.full_like(piece[:, :1], rest_z_cup)], dim=-1
        )
        err = target - piece                                 # (N, 3)

        # Jacobian of the wrist in the world frame. Rigid attachment means
        # the cup follows it; the small lever arm is absorbed by the
        # proportional loop rather than modelled.
        jac = robot.root_physx_view.get_jacobians()          # (N, nb-1, 6, ndof)
        if not diag["shape_logged"]:
            diag["shape_logged"] = True
            print(f"[diag] jacobian shape={tuple(jac.shape)} bodies={len(robot.body_names)} "
                  f"tool_idx={tool_idx} -> using jac index {tool_idx - 1}; "
                  f"dofs={robot.data.joint_pos.shape[-1]}", flush=True)
        # FULL 6-row Jacobian: position AND orientation, so the descent
        # cannot quietly rotate the cup out of upright.
        from isaaclab.utils.math import (  # type: ignore[import-not-found]
            axis_angle_from_quat, quat_conjugate, quat_mul,
        )

        # Holding the wrist orientation did NOT restore upright (58/137 both
        # before and after), because the cup is ALREADY tilted when the
        # takeover engages -- the carry delivers it that way and at_plate
        # never checked orientation. So level the CUP actively: compute the
        # minimal rotation taking the cup's own axis back to vertical and
        # apply it to the wrist (rigid attachment -> same angular velocity).
        cup_q = env.scene["piece"].data.root_quat_w
        _w, _x, _y, _z = cup_q[:, 0], cup_q[:, 1], cup_q[:, 2], cup_q[:, 3]
        up = torch.stack([2 * (_x * _z + _w * _y),
                          2 * (_y * _z - _w * _x),
                          1.0 - 2 * (_x * _x + _y * _y)], dim=-1)   # cup axis
        zw = torch.zeros_like(up)
        zw[:, 2] = 1.0
        axis = torch.cross(up, zw, dim=-1)
        s_norm = torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
        angle = torch.atan2(s_norm.squeeze(-1), (up * zw).sum(-1)).unsqueeze(-1)
        ang_err = (axis / s_norm) * angle
        twist = torch.cat([err, 0.8 * ang_err], dim=-1).unsqueeze(-1)   # (N,6,1)
        j = jac[:, tool_idx - 1, :, :n_arm]                   # all 6 rows
        lam = 0.05
        jt = j.transpose(1, 2)
        reg = (lam ** 2) * torch.eye(6, device=j.device).unsqueeze(0)
        dq = (jt @ torch.linalg.solve(j @ jt + reg, twist)).squeeze(-1)
        dq = torch.clamp(dq * args.takeover_gain, -0.05, 0.05)

        q_arm = robot.data.joint_pos[:, :n_arm]
        new_act = act.clone()
        new_act[:, :n_arm] = torch.where(
            engaged.unsqueeze(-1), q_arm + dq, act[:, :n_arm]
        )

        # Open once the cup is on the plate and settled; keep closed while
        # still descending. Gripper channel is the BinaryJointPositionAction:
        # > 0 opens (50 mm), < 0 closes.
        try:
            v = torch.norm(env.scene["piece"].data.root_lin_vel_w, dim=-1)
        except (KeyError, AttributeError):
            v = torch.zeros_like(err[:, 0])
        # Release condition mirrors release_at_zone(): resting, inside the
        # placement radius, moving slowly. Nothing tighter.
        z_err = (piece[:, 2] - rest_z_cup).abs()
        e_norm = torch.norm(err, dim=-1)
        diag["steps_engaged"] += int(engaged.sum())
        min_err.copy_(torch.where(engaged, torch.minimum(min_err, z_err), min_err))
        cup_up_z = 1.0 - 2 * (cup_q[:, 1] ** 2 + cup_q[:, 2] ** 2)
        ready = (z_err <= 0.015) & (d_zone <= 0.14) & (cup_up_z >= UPRIGHT_COS)
        diag["reached_tol"] |= engaged & ready
        # LATCH the open. Recomputing it every step let the command flip back
        # to CLOSE whenever the cup twitched out of tolerance, so the fingers
        # oscillated and never crossed the 42 mm detach threshold -- 28 of 134
        # envs never actually released.
        open_latched.__ior__(engaged & ready & (v <= 0.05))
        drop_now = engaged & open_latched
        diag["drop_fired"] += int(drop_now.sum())
        diag["opened"] |= drop_now
        new_act[:, n_arm] = torch.where(
            drop_now, torch.ones_like(act[:, n_arm]),
            torch.where(engaged, -torch.ones_like(act[:, n_arm]), act[:, n_arm]),
        )
        return new_act

    obs = wrapper.get_observations()
    if takeover_on:
        act_probe = policy(obs)
        print(f"[eval] TAKEOVER=release engaged; action dim={act_probe.shape[-1]} "
              f"(expect {n_arm} arm + 1 gripper)", flush=True)
        assert act_probe.shape[-1] == n_arm + 1, (
            f"action layout changed (dim {act_probe.shape[-1]}); the scripted "
            f"release writes indices 0..{n_arm} and must be re-checked"
        )
    for step in range(args.steps):
        with torch.inference_mode():
            act = policy(obs)
            if takeover_on:
                act = _scripted_release(act)
        obs, _, _, _ = wrapper.step(act)

        # Episode boundary: episode_length_buf went backwards -> env reset.
        ep_len = env.episode_length_buf
        finished = ep_len < prev_ep_len
        if bool(finished.any()):
            episodes_done += int(finished.sum())
            near = ep_spawn_dist < 0.35
            bin_tot["near"] += int((finished & near).sum())
            bin_tot["far"] += int((finished & ~near).sum())
            bin_grasp["near"] += int((f["grasp"] & finished & near).sum())
            bin_grasp["far"] += int((f["grasp"] & finished & ~near).sum())
            bin_full["near"] += int((f["full"] & finished & near).sum())
            bin_full["far"] += int((f["full"] & finished & ~near).sum())
            # Snapshot the latched flags BEFORE clearing them -- the first
            # version of the per-episode dump read them after this loop and
            # recorded every stage as False.
            _snap = {k: f[k].clone() for k in f}
            for k in f:
                totals[k] += int((f[k] & finished).sum())
                f[k] = f[k] & ~finished
            totals_takeover += int((ep_takeover & finished).sum())
            ep_takeover &= ~finished
            hold_set &= ~finished
            open_latched &= ~finished
            # ONE ROW PER COMPLETED EPISODE. Every conditional in E27/E27b
            # was computed from OR-latched flags, so "placed given upright at
            # plate" was an indicator rather than a real conditional. These
            # rows carry each episode's facts together, so the conditionals
            # are exact.
            _fi = torch.nonzero(finished).squeeze(-1)
            for _i in _fi.tolist():
                ep_rows.append({
                    "grasp": bool(_snap["grasp"][_i]),
                    "grasp_upright": bool(_snap["grasp_upright"][_i]),
                    "align": bool(_snap["align"][_i]),
                    "at_plate": bool(_snap["at_plate"][_i]),
                    "at_plate_upright": bool(_snap["at_plate_upright"][_i]),
                    "place": bool(_snap["place"][_i]),
                    "full": bool(_snap["full"][_i]),
                    "min_dxy_mm": round(float(best_dxy[_i]) * 1000, 2)
                    if float(best_dxy[_i]) < 1e8 else None,
                    "dz_at_min_mm": round(float(best_dz[_i]) * 1000, 2),
                    "width_at_min_mm": round(float(best_w[_i]) * 1000, 2),
                    "min_dxy_open_mm": round(float(best_dxy_open[_i]) * 1000, 2)
                    if float(best_dxy_open[_i]) < 1e8 else None,
                    "grasp_dxy_mm": round(float(g_dxy[_i]) * 1000, 2)
                    if g_dxy[_i] == g_dxy[_i] else None,
                    "grasp_dz_mm": round(float(g_dz[_i]) * 1000, 2)
                    if g_dz[_i] == g_dz[_i] else None,
                    "grasp_width_mm": round(float(g_w[_i]) * 1000, 2)
                    if g_w[_i] == g_w[_i] else None,
                })
            lift_run[finished] = 0
            _big = torch.full_like(best_dxy, 1e9)
            _nan = torch.full_like(g_dxy, float("nan"))
            best_dxy = torch.where(finished, _big, best_dxy)
            best_dxy_open = torch.where(finished, _big, best_dxy_open)
            g_dxy = torch.where(finished, _nan, g_dxy)
            g_dz = torch.where(finished, _nan, g_dz)
            g_w = torch.where(finished, _nan, g_w)
            # New episode began in these envs: capture its spawn distance.
            ep_spawn_dist = torch.where(
                finished, torch.norm(mgr.origin_xy - base_xy, dim=-1), ep_spawn_dist
            )
        prev_ep_len = ep_len.clone()

        piece = mdp._get_piece_pos(env)
        gc = mdp._grasp_center_local(env)
        width = mdp._gripper_width_m(env).squeeze(-1)
        phase = mgr.phase
        lifted = mdp._piece_lifted_mask(env)

        # ORIENTATION ATTRITION PROBE (E27b). E27 showed 74.9% of plate
        # arrivals are tilted; this localises WHERE the tilt appears by
        # checking the same upright test at every stage transition. The cup
        # is kinematically attached at grasp, so orientation is inherited
        # from the hand: tilt present at grasp was picked up tilted, tilt
        # appearing later was introduced by wrist rotation in transit.
        try:
            _cq = env.scene["piece"].data.root_quat_w
            cup_upright = (1.0 - 2.0 * (_cq[:, 1] ** 2 + _cq[:, 2] ** 2)) >= UPRIGHT_COS
        except (KeyError, AttributeError):
            cup_upright = torch.ones(n, dtype=torch.bool, device=dev)

        f["reach"] |= torch.norm(gc - piece, dim=-1) <= REACH_R
        _xy_ok = torch.norm(gc[:, :2] - piece[:, :2], dim=-1) <= ALIGN_XY
        _dz = (gc[:, 2] - piece[:, 2]).abs()
        _open = width >= 0.04
        f["align"] |= _xy_ok & (_dz <= ALIGN_Z) & _open
        f["align_legacy"] |= _xy_ok & (_dz <= ALIGN_Z_LEGACY) & _open
        # Closest approach, unconditioned on gripper state.
        _dxy = torch.norm(gc[:, :2] - piece[:, :2], dim=-1)
        _better = _dxy < best_dxy
        best_dxy = torch.where(_better, _dxy, best_dxy)
        best_dz = torch.where(_better, gc[:, 2] - piece[:, 2], best_dz)
        best_w = torch.where(_better, width, best_w)
        _bo = (width >= 0.04) & (_dxy < best_dxy_open)
        best_dxy_open = torch.where(_bo, _dxy, best_dxy_open)

        _grasp_now = (prev_phase == int(TaskPhase.PICK_FROM_BOARD)) & (
            phase == int(TaskPhase.PLACE_ON_ZONE)
        )
        if bool(_grasp_now.any()):
            g_dxy = torch.where(_grasp_now, _dxy, g_dxy)
            g_dz = torch.where(_grasp_now, gc[:, 2] - piece[:, 2], g_dz)
            g_w = torch.where(_grasp_now, width, g_w)
        f["grasp"] |= _grasp_now
        f["grasp_upright"] |= _grasp_now & cup_upright
        lift_run = torch.where(lifted, lift_run + 1, torch.zeros_like(lift_run))
        f["lift"] |= lift_run >= LIFT_STEPS
        f["lift_upright"] |= (lift_run >= LIFT_STEPS) & cup_upright
        _transport_now = (prev_phase == int(TaskPhase.PLACE_ON_ZONE)) & (
            phase == int(TaskPhase.PICK_FROM_ZONE)
        )
        f["transport"] |= _transport_now
        f["transport_upright"] |= _transport_now & cup_upright
        # At-plate arrival: carried cup within the placement radius during
        # SETDOWN — separates "never arrives" from "arrives, never releases".
        f["at_plate"] |= (
            (phase == int(TaskPhase.PICK_FROM_ZONE))
            & lifted
            & (torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1) <= 0.15)
        )
        # AT-PLATE ORIENTATION. at_plate checks position and lift and never
        # orientation, so "the policy arrives at the plate" (0.754) has been
        # carrying more weight than it can bear: a tilted arrival cannot be
        # placed no matter how good the release is, because release_at_zone()
        # requires upright. Measuring the split here decides whether the
        # bottleneck is the release at all.
        _at_plate_now = (
            (phase == int(TaskPhase.PICK_FROM_ZONE))
            & lifted
            & (torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1) <= 0.15)
        )
        if bool(_at_plate_now.any()):
            try:
                _q = env.scene["piece"].data.root_quat_w
                _upz = 1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)
                f["at_plate_upright"] |= _at_plate_now & (_upz >= UPRIGHT_COS)
            except (KeyError, AttributeError):
                f["at_plate_upright"] |= _at_plate_now

        place_now = (prev_phase == int(TaskPhase.PICK_FROM_ZONE)) & (
            phase == int(TaskPhase.RETURN_TO_BOARD)
        )
        f["place"] |= place_now
        if bool(place_now.any()):
            try:
                quat = env.scene["piece"].data.root_quat_w  # (N, 4) w,x,y,z
                w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                up_z = 1.0 - 2.0 * (x * x + y * y)  # z-component of rotated z-axis
                f["upright"] |= place_now & (up_z >= UPRIGHT_COS)
            except (KeyError, AttributeError):
                f["upright"] |= place_now
        f["full"] |= (mgr.cycles - prev_cycles) > 0
        if takeover_on and bool(diag["opened"].any()):
            _p = mdp._get_piece_pos(env)
            _rz = TABLE_SURFACE_Z + 0.05 / 2.0
            _o = diag["opened"] & (mgr.phase == int(TaskPhase.PICK_FROM_ZONE))
            post["resting"] |= _o & ((_p[:, 2] - _rz).abs() <= 0.02)
            post["released"] |= _o & ~mgr.attached
            try:
                _v = torch.norm(env.scene["piece"].data.root_lin_vel_w, dim=-1)
            except (KeyError, AttributeError):
                _v = torch.zeros_like(_p[:, 0])
            post["gentle"] |= _o & (_v <= 0.10)
            post["at_target"] |= _o & (
                torch.norm(_p[:, :2] - mgr.zone_xy, dim=-1) <= 0.15)
            try:
                _q = env.scene["piece"].data.root_quat_w
                post["upright"] |= _o & (
                    (1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)) >= UPRIGHT_COS)
            except (KeyError, AttributeError):
                post["upright"] |= _o
            post["carried"] |= _o & mgr.carried_aloft
        prev_phase = phase.clone()
        prev_cycles = mgr.cycles.clone()
        if step % 500 == 0:
            print(f"[eval] step={step} episodes_done={episodes_done}", flush=True)

    # flush=True on EVERY line of this block. Without it the result table is
    # block-buffered into a redirected stdout and Kit's teardown discards the
    # buffer, so the run exits 0 having silently thrown away the numbers --
    # observed 2026-08-05 on the E23 eval. This harness is the only metric
    # source for keep/revert decisions; losing its output must not be quiet.
    print("[eval] ======== RESULT ========", flush=True)
    print(f"[eval] completed_episodes: {episodes_done}", flush=True)
    stages = (
        "reach", "align", "align_legacy", "grasp", "grasp_upright", "lift", "lift_upright",
        "transport", "transport_upright", "at_plate", "at_plate_upright",
        "place", "upright", "full",
    )
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "completed_episodes": episodes_done,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "eval_seed": args.seed,  # NOT the training seed; runs differ by checkpoint
        "takeover_stage": args.takeover_stage,
        "takeover_engaged_episodes": totals_takeover,
        "stages": {},
    }
    if takeover_on:
        # If the takeover rarely engaged, the run says nothing about the
        # release -- it says the policy rarely got there. Report it next to
        # the stages so the two cannot be confused.
        rate = totals_takeover / episodes_done if episodes_done else 0.0
        print(f"[eval] TAKEOVER engaged in {totals_takeover}/{episodes_done} "
              f"episodes ({rate:.3f}) — the scripted release can only be "
              f"credited where it actually ran", flush=True)
        finite = min_err[min_err < 1e8]
        print(f"[diag] controller: steps_engaged={diag['steps_engaged']} "
              f"drop_fired={diag['drop_fired']} "
              f"envs_that_reached_tol={int(diag['reached_tol'].sum())} "
              f"envs_that_opened={int(diag['opened'].sum())}", flush=True)
        n_open = int(diag["opened"].sum())
        print(f"[diag] post-open clause satisfaction (of {n_open} envs that "
              f"opened): " + "  ".join(
                  f"{k}={int(v.sum())}" for k, v in post.items()), flush=True)
        if finite.numel():
            q = torch.quantile(finite, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0],
                                                    device=finite.device))
            print(f"[diag] closest cup-to-target distance reached (m): "
                  f"min={q[0]:.4f} p25={q[1]:.4f} median={q[2]:.4f} "
                  f"p75={q[3]:.4f} max={q[4]:.4f}  (tol={args.takeover_tol})",
                  flush=True)
    if episodes_done:
        # Wilson intervals, not the normal approximation: the stage that
        # decides things here is `place`, which has come in at 1/259 and 4/263,
        # and the textbook interval reports a NEGATIVE lower bound at 1/259.
        for k in stages:
            iv = wilson_interval(int(totals[k]), episodes_done)
            report["stages"][k] = {
                "successes": int(totals[k]),
                "trials": episodes_done,
                "rate": iv.rate,
                "ci95_low": iv.low,
                "ci95_high": iv.high,
            }
            print(
                f"[eval] stage {k}: {totals[k]}/{episodes_done} = "
                f"{iv.rate:.3f}  95% CI [{iv.low:.3f}, {iv.high:.3f}]",
                flush=True,
            )
        print(
            "[eval] NOTE: the CIs above are SAMPLING error for this one run. "
            "Seed variance is much larger -- E25 measured 2.8x on grasp and "
            "~18x on full across seeds at a fixed budget. A single run cannot "
            "support a keep/revert decision; use aggregate_evals.py over >=3 "
            "seeds.",
            flush=True,
        )
        for b in ("near", "far"):
            n = bin_tot[b]
            if n:
                print(
                    f"[eval] spawn-bin {b}: episodes={n} "
                    f"grasp={bin_grasp[b]}/{n}={bin_grasp[b] / n:.3f} "
                    f"full={bin_full[b]}/{n}={bin_full[b] / n:.3f}",
                    flush=True,
                )

    # Persist the report. Until now the only record of an eval was its stdout,
    # which is how one run's numbers were lost outright to buffering and why
    # every cross-run comparison meant grepping logs by hand.
    report["episodes"] = ep_rows
    _g = [r for r in ep_rows if r["grasp"]]
    if _g:
        def _q(v, p):
            v = sorted(x for x in v if x is not None)
            return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

        up = [r for r in _g if r["grasp_upright"]]
        ti = [r for r in _g if not r["grasp_upright"]]
        print(f"[appr] grasps={len(_g)} upright={len(up)} tilted={len(ti)}", flush=True)
        for nm, grp in (("UPRIGHT", up), ("TILTED ", ti)):
            if not grp:
                continue
            print(f"[appr] {nm}: min_dxy(any) median={_q([r['min_dxy_mm'] for r in grp], .5):.1f}mm "
                  f"| width@min median={_q([r['width_at_min_mm'] for r in grp], .5):.1f}mm "
                  f"| dz@min median={_q([r['dz_at_min_mm'] for r in grp], .5):.1f}mm "
                  f"| dxy@grasp median={_q([r['grasp_dxy_mm'] for r in grp], .5):.1f}mm "
                  f"| width@grasp median={_q([r['grasp_width_mm'] for r in grp], .5):.1f}mm",
                  flush=True)

    out = args.out or (
        _REPO_ROOT / "reports" / "eval" / f"{Path(args.checkpoint).parent.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[eval] report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

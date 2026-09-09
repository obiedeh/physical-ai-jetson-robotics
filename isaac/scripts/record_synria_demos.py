"""Hybrid expert demonstrator + transition recorder for the Synria cup task.

The trained policy already grasps at ~78% and completes the 5 s carry at
~41% (fixed harness, E6) — its broken legs are the gentle set-down and
the return home. This expert composes both strengths, with NO IK:

    PICK + CARRY : trained policy (deterministic), which naturally
                   satisfies the 5 s + path carry requirement
    LOWER        : joint-space interpolation back to the arm pose
                   snapshotted at the grasp moment — which by
                   construction holds the cup at table height
    OPEN         : open fingers, arm held
    RETREAT      : zero action (= default pose under use_default_offset)
                   until the cycle completes, then hand back to policy

Recorded (obs, action) pairs live in the trained action space and are
directly cloneable.

Usage:
    $ISAAC_PYTHON isaac/scripts/record_synria_demos.py --headless \
        --checkpoint <model.pt> [--num_envs 64] [--steps 2000] [--seed 7] \
        [--out demos.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Expert modes: the policy flies the arm at ALL times except the
# zero-action retreat; the script only forces the fingers open once
# the cup is low (ASSIST) — the two atomic skills RL never learned.
POLICY, WANDER, RELEASE, RETREAT = range(4)

YAW_AMP = 0.35          # rad: base-yaw wander amplitude (horizontal arc)
YAW_PERIOD = 60         # steps per wander cycle
LIFT_SOLID = 0.05       # m above rest that counts as a solid lift


def main() -> int:
    parser = argparse.ArgumentParser(prog="record_synria_demos")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="reports/demos/synria_cup_demos.pt")
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
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PIECE_HEIGHT_M,
        TABLE_SURFACE_Z,
    )
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    # Scratch-only: the expert performs the full task from reset.
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
    print(f"[demo] policy loaded from {args.checkpoint}")

    env = raw.unwrapped
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    dev = env.device
    n = env.num_envs
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    default_arm = robot.data.default_joint_pos[:, arm_ids]
    default_l = robot.data.default_joint_pos[:, rids["left"]]
    default_r = robot.data.default_joint_pos[:, rids["right"]]

    mgr = mdp._get_trial_mgr(env)
    mode = torch.zeros(n, dtype=torch.long, device=dev)
    freeze_pose = default_arm.clone()
    hold_l = torch.full((n,), 0.02, device=dev)
    mode_t = torch.zeros(n, dtype=torch.long, device=dev)
    prev_phase = mgr.phase.clone()
    prev_cycles = mgr.cycles.clone()
    prev_ep = env.episode_length_buf.clone()

    obs_buf, act_buf, done_buf = [], [], []
    completed = 0
    obs = wrapper.get_observations()

    for step in range(args.steps):
        piece = mdp._get_piece_pos(env)
        phase = mgr.phase

        # --- mode transitions -----------------------------------------
        # Assist takeover when the carry requirement is met (SETDOWN):
        # the policy keeps flying the arm; the script only opens fingers.
        # Take over right after a SOLID LIFT: the policy's own carry drags
        # the cup low with the flange grinding the table, crush-jamming the
        # fingers. The scripted wander instead sweeps base yaw at the
        # frozen lift pose — a constant-height horizontal arc that cannot
        # press into the table and racks up carry path safely.
        take = (
            (mode == POLICY)
            & (phase == int(TaskPhase.PLACE_ON_ZONE))
            & ((piece[:, 2] - rest_z) >= LIFT_SOLID)
            & (mgr.hold_steps >= 20)
        )
        if bool(take.any()):
            freeze_pose[take] = robot.data.joint_pos[take][:, arm_ids]
            hold_l[take] = robot.data.joint_pos[take, rids["left"]].clamp(0.0155, 0.025)
            mode[take], mode_t[take] = WANDER, 0
        # Trial machine flips CARRY -> SETDOWN when duration+path are met.
        m = (mode == WANDER) & (phase == int(TaskPhase.PICK_FROM_ZONE))
        if bool(m.any()):
            mode[m], mode_t[m] = RELEASE, 0
        # Wander failed (cup slipped out mid-arc): back to the policy.
        m = (mode == WANDER) & ((piece[:, 2] - rest_z) < 0.02) & (mode_t > 10)
        mode[m], mode_t[m] = POLICY, 0
        width_now = mdp._gripper_width_m(env).squeeze(-1)
        released_low = (width_now >= 0.038) & ((piece[:, 2] - rest_z).abs() < 0.02)
        m = (mode == RELEASE) & (released_low | (mode_t >= 60))
        mode[m], mode_t[m] = RETREAT, 0
        # Cycle completed (return-home fired): hand back to the policy.
        cyc = (mgr.cycles - prev_cycles) > 0
        m = (mode == RETREAT) & cyc
        mode[m], mode_t[m] = POLICY, 0
        mode_t += 1

        # --- compose action ---------------------------------------------
        with torch.inference_mode():
            pol_act = policy(obs)
        action = pol_act.clone()

        # WANDER: frozen lift pose + sinusoidal base-yaw sweep; grip held
        # at the policy's own width (clamped into the action clip).
        m = mode == WANDER
        if bool(m.any()):
            yaw = YAW_AMP * torch.sin(mode_t.float() * 6.28318 / YAW_PERIOD)
            wt = freeze_pose.clone()
            wt[:, 0] = wt[:, 0] + yaw
            action[m, :6] = (wt - default_arm)[m]
            action[m, 6] = (hold_l - default_l)[m]
            action[m, 7] = (-hold_l - default_r)[m]
        # RELEASE: arm frozen at the wander pose; fingers forced open.
        m = mode == RELEASE
        if bool(m.any()):
            wt = freeze_pose.clone()
            action[m, :6] = (wt - default_arm)[m]
            action[m, 6] = 0.025 - default_l[m]
            action[m, 7] = -0.025 - default_r[m]
        # RETREAT: zero action = default pose (arm home, fingers open).
        m = mode == RETREAT
        if bool(m.any()):
            action[m] = 0.0

        obs_buf.append(obs["policy"].clone() if isinstance(obs, dict) else obs.clone())
        act_buf.append(action.clone())

        obs, _, _, _ = wrapper.step(action)

        # --- bookkeeping --------------------------------------------------
        ep = env.episode_length_buf
        finished = ep < prev_ep
        done_buf.append(finished.clone())
        if bool(finished.any()):
            mode[finished] = POLICY
            mode_t[finished] = 0
        prev_ep = ep.clone()
        completed += int((mgr.cycles - prev_cycles).clamp(min=0).sum())
        prev_cycles = mgr.cycles.clone()
        prev_phase = phase.clone()
        if step % 200 == 0:
            counts = [int((mode == p).sum()) for p in range(4)]
            pcounts = [int((mgr.phase == p).sum()) for p in range(5)]
            print(
                f"[demo] step={step} sequences_completed={completed} "
                f"modes(policy/wander/release/retreat)={counts} phases={pcounts}",
                flush=True,
            )
            stuck = (mode == RETREAT) & (mgr.phase == int(TaskPhase.PICK_FROM_ZONE))
            if bool(stuck.any()):
                e = int(stuck.nonzero()[0])
                pz = float(piece[e, 2])
                w = float(mdp._gripper_width_m(env).squeeze(-1)[e])
                try:
                    v = float(torch.norm(env.scene["piece"].data.root_lin_vel_w[e]))
                except (KeyError, AttributeError):
                    v = -1.0
                lf = float(robot.data.joint_pos[e, rids["left"]])
                rf = float(robot.data.joint_pos[e, rids["right"]])
                try:
                    lt = float(robot.data.joint_pos_target[e, rids["left"]])
                    rt = float(robot.data.joint_pos_target[e, rids["right"]])
                except AttributeError:
                    lt = rt = float("nan")
                tool_z = float(robot.data.body_pos_w[e, rids["tool0"], 2] - env.scene.env_origins[e, 2])
                print(
                    f"[dbg] stuck env{e}: piece_z={pz:.3f} (rest {rest_z:.3f}) "
                    f"width={w*1000:.1f}mm L={lf*1000:.1f}mm R={rf*1000:.1f}mm "
                    f"Ltgt={lt*1000:.1f} Rtgt={rt*1000:.1f} tool_z={tool_z:.3f} vel={v:.3f} "
                    f"piece_xy=({float(piece[e,0]):.3f},{float(piece[e,1]):.3f})",
                    flush=True,
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs": torch.cat([o.reshape(n, -1) for o in obs_buf]).cpu(),
            "actions": torch.cat(act_buf).cpu(),
            "dones": torch.cat(done_buf).cpu(),
        },
        out,
    )
    ep_eq = n * args.steps / int(env.max_episode_length)
    print("[demo] ======== RESULT ========")
    print(
        f"[demo] sequences_completed: {completed} over ~{ep_eq:.0f} "
        f"episode-equivalents ({completed / ep_eq:.2f}/episode)"
    )
    print(f"[demo] saved {len(obs_buf) * n} transitions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

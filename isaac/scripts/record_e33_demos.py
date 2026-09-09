"""Record demos: RL policy picks + E33 scripted release — success-filtered.

Hybrid expert, same architecture as record_synria_lerobot_demos.py but
updated for the CURRENT 7-dim env (6 arm offsets + binary gripper) and with
the release replaced by the E33 cycle proven 12/12 (ledger E33):

    POLICY   : E26 checkpoint flies approach -> grasp -> lift -> carry
               (grasp chain measured at 0.79; this is what RL solved)
    TAKEOVER : at SETDOWN, still attached and carried (the state the policy
               reaches 0.74 of the time), the script takes the arm:
      PLACE_DESCEND : hold CURRENT xy and the takeover-time wrist quat,
                      descend straight down; stop at pad-tip table contact
                      (surface + 36.1 mm underhang + 1 mm). Flip the binary
                      gripper OPEN once the cup is within 12 mm of resting —
                      E33's open-while-descending, so the cup seats under
                      its own weight.
      RELEASE       : wait until the measured width actually exceeds the
                      cup (this project shipped an inverted release once;
                      polarity is probed at startup and the run aborts if
                      OPEN does not widen).
      ASCEND        : straight up, xy frozen — lateral motion is what
                      clips the rim.
      YAW_HOME      : rotate Joint1 home AT altitude, out of the cup's
                      sector.
      GO_HOME       : joint-space interpolation to the reset pose.

Episodes are saved ONLY when the env's own cycle counter fires (set-down at
the zone + return home) — the task's success definition, not the script's.
A full-scripted approach was tried first and abandoned: closed-loop DLS to
a point target with the production 5 Nm drooping arm oscillated (smoke runs
1-4, reports/logs/2026-08-06/e33rec_smoke*.log); the policy already solves
that part at 0.79 and there is no reason to re-solve it worse.

Output format matches the existing raw corpus (states (T,8), actions (T,8)
= 6 arm offsets + 2 finger-target offsets, wrist/overhead mp4, manifest),
so convert_synria_lerobot.py / validate_synria_lerobot.py work unchanged.

    ~/.venv/isaacsim5/bin/python isaac/scripts/record_e33_demos.py \
        --headless --checkpoint reports/training/synria_chess_pickplace_v6_e26_s43/model_final.pt \
        [--num_envs 16] [--steps 40000] [--max_episodes 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GRASP_Z = 0.040
PAD_UNDERHANG = 0.0361
CLEAR_RISE = 0.08        # ascend this far above the release before yawing
                         # (release pad z ~87mm; +80 puts the pad bottom at
                         # ~131mm, clear of the 100mm cup top)
OPEN_TRIGGER = 0.012
MAX_SEG_STEPS = 1400   # extended: the E33 exit needs ~300-400 steps after a
                       # late takeover; 900 was truncating mid-script
FPS = 30
INSTRUCTION = "pick up the cup, carry it to the plate, and set it down gently"

(POLICY, PLACE_DESCEND, RELEASE, ASCEND, YAW_HOME, GO_HOME,
 S_YAW, S_PLANE, S_DESCEND, S_CLOSE, S_LIFT, S_CYAW, S_CPLANE,
 S_OPEN) = range(14)
PHASE_NAMES = ["policy", "place_descend", "release", "ascend", "yaw_home",
               "go_home", "s_yaw", "s_plane", "s_descend", "s_close",
               "s_lift", "s_cyaw", "s_cplane", "s_open"]
BASE_XY = (-0.22, 0.0)          # arm base, env-local (spawn-bin reference)
PREGRASP_ALT = 0.13             # pad altitude during approach/carry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="required for --expert hybrid; unused for scripted")
    parser.add_argument("--expert", choices=("hybrid", "scripted", "hybrid2"),
                        default="hybrid",
                        help="'scripted' (E35): fully scripted smooth expert — "
                             "no policy anywhere; the corpus v1 lesson is that "
                             "imitating the RL policy's noisy approach kills "
                             "the clone by covariate shift")
    parser.add_argument("--gate_only", type=int, default=0,
                        help="E35 Phase A: run N scripted trials headless and "
                             "report grasp/upright/smoothness, record nothing")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_root", type=str,
                        default="reports/training/synria_e33_lerobot_raw")
    parser.add_argument("--no_cameras", action="store_true")
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.no_cameras:
        args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        import os

        os._exit(1)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab.utils.math import (  # type: ignore[import-not-found]
        axis_angle_from_quat, quat_conjugate, quat_mul,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    if not args.no_cameras:
        import cv2  # type: ignore[import-not-found]
        import imageio  # type: ignore[import-not-found]
        from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
            add_recorder_cameras,
        )

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
    # Give the script room: every mid-script death in smoke 3 was the
    # episode clock expiring during release/ascend/home, not a failure of
    # the cycle itself. Recorder-only override; the corpus consumer sees
    # episode boundaries from the manifest, not the env horizon.
    env_cfg.episode_length_s = env_cfg.episode_length_s * 1.6
    if not args.no_cameras:
        add_recorder_cameras(env_cfg)

    suppress_state: dict = {"mask": None}
    _orig_attach = mdp.attach_carried_pieces

    def _gated_attach(env, env_ids):
        mgr2 = mdp._get_trial_mgr(env)
        mask = suppress_state["mask"]
        if mask is None:
            return _orig_attach(env, env_ids)
        import torch as _t

        saved_pin = mgr2.pin_steps.clone()
        mgr2.attached = mgr2.attached & ~mask
        mgr2.pin_steps = _t.where(mask, _t.full_like(saved_pin, 9999), saved_pin)
        _orig_attach(env, env_ids)
        mgr2.pin_steps = saved_pin
        mgr2.attached = mgr2.attached & ~mask
        return None

    env_cfg.events.attach_carried.func = _gated_attach

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    wrapper = RslRlVecEnvWrapper(raw)
    policy = None
    if args.expert in ("hybrid", "hybrid2"):
        assert args.checkpoint, "--checkpoint required for hybrid expert"
        rc = TASK_TRAIN_CFG["Synria-Chess-PickPlace-v0"]()
        runner = OnPolicyRunner(wrapper, rc.to_dict(), log_dir=None, device="cuda")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device="cuda")
        print(f"[e33rec] policy: {args.checkpoint}", flush=True)
    else:
        print("[e33rec] fully scripted expert (E35)", flush=True)

    env = raw.unwrapped
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    lf, rf = int(rids["left"]), int(rids["right"])
    tool_idx = int(rids["tool0"])
    dev = env.device
    n = env.num_envs
    mgr = mdp._get_trial_mgr(env)
    rest_z = TABLE_SURFACE_Z + 0.05 / 2.0
    # +4 mm, not +1: the policy's wrist orientation at takeover is not
    # E33's, and the pad underhang grows with tilt -- at +1 mm the pads
    # sometimes bear on the table and the 5 N fingers pin shut (three
    # smoke-4 deaths stuck in RELEASE with the width never opening).
    pad_stop = TABLE_SURFACE_Z + PAD_UNDERHANG + 0.004
    default_arm = robot.data.default_joint_pos[:, arm_ids].clone()
    default_l = float(robot.data.default_joint_pos[0, lf])
    default_r = float(robot.data.default_joint_pos[0, rf])
    jlim = torch.tensor(
        [[-2.749, 2.749], [-2.0, 2.0], [-0.5, 3.14159],
         [-2.79, 2.79], [-1.57, 1.57], [-3.14159, 3.14159]], device=dev)
    assert env.action_manager.total_action_dim == 7

    # ---- gripper polarity: measured, never assumed -------------------------
    def width_now():
        return mdp._gripper_width_m(env).squeeze(-1)

    probe = torch.zeros(n, 7, device=dev)
    probe[:, 6] = -1.0
    for _ in range(45):
        wrapper.step(probe)
    w_close = float(width_now()[0])
    probe[:, 6] = 1.0
    for _ in range(45):
        wrapper.step(probe)
    w_open = float(width_now()[0])
    print(f"[e33rec] polarity: -1 -> {w_close*1000:.1f} mm, +1 -> {w_open*1000:.1f} mm",
          flush=True)
    if not (w_open > w_close + 0.010):
        print("[e33rec] ABORT: +1 does not open the gripper", flush=True)
        import os
        os._exit(7)
    OPEN_CMD, CLOSE_CMD = 1.0, -1.0

    obs = wrapper.get_observations()

    phase = torch.full((n,), S_YAW if args.expert == "scripted" else POLICY,
                       dtype=torch.long, device=dev)
    phase_t = torch.zeros(n, dtype=torch.long, device=dev)
    arm_tgt = robot.data.joint_pos[:, arm_ids].clone()
    grip_cmd = torch.full((n,), CLOSE_CMD, device=dev)
    hold_quat = torch.zeros(n, 4, device=dev)
    hold_xy = torch.zeros(n, 2, device=dev)
    release_z = torch.zeros(n, device=dev)
    prev_cycles = mgr.cycles.clone()
    prev_ep = env.episode_length_buf.clone()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "defaults.json").write_text(json.dumps({
        "arm_default_rad": default_arm[0].cpu().tolist(),
        "left_finger_default_m": default_l,
        "right_finger_default_m": default_r,
        "expert": "hybrid: E26 policy pick + E33 scripted release",
        "checkpoint": str(args.checkpoint),
        "gripper_action": "binary; recorded dims 6,7 are finger position "
                          "targets as offsets from finger defaults",
        "joint_limits_rad": {
            "Joint1": [-2.749, 2.749], "Joint2": [-2.0, 2.0],
            "Joint3": [-0.5, 3.14159], "Joint4": [-2.79, 2.79],
            "Joint5": [-1.57, 1.57], "Joint6": [-3.14159, 3.14159]},
        "finger_limits_m": {"left": [0.0, 0.025], "right": [-0.025, 0.0]},
    }, indent=2))
    manifest = (out_root / "raw_manifest.jsonl").open("a")
    bufs: list[deque] = [deque(maxlen=MAX_SEG_STEPS + 1) for _ in range(n)]
    saved = 0
    funnel = {"takeover": 0, "released": 0, "setdown": 0, "cycle": 0,
              "reset_mid_script": 0}
    gate = {"eps": 0, "grasp": 0, "grasp_upright": 0, "act_delta_sum": 0.0,
            "act_steps": 0}
    prev_arm_tgt = arm_tgt.clone()
    prev_tphase = mgr.phase.clone()

    def grab_frames():
        outs = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"]
            fr = rgb.detach().cpu().numpy()
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
            outs.append(fr[..., :3])
        return outs[0], outs[1]

    def save_episode(e: int) -> None:
        nonlocal saved
        seg = list(bufs[e])
        if len(seg) < 60 or len(seg) > MAX_SEG_STEPS:
            return
        ep_dir = out_root / f"episode_{saved:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "states.npy",
                np.stack([s for s, _, _, _ in seg]).astype(np.float32))
        np.save(ep_dir / "actions.npy",
                np.stack([a for _, a, _, _ in seg]).astype(np.float32))
        if not args.no_cameras:
            for vid_idx, name in ((2, "wrist"), (3, "overhead")):
                writer = imageio.get_writer(
                    ep_dir / f"{name}.mp4", fps=FPS, codec="libx264",
                    quality=8, macro_block_size=1)
                for row in seg:
                    bgr = cv2.imdecode(row[vid_idx], cv2.IMREAD_COLOR)
                    writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                writer.close()
        manifest.write(json.dumps({
            "episode_index": saved, "length": len(seg), "env": e,
            "seed": args.seed, "fps": FPS, "instruction": INSTRUCTION,
            "expert": "E26+E33 hybrid",
            "checkpoint": str(args.checkpoint)}) + "\n")
        manifest.flush()
        saved += 1
        print(f"[e33rec] episode {saved}/{args.max_episodes} saved "
              f"(len={len(seg)}, env={e})", flush=True)

    def dls_step(target_pos, hq):
        gc2 = mdp._grasp_center_local(env)
        err = target_pos - gc2
        jac = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :, :6]
        cq = robot.data.body_quat_w[:, tool_idx]
        ae = axis_angle_from_quat(quat_mul(hq, quat_conjugate(cq)))
        twist = torch.cat([err, 0.8 * ae], dim=-1).unsqueeze(-1)
        reg = (0.05 ** 2) * torch.eye(6, device=dev).unsqueeze(0)
        jt_ = jac.transpose(1, 2)
        dq = (jt_ @ torch.linalg.solve(jac @ jt_ + reg, twist)).squeeze(-1)
        # descent stays gentle; the ASCEND leg is unloaded (cup released),
        # so it may move 3x faster -- the smoke run lost 30 of 35 takeovers
        # to episode timeout in the slow exit
        fast = (phase == ASCEND).unsqueeze(-1)
        dq = torch.where(fast, torch.clamp(dq * 1.2, -0.030, 0.030),
                         torch.clamp(dq * 0.6, -0.010, 0.010))
        return dq

    for step in range(args.steps):
        piece = mdp._get_piece_pos(env)
        gc = mdp._grasp_center_local(env)
        width = width_now()
        tphase = mgr.phase

        if args.expert in ("scripted", "hybrid2"):
            base = torch.tensor(BASE_XY, device=dev)

            def azim(xy):
                v = xy - base
                return torch.atan2(v[:, 1], v[:, 0])

            def radius(xy):
                return torch.norm(xy - base, dim=-1)

            az_pad, az_cup = azim(gc[:, :2]), azim(piece[:, :2])
            az_zone = azim(mgr.zone_xy)
            r_pad, r_cup = radius(gc[:, :2]), radius(piece[:, :2])
            r_zone = radius(mgr.zone_xy)

            def wrap(a):
                return torch.atan2(torch.sin(a), torch.cos(a))

            # ---- transitions ----
            if args.expert == "hybrid2":
                dxy_cup = torch.norm(gc[:, :2] - piece[:, :2], dim=-1)
                hover = (
                    (phase == POLICY)
                    & (tphase == int(TaskPhase.PICK_FROM_BOARD))
                    & (dxy_cup < 0.028)
                    & (gc[:, 2] > TABLE_SURFACE_Z + 0.058)
                    & (gc[:, 2] < TABLE_SURFACE_Z + 0.26)
                )
                if bool(hover.any()):
                    arm_tgt[hover] = robot.data.joint_pos[hover][:, arm_ids]
                phase[hover], phase_t[hover] = S_OPEN, 0
                # bounded dwell: 5 of 13 deaths sat in S_OPEN waiting for a
                # width that never crossed 44 mm while the episode expired
                m = (phase == S_OPEN) & (
                    ((width > 0.042) & (phase_t > 10)) | (phase_t > 80))
                phase[m], phase_t[m] = S_DESCEND, 0
            m = (phase == S_YAW) & (wrap(az_cup - az_pad).abs() < 0.02)
            phase[m], phase_t[m] = S_PLANE, 0
            m = (phase == S_PLANE) & ((r_pad - r_cup).abs() < 0.012) &                 ((gc[:, 2] - (TABLE_SURFACE_Z + PREGRASP_ALT)).abs() < 0.015)
            phase[m], phase_t[m] = S_DESCEND, 0
            m = (phase == S_DESCEND) &                 (torch.norm(gc[:, :2] - piece[:, :2], dim=-1) < 0.015) &                 ((gc[:, 2] - (TABLE_SURFACE_Z + GRASP_Z)).abs() < 0.006)
            phase[m], phase_t[m] = S_CLOSE, 0
            m = (phase == S_CLOSE) & (width < 0.045) & (phase_t > 40)
            phase[m], phase_t[m] = S_LIFT, 0
            # env confirms the grasp on lift (phase -> PLACE_ON_ZONE + attach)
            m = (phase == S_LIFT) & (piece[:, 2] > rest_z + 0.075) & mgr.attached
            phase[m], phase_t[m] = S_CYAW, 0
            m = (phase == S_CYAW) & (wrap(az_zone - az_pad).abs() < 0.03)
            phase[m], phase_t[m] = S_CPLANE, 0
            arrive = (phase == S_CPLANE) & ((r_pad - r_zone).abs() < 0.02)
            if bool(arrive.any()):
                hold_quat[arrive] = robot.data.body_quat_w[arrive, tool_idx]
                hold_xy[arrive] = gc[arrive, :2]
                funnel["takeover"] += int(arrive.sum())
            phase[arrive], phase_t[arrive] = PLACE_DESCEND, 0
            # stuck anywhere in the approach for 400 steps -> restart episode arc
            stuck = (phase >= S_YAW) & (phase_t > 600)
            phase[stuck] = S_YAW if args.expert == "scripted" else POLICY
            phase_t[stuck] = 0

        # ---- takeover: the state the policy reaches 0.74 of the time ----
        try:
            _q = env.scene["piece"].data.root_quat_w
            cup_upright = (1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)) >= 0.966
        except (KeyError, AttributeError):
            cup_upright = torch.ones(n, dtype=torch.bool, device=dev)
        take = (
            (phase == POLICY)
            & (tphase == int(TaskPhase.PICK_FROM_ZONE))
            & mgr.attached
            & mgr.carried_aloft
            & cup_upright
            & (torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1) < 0.13)
        )
        if bool(take.any()):
            hold_quat[take] = robot.data.body_quat_w[take, tool_idx]
            hold_xy[take] = gc[take, :2]
            arm_tgt[take] = robot.data.joint_pos[take][:, arm_ids]
            phase[take], phase_t[take] = PLACE_DESCEND, 0
            funnel["takeover"] += int(take.sum())

        pad_low = gc[:, 2] <= pad_stop + 0.0015
        cup_low = (piece[:, 2] - rest_z) < 0.004
        m = (phase == PLACE_DESCEND) & (pad_low | (cup_low & (phase_t > 90)))
        phase[m], phase_t[m] = RELEASE, 0
        # unpin escape: if the width has not opened after 150 steps the pads
        # are bearing on something -- ascend anyway with OPEN commanded;
        # rising unpins the pads and the cup (already near rest) settles
        stuck = (phase == RELEASE) & (phase_t > 150)
        if bool(stuck.any()):
            release_z[stuck] = gc[stuck, 2]
        phase[stuck], phase_t[stuck] = ASCEND, 0
        rel = (phase == RELEASE) & (width > 0.043) & (phase_t > 15)
        if bool(rel.any()):
            release_z[rel] = gc[rel, 2]
            funnel["released"] += int(rel.sum())
        phase[rel], phase_t[rel] = ASCEND, 0
        m = (phase == ASCEND) & (gc[:, 2] > release_z + CLEAR_RISE)
        phase[m], phase_t[m] = YAW_HOME, 0
        m = (phase == YAW_HOME) & (
            (arm_tgt[:, 0] - default_arm[:, 0]).abs() < 0.02)
        phase[m], phase_t[m] = GO_HOME, 0
        phase_t += 1

        scripted = (phase != POLICY) & (phase < S_YAW)  # release chain only
        # cup fully physical once the open has been commanded
        suppress_state["mask"] = scripted & (grip_cmd > 0)

        # ---- policy action ------------------------------------------------
        if policy is not None:
            with torch.inference_mode():
                pol_act = policy(obs)
            action = pol_act.clone()
        else:
            pol_act = torch.zeros(n, 7, device=dev)
            action = pol_act.clone()

        if args.expert == "scripted" and bool((phase >= S_YAW).any()):
            # YAW feedback runs in EVERY scripted phase, not just the yaw
            # ones: freezing Joint1 during the plane/descend phases let
            # elbow motion and droop swing the pad's azimuth by up to
            # 0.39 rad (~95 mm lateral at this radius), so CLOSE never
            # fired and the stuck-timer recycled the approach forever
            # (iteration-1 trace). Align, then HOLD.
            az_target = torch.where(phase >= S_CYAW, az_zone, az_cup)
            m_yaw = phase >= S_YAW
            if bool(m_yaw.any()):
                err = wrap(az_target - az_pad)
                arm_tgt[m_yaw, 0] += torch.clamp(0.5 * err[m_yaw], -0.02, 0.02)
            # PLANE phases: DLS restricted to the vertical plane (radial, z)
            m = (phase == S_PLANE) | (phase == S_DESCEND) | (phase == S_LIFT) |                 (phase == S_CPLANE) | (phase == S_CLOSE)
            if bool(m.any()):
                r_t = torch.where(phase == S_CPLANE, r_zone, r_cup)
                z_t = torch.full((n,), TABLE_SURFACE_Z + PREGRASP_ALT, device=dev)
                z_t = torch.where(phase == S_DESCEND,
                                  torch.full_like(z_t, TABLE_SURFACE_Z + GRASP_Z), z_t)
                z_t = torch.where((phase == S_CLOSE) | (phase == S_OPEN),
                                  gc[:, 2], z_t)  # hold still
                jac = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :3, :6]
                u = (gc[:, :2] - base)
                u = u / torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-6)
                Jr = (jac[:, 0, :] * u[:, 0:1] + jac[:, 1, :] * u[:, 1:2])
                Jz = jac[:, 2, :]
                J2 = torch.stack([Jr, Jz], dim=1)              # (N,2,6)
                e2 = torch.stack([r_t - r_pad, z_t - gc[:, 2]], dim=1).unsqueeze(-1)
                reg = (0.05 ** 2) * torch.eye(2, device=dev).unsqueeze(0)
                jt2 = J2.transpose(1, 2)
                dq2 = (jt2 @ torch.linalg.solve(J2 @ jt2 + reg, e2)).squeeze(-1)
                dq2 = torch.clamp(dq2 * 0.7, -0.012, 0.012)
                dq2[:, 0] = 0.0                                # yaw stays put
                arm_tgt = torch.where(m.unsqueeze(-1), arm_tgt + dq2, arm_tgt)
            meas0 = robot.data.joint_pos[:, arm_ids]
            arm_tgt = torch.clamp(arm_tgt, meas0 - 0.45, meas0 + 0.45)
            arm_tgt = arm_tgt.clamp(jlim[:, 0], jlim[:, 1])

        # ---- scripted arm target -----------------------------------------
        if bool(scripted.any()):
            tgt = gc.clone()
            m = phase == PLACE_DESCEND
            tgt[m, :2] = hold_xy[m]
            tgt[m, 2] = pad_stop
            m = phase == ASCEND
            tgt[m, :2] = hold_xy[m]
            tgt[m, 2] = release_z[m] + CLEAR_RISE + 0.03
            dq = dls_step(tgt, hold_quat)
            move = (phase == PLACE_DESCEND) | (phase == ASCEND)
            arm_tgt = torch.where(move.unsqueeze(-1), arm_tgt + dq, arm_tgt)
            m = phase == YAW_HOME
            if bool(m.any()):
                arm_tgt[m, 0] += (default_arm[m, 0] - arm_tgt[m, 0]) * 0.10
            m = phase == GO_HOME
            if bool(m.any()):
                arm_tgt[m] += (default_arm[m] - arm_tgt[m]) * 0.08
            meas = robot.data.joint_pos[:, arm_ids]
            arm_tgt = torch.clamp(arm_tgt, meas - 0.55, meas + 0.55)
            arm_tgt = arm_tgt.clamp(jlim[:, 0], jlim[:, 1])

        # ---- gripper ------------------------------------------------------
        open_now = (
            ((phase == PLACE_DESCEND) & ((piece[:, 2] - rest_z) < OPEN_TRIGGER))
            | ((phase >= RELEASE) & scripted)
        )
        grip_scripted = torch.where(
            open_now, torch.full_like(grip_cmd, OPEN_CMD),
            torch.full_like(grip_cmd, CLOSE_CMD))
        appr = phase >= S_YAW
        appr_grip = torch.where((phase >= S_CLOSE) & (phase != S_OPEN),
                                torch.full_like(grip_cmd, CLOSE_CMD),
                                torch.full_like(grip_cmd, OPEN_CMD))
        grip_cmd = torch.where(scripted, grip_scripted,
                               torch.where(appr, appr_grip,
                                           pol_act[:, 6] if policy is not None
                                           else appr_grip))
        drive = scripted | appr
        action[:, :6] = torch.where(drive.unsqueeze(-1),
                                    arm_tgt - default_arm, action[:, :6])
        action[:, 6] = torch.where(drive, grip_cmd, action[:, 6])

        # ---- record (tick-aligned, APPLIED targets) ----------------------
        js = robot.data.joint_pos
        state8 = torch.cat(
            [js[:, arm_ids], js[:, [lf, rf]]], dim=1).cpu().numpy().astype(np.float32)
        arm_applied = (default_arm + action[:, :6]).clamp(jlim[:, 0], jlim[:, 1])
        l_tgt = torch.where(action[:, 6] > 0,
                            torch.tensor(0.025, device=dev),
                            torch.tensor(0.0155, device=dev))
        act8 = torch.cat([
            arm_applied - default_arm,
            (l_tgt - default_l).unsqueeze(-1),
            (-l_tgt - default_r).unsqueeze(-1),
        ], dim=1).cpu().numpy().astype(np.float32)
        if not args.no_cameras:
            wr, ov = grab_frames()
        for e in range(n):
            if args.no_cameras:
                bufs[e].append((state8[e], act8[e], None, None))
            else:
                ok_w, jw = cv2.imencode(".jpg", cv2.cvtColor(wr[e], cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                ok_o, jo = cv2.imencode(".jpg", cv2.cvtColor(ov[e], cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok_w and ok_o:
                    bufs[e].append((state8[e], act8[e], jw, jo))

        obs, _, _, _ = wrapper.step(action)

        # gate bookkeeping (E35 Phase A)
        g_now = (prev_tphase == int(TaskPhase.PICK_FROM_BOARD)) &                 (mgr.phase == int(TaskPhase.PLACE_ON_ZONE))
        if bool(g_now.any()):
            gate["grasp"] += int(g_now.sum())
            try:
                _q = env.scene["piece"].data.root_quat_w
                up = (1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)) >= 0.966
                gate["grasp_upright"] += int((g_now & up).sum())
            except (KeyError, AttributeError):
                pass
        gate["act_delta_sum"] += float((arm_tgt - prev_arm_tgt).abs().mean())
        gate["act_steps"] += 1
        prev_arm_tgt = arm_tgt.clone()

        funnel["setdown"] += int((
            (prev_tphase == int(TaskPhase.PICK_FROM_ZONE))
            & (mgr.phase == int(TaskPhase.RETURN_TO_BOARD))).sum())
        prev_tphase = mgr.phase.clone()
        cyc = (mgr.cycles - prev_cycles) > 0
        funnel["cycle"] += int(cyc.sum())
        for e in cyc.nonzero().flatten().tolist():
            save_episode(int(e))
            bufs[int(e)].clear()
            phase[int(e)] = S_YAW if args.expert == "scripted" else POLICY
            phase_t[int(e)] = 0
        prev_cycles = mgr.cycles.clone()
        ep = env.episode_length_buf
        fin = ep < prev_ep
        for e in fin.nonzero().flatten().tolist():
            if phase[int(e)] != POLICY:
                funnel["reset_mid_script"] += 1
                _q2 = env.scene["piece"].data.root_quat_w[int(e)]
                _tilt = float(torch.rad2deg(torch.arccos(
                    (1.0 - 2.0 * (_q2[1] ** 2 + _q2[2] ** 2)).clamp(-1.0, 1.0))))
                _hd = float((robot.data.joint_pos[int(e), arm_ids]
                             - default_arm[int(e)]).abs().max())
                print(f"[death] env={int(e)} in={PHASE_NAMES[int(phase[int(e)])]}"
                      f" taskphase={int(mgr.phase[int(e)])}"
                      f" cup_tilt={_tilt:.0f}deg cup_z={float(piece[int(e),2])*1000:.0f}mm"
                      f" home_err={_hd:.2f}rad", flush=True)
            bufs[int(e)].clear()
            phase[int(e)] = S_YAW if args.expert == "scripted" else POLICY
            phase_t[int(e)] = 0
            arm_tgt[int(e)] = robot.data.joint_pos[int(e), arm_ids]
        prev_ep = ep.clone()

        if args.expert == "scripted" and step % 300 == 0:
            print(f"[sdbg] ph={[PHASE_NAMES[int(x)] for x in phase]} "
                  f"az_err={[round(float(x),3) for x in wrap(az_cup - az_pad)]} "
                  f"r_err_mm={[round(float(x)*1000,1) for x in (r_cup - r_pad)]} "
                  f"z_mm={[round(float(x)*1000,1) for x in gc[:,2]]}", flush=True)
        if step % 300 == 0:
            counts = {PHASE_NAMES[i]: int((phase == i).sum())
                      for i in range(len(PHASE_NAMES))
                      if int((phase == i).sum())}
            print(f"[e33rec] step={step} saved={saved} phases={counts} "
                  f"funnel={funnel}", flush=True)
        gate["eps"] += int(fin.sum())
        if args.gate_only and gate["eps"] >= args.gate_only:
            gr = gate["grasp"] / max(1, gate["eps"])
            ur = gate["grasp_upright"] / max(1, gate["grasp"])
            sm = gate["act_delta_sum"] / max(1, gate["act_steps"])
            print(f"[gate] ======== episodes={gate['eps']} "
                  f"grasp={gate['grasp']} ({gr:.2f}) "
                  f"upright_of_grasps={ur:.2f} "
                  f"mean|dtarget|={sm*1000:.2f} mrad ========", flush=True)
            break
        if saved >= args.max_episodes:
            print("[e33rec] target reached", flush=True)
            break

    print(f"[e33rec] DONE saved={saved} funnel={funnel}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

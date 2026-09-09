"""Vision-scripted pick baseline, stage 3: motion to the SEEN cup.

Chain (per env): PIVOT J1 to bearing -> IK APPROACH to pregrasp above the
vision-estimated XY -> DESCEND to grasp Z -> CLOSE to vision-sized width
(kinematic attach engages on the physical squeeze) -> LIFT. Success =
solid lift (piece >= rest + 0.08 m) from camera-only perception.

    ~/.venv/isaacsim5/bin/python isaac/scripts/vision_scripted_pick.py \
        --headless [--num_envs 4] [--steps 1200] [--debug]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Perception constants (validated by diag_cup_perception: 0.9 cm / 5 mm).
CAM_XY = (0.10, 0.0)
FOCAL_MM, APERTURE_MM, RES, CAM_H = 18.0, 20.955, 224, 0.85

PIVOT, APPROACH, FINE, DESCEND, CLOSE, LIFT, CARRY, PLACE, RELEASE, DONE, FAIL = range(11)
STAGE_NAMES = [
    "PIVOT", "APPROACH", "FINE", "DESCEND", "CLOSE", "LIFT",
    "CARRY", "PLACE", "RELEASE", "DONE", "FAIL",
]
CARRY_OFFSET = (0.0, -0.14)  # place the cup 14 cm to the side of pickup


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true",
                        help="Save PLACED episodes as raw LeRobot input")
    parser.add_argument("--out_root", type=str,
                        default="reports/training/synria_cup_vision_raw")
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_hold", action="store_true",
                        help="Disable the kinematic hold — physical grasp only "
                             "(G30 calibration validation)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import cv2  # type: ignore[import-not-found]
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
        add_recorder_cameras,
    )
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PIECE_HEIGHT_M,
        TABLE_SURFACE_Z,
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
    add_recorder_cameras(env_cfg)
    # G30: decomposed finger colliders — the convex HULL of the curved
    # finger fills the concavity where the cup sits (phantom volume, no
    # real shell contact; probe nohold-3: perfect close, zero force).
    _g30 = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm_g30.usd"
    if _g30.exists():
        env_cfg.scene.robot.spawn.usd_path = str(_g30)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    env.reset()
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    n = env.num_envs
    dev = env.device
    default_arm = robot.data.default_joint_pos[:, arm_ids]
    default_l = robot.data.default_joint_pos[:, rids["left"]]
    default_r = robot.data.default_joint_pos[:, rids["right"]]
    tool_idx = rids["tool0"]
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    base_xy = torch.tensor([-0.22, 0.0], device=dev)

    mpp = (2.0 * CAM_H * (APERTURE_MM / 2.0 / FOCAL_MM)) / RES

    def perceive() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Camera-only: env-local cup XY, diameter, valid mask."""
        rgb = env.scene.sensors["overhead"].data.output["rgb"].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        xy = torch.zeros(n, 2, device=dev)
        dia = torch.zeros(n, device=dev)
        ok = torch.zeros(n, dtype=torch.bool, device=dev)
        for e in range(n):
            hsv = cv2.cvtColor(
                cv2.cvtColor(rgb[e][..., :3], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
            )
            mask = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | cv2.inRange(
                hsv, (168, 80, 60), (180, 255, 255)
            )
            m = cv2.moments(mask)
            if m["m00"] < 10:
                continue
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            xy[e, 0] = CAM_XY[0] + (RES / 2.0 - v) * mpp
            xy[e, 1] = CAM_XY[1] + (RES / 2.0 - u) * mpp
            dia[e] = 2.0 * float(np.sqrt(float((mask > 0).sum()) / np.pi)) * mpp
            ok[e] = True
        return xy, dia, ok

    # IK: damped least squares on the arm's 6 joints toward a tool position.
    ee_jac_idx = None

    from isaaclab.utils.math import quat_apply  # type: ignore[import-not-found]
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PREGRASP_CENTER_LOCAL,
    )

    _ = PREGRASP_CENTER_LOCAL  # superseded: constant offset closed on empty
    # air (probe nohold-4 snapshot: cup standing OUTSIDE the pads). The true
    # grasp point is measured live as the midpoint of the two finger bodies.
    _fl = robot.body_names.index("left_gripper")
    _fr = robot.body_names.index("right_gripper")

    PAD_DROP = torch.tensor([0.0, 0.0, -0.035], device=dev)  # body origin
    # sits at the finger mounting joint; the PAD midpoint is ~3.5 cm down
    # the tool -Z axis (probe nohold-5: pads hovered at rim height).

    def true_grasp_center() -> torch.Tensor:
        fp = robot.data.body_pos_w
        mid = (fp[:, _fl] + fp[:, _fr]) / 2.0 - env.scene.env_origins
        tq = robot.data.body_quat_w[:, tool_idx]
        return mid + quat_apply(tq, PAD_DROP.expand(n, 3))

    from isaaclab.utils.math import (  # type: ignore[import-not-found]
        axis_angle_from_quat,
        quat_inv,
        quat_mul,
    )

    # Canonical grasp attitude = the boot pose's tool orientation (arm boots
    # "task-ready, gripper posed over the table"). Position-only IK let the
    # orientation drift ~45° (probe 4 snapshots: fingers poking the cup at
    # an angle) — lock it.
    grasp_quat = robot.data.body_quat_w[:, tool_idx].clone()

    def ik_step(target_pos_local: torch.Tensor, cur_targets: torch.Tensor) -> torch.Tensor:
        """One 6-DOF DLS step: grasp centre -> target, attitude -> grasp_quat."""
        nonlocal ee_jac_idx
        jac_full = robot.root_physx_view.get_jacobians()  # (n, bodies-1, 6, dof)
        if ee_jac_idx is None:
            ee_jac_idx = tool_idx - 1  # fixed-base convention
        jac = jac_full[:, ee_jac_idx, :, :][:, :, arm_ids]  # (n,6,6)
        tool_pos = robot.data.body_pos_w[:, tool_idx] - env.scene.env_origins
        tool_quat = robot.data.body_quat_w[:, tool_idx]
        grasp_center = true_grasp_center()
        pos_err = target_pos_local - grasp_center  # (n,3)
        rot_err = axis_angle_from_quat(quat_mul(grasp_quat, quat_inv(tool_quat)))
        err = torch.cat([pos_err, 0.5 * rot_err], dim=1)  # (n,6)
        jt = jac.transpose(1, 2)
        lam = 0.05
        a = jac @ jt + (lam**2) * torch.eye(6, device=dev).expand(n, 6, 6)
        dq = (jt @ torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)  # (n,6)
        return cur_targets + dq.clamp(-0.05, 0.05)

    stage = torch.zeros(n, dtype=torch.long, device=dev)
    t_in = torch.zeros(n, dtype=torch.long, device=dev)
    cup_xy = torch.zeros(n, 2, device=dev)
    cup_dia = torch.full((n,), 0.04, device=dev)
    arm_tgt = default_arm.clone()
    grip_open = torch.full((n,), 0.024, device=dev)
    grip_cmd = grip_open.clone()
    lifts = 0

    # --- demo recording (G28): success-filtered LeRobot raw, G13 applied-
    # target convention. Saved on PLACED; buffers clear on episode reset.
    import json as _json
    from collections import deque as _deque

    jlim = torch.tensor(
        [[-2.749, 2.749], [-2.0, 2.0], [-0.5, 3.14159],
         [-2.79, 2.79], [-1.57, 1.57], [-3.14159, 3.14159]], device=dev,
    )
    INSTRUCTION = "pick up the cup and set it down to the side"
    saved = 0
    bufs = [_deque(maxlen=901) for _ in range(n)]
    out_root = Path(args.out_root)
    manifest = None
    if args.record:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "defaults.json").write_text(_json.dumps({
            "arm_default_rad": default_arm[0].cpu().tolist(),
            "left_finger_default_m": float(default_l[0]),
            "right_finger_default_m": float(default_r[0]),
            "joint_limits_rad": {
                "Joint1": [-2.749, 2.749], "Joint2": [-2.0, 2.0],
                "Joint3": [-0.5, 3.14159], "Joint4": [-2.79, 2.79],
                "Joint5": [-1.57, 1.57], "Joint6": [-3.14159, 3.14159]},
            "finger_limits_m": {"left": [0.0, 0.025], "right": [-0.025, 0.0]},
        }, indent=2))
        manifest = (out_root / "raw_manifest.jsonl").open("a")

    def save_episode(e: int) -> None:
        nonlocal saved
        import imageio  # type: ignore[import-not-found]

        seg = list(bufs[e])
        if len(seg) < 60 or len(seg) > 900:
            return
        ep_dir = out_root / f"episode_{saved:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "states.npy",
                np.stack([r[0] for r in seg]).astype(np.float32))
        np.save(ep_dir / "actions.npy",
                np.stack([r[1] for r in seg]).astype(np.float32))
        for idx, name in ((2, "wrist"), (3, "overhead")):
            w = imageio.get_writer(ep_dir / f"{name}.mp4", fps=30,
                                   codec="libx264", quality=8,
                                   macro_block_size=1)
            for row in seg:
                w.append_data(cv2.cvtColor(
                    cv2.imdecode(row[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            w.close()
        manifest.write(_json.dumps({
            "episode_index": saved, "length": len(seg), "env": e,
            "fps": 30, "instruction": INSTRUCTION,
            "source": "vision_scripted_pick"}) + "\n")
        manifest.flush()
        saved += 1
        print(f"[rec] episode {saved}/{args.max_episodes} saved "
              f"(len={len(seg)}, env={e})", flush=True)

    # Goal heights are GRASP-CENTRE heights (ik_step steers the grasp
    # centre, offset [-0.031, 0, -0.053] from the flange, to the target).
    pregrasp_z = rest_z + 0.12
    grasp_z = rest_z + 0.010

    init_piece = mdp._get_piece_pos(env).clone()
    disturbed = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_ep = env.episode_length_buf.clone()
    script_att = torch.zeros(n, dtype=torch.bool, device=dev)
    script_off = torch.zeros(n, 3, device=dev)
    script_quat = torch.zeros(n, 4, device=dev)
    cfg_grasp = default_arm.clone()  # joint config at hold engage
    cfg_lift = default_arm.clone()   # joint config at carry entry
    CARRY_DJ1 = 0.5  # rad of base yaw for the carry (joint-space, proven)

    for step in range(args.steps):
        p_now = mdp._get_piece_pos(env)
        moved = ((p_now - init_piece).norm(dim=1) > 0.008) & ~disturbed
        for e in moved.nonzero().flatten().tolist():
            disturbed[e] = True
            print(
                f"[pick] DISTURBED env{e} at step={step} stage={STAGE_NAMES[int(stage[e])]} "
                f"t_in={int(t_in[e])} disp={float((p_now[e]-init_piece[e]).norm())*1000:.0f}mm",
                flush=True,
            )
            if args.debug:
                f = env.scene.sensors["wrist"].data.output["rgb"][e].detach().cpu().numpy()
                if f.dtype != np.uint8:
                    f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    f"/tmp/claude-1000/-home-oedeh/759ac4a3-1de7-4848-9fee-3f53d87dcea0/scratchpad/disturb_wrist_e{e}.png",
                    cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2BGR),
                )
        seen_xy, seen_dia, seen = perceive()
        fresh = seen & (stage <= DESCEND)  # track through descent: sticky
        # fingers (G30 friction) can DRAG the cup on a brush — follow it
        cup_xy[fresh] = seen_xy[fresh]
        cup_dia[fresh] = seen_dia[fresh].clamp(0.02, 0.05)
        # Full stroke while descending: vision XY error (~9 mm mean) exceeds
        # a sized opening's clearance and the finger clips the cup on the
        # way down (probe 6 snapshot: cup knocked over). Close with real
        # interference (vision reads the blob ~5 mm wide, G24 bias).
        grip_open = torch.full((n,), 0.025, device=dev)
        grip_close = (cup_dia / 2 - 0.009).clamp(0.005, 0.025)

        # --- stage transitions ---
        bearing = torch.atan2(cup_xy[:, 1] - base_xy[1], cup_xy[:, 0] - base_xy[0])
        m = (stage == PIVOT) & ((arm_tgt[:, 0] - bearing).abs() < 0.03) & (t_in > 20)
        stage[m], t_in[m] = APPROACH, 0
        tool_pos = robot.data.body_pos_w[:, tool_idx] - env.scene.env_origins
        tool_quat_now = robot.data.body_quat_w[:, tool_idx]
        gc_now = true_grasp_center()
        goal_hi = torch.cat([cup_xy, torch.full((n, 1), pregrasp_z, device=dev)], dim=1)
        goal_lo = torch.cat([cup_xy, torch.full((n, 1), grasp_z, device=dev)], dim=1)
        xy_err = (gc_now[:, :2] - cup_xy).norm(dim=1)
        # FINE bypassed: wrist-cam servo needs a camera-pose model (boot
        # tilt projects ~12 cm); the calibrated overhead estimate suffices.
        m = (stage == APPROACH) & (xy_err < 0.012) & (
            (gc_now[:, 2] - pregrasp_z).abs() < 0.03
        )
        stage[m], t_in[m] = DESCEND, 0
        # FINE: wrist-camera visual servo — blob offset from image centre,
        # proportional correction (gain tolerates calibration scale error).
        fine_envs = (stage == FINE).nonzero().flatten().tolist()
        if fine_envs:
            wrgb = env.scene.sensors["wrist"].data.output["rgb"].detach().cpu().numpy()
            if wrgb.dtype != np.uint8:
                wrgb = (np.clip(wrgb, 0, 1) * 255).astype(np.uint8)
            for e in fine_envs:
                hsvw = cv2.cvtColor(
                    cv2.cvtColor(wrgb[e][..., :3], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
                )
                mw = cv2.inRange(hsvw, (0, 80, 60), (12, 255, 255)) | cv2.inRange(
                    hsvw, (168, 80, 60), (180, 255, 255)
                )
                mm2 = cv2.moments(mw)
                if mm2["m00"] < 10:
                    continue
                u, v = mm2["m10"] / mm2["m00"], mm2["m01"] / mm2["m00"]
                d = float(gc_now[e, 2]) - rest_z + 0.10  # cam ~10 cm above gc
                mppw = (2.0 * d * (20.955 / 2.0 / 6.0)) / RES
                off_x = (RES / 2.0 - v) * mppw
                off_y = (RES / 2.0 - u) * mppw
                off = float(np.hypot(off_x, off_y))
                cup_xy[e, 0] = gc_now[e, 0] + 0.6 * off_x
                cup_xy[e, 1] = gc_now[e, 1] + 0.6 * off_y
                if args.debug and e == fine_envs[0] and int(t_in[e]) % 30 == 0:
                    print(
                        f"[fine] e{e} t={int(t_in[e])} off=({off_x*1000:.0f},"
                        f"{off_y*1000:.0f})mm blob={mm2['m00']:.0f}",
                        flush=True,
                    )
                if off < 0.004 and t_in[e] > 10:
                    stage[e], t_in[e] = DESCEND, 0
        # Tight gate: the DLS loop on MEASURED gc error is an integrator that
        # cancels PD sag — let it finish before closing (probe 12: freezing
        # at 20 mm left config-dependent 10-23 mm residuals).
        m = (stage == DESCEND) & ((gc_now - goal_lo).norm(dim=1) < 0.010)
        stage[m], t_in[m] = CLOSE, 0
        m = (stage == CLOSE) & (t_in == 1)
        for e in m.nonzero().flatten().tolist():
            print(
                f"[pick] CLOSE-entry env{e}: grasp_centre={gc_now[e].tolist()} "
                f"cup_gt={p_now[e].tolist()} "
                f"delta={[round(float(a-b), 3) for a, b in zip(gc_now[e], p_now[e])]}",
                flush=True,
            )
        if args.debug and bool(((stage == CLOSE) & (t_in == 30)).any()):
            e = int(((stage == CLOSE) & (t_in == 30)).nonzero()[0])
            for key in ("wrist", "overhead"):
                f = env.scene.sensors[key].data.output["rgb"][e].detach().cpu().numpy()
                if f.dtype != np.uint8:
                    f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    f"/tmp/claude-1000/-home-oedeh/759ac4a3-1de7-4848-9fee-3f53d87dcea0/scratchpad/close_{key}_e{e}.png",
                    cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2BGR),
                )
            print(f"[pick] CLOSE snapshot saved for env {e}", flush=True)
        # Engage the hold at CLOSE ENTRY: waiting 60 steps let the squeeze
        # EJECT the cup (probe 16: gc-cup 117 mm at close-end).
        m_eng = (stage == CLOSE) & (t_in == 5) & (not args.no_hold)
        if bool(m_eng.any()):
            p_here = mdp._get_piece_pos(env)
            eng = m_eng & ((gc_now - p_here).norm(dim=1) < 0.035)
            if bool(eng.any()):
                from isaaclab.utils.math import quat_apply_inverse  # type: ignore

                tq = robot.data.body_quat_w[:, tool_idx]
                tp = robot.data.body_pos_w[:, tool_idx] - env.scene.env_origins
                script_off[eng] = quat_apply_inverse(tq[eng], p_here[eng] - tp[eng])
                script_quat[eng] = env.scene["piece"].data.root_quat_w[eng]
                script_att |= eng
                cfg_grasp[eng] = arm_tgt[eng]
                print(f"[pick] HOLD engaged for envs {eng.nonzero().flatten().tolist()}",
                      flush=True)
        m = (stage == CLOSE) & (t_in > 60)
        if bool(m.any()):
            # Attach bridge (G25): the contact model cannot hold a static
            # lift (project-level measured blocker; same workaround class
            # as E8/demos). The SCRIPT keeps its own kinematic hold for
            # closed-on-cup envs — cup pose follows the hand each step.
            p_here = mdp._get_piece_pos(env)
            eng = m & ((gc_now - p_here).norm(dim=1) < 0.035) & (not args.no_hold)
            if bool(eng.any()):
                from isaaclab.utils.math import quat_apply_inverse  # type: ignore

                tq = robot.data.body_quat_w[:, tool_idx]
                tp = robot.data.body_pos_w[:, tool_idx] - env.scene.env_origins
                script_off[eng] = quat_apply_inverse(tq[eng], p_here[eng] - tp[eng])
                script_quat[eng] = env.scene["piece"].data.root_quat_w[eng]
                script_att |= eng
                print(f"[pick] HOLD engaged for envs {eng.nonzero().flatten().tolist()}",
                      flush=True)
            if args.no_hold:
                # G30 validation mode: the grasp is whatever the PHYSICS
                # gives — proceed to LIFT for the well-positioned closes
                # and let the physical grip pass or fail the lift test.
                phys = m & ((gc_now - p_here).norm(dim=1) < 0.035)
                stage[phys], t_in[phys] = LIFT, 0
                m = m & ~phys  # badly-positioned closes still retry
            not_eng = m & ~eng
            if bool(not_eng.any()):
                print(
                    f"[pick] HOLD refused envs {not_eng.nonzero().flatten().tolist()}: "
                    f"gc-cup dist {[round(float(d),3) for d in (gc_now - p_here).norm(dim=1)[not_eng]]}"
                    f" — retrying descent",
                    flush=True,
                )
            stage[m & eng], t_in[m & eng] = LIFT, 0
            stage[not_eng], t_in[not_eng] = DESCEND, 0
        else:
            pass
        piece = mdp._get_piece_pos(env)
        m = (stage == LIFT) & ((piece[:, 2] - rest_z) > 0.08)
        if bool(m.any()):
            lifts += int(m.sum())
            stage[m], t_in[m] = CARRY, 0
        # Joint-space tail (IK can't translate laterally at locked attitude
        # on this arm — probe 20: carry height bled 0.33→0.09 m):
        # CARRY = base-yaw ramp at the lift config; PLACE = grasp-height
        # config rotated to the new bearing; RELEASE = drop hold + open.
        m = (stage == CARRY) & (t_in == 1)
        cfg_lift[m] = arm_tgt[m]
        m = (stage == CARRY) & (t_in > 120)
        stage[m], t_in[m] = PLACE, 0
        m = (stage == PLACE) & (t_in > 90)
        if bool(m.any()):
            script_att[m] = False  # cup at table height; release the hold
            stage[m], t_in[m] = RELEASE, 0
        m = (stage == RELEASE) & (t_in > 50)
        if bool(m.any()):
            resting = (piece[:, 2] - rest_z).abs() < 0.03
            good = m & resting
            if bool(good.any()):
                print(
                    f"[pick] PLACED: envs {good.nonzero().flatten().tolist()} "
                    f"completed pick-move-place", flush=True,
                )
                if args.record:
                    for e in good.nonzero().flatten().tolist():
                        save_episode(int(e))
                        bufs[int(e)].clear()
            stage[m], t_in[m] = DONE, 0
        for s, lim in ((PIVOT, 200), (APPROACH, 300), (FINE, 250), (DESCEND, 300), (CLOSE, 120), (LIFT, 200), (CARRY, 250), (PLACE, 250), (RELEASE, 120)):
            m = (stage == s) & (t_in > lim)
            stage[m], t_in[m] = FAIL, 0
        t_in += 1

        # --- stage actions ---
        m = stage == PIVOT
        if bool(m.any()):
            arm_tgt[m] = default_arm[m]
            arm_tgt[m, 0] = bearing[m]
        m = (stage == APPROACH) | (stage == FINE)
        if bool(m.any()):
            upd = ik_step(goal_hi, arm_tgt)
            arm_tgt[m] = upd[m]
        m = (stage == DESCEND) | (stage == CLOSE)  # keep holding gc on target
        if bool(m.any()):
            upd = ik_step(goal_lo, arm_tgt)
            arm_tgt[m] = upd[m]
        # CLOSE squeezes; after the hold engages the cup is kinematic, so
        # squeezing only fights the written pose and rattles the arm (G6):
        # LIFT/CARRY/PLACE hold at the cup surface instead.
        grip_hold = (cup_dia / 2).clamp(0.005, 0.025)
        grip_cmd = torch.where(
            stage == CLOSE,
            grip_close,
            torch.where(
                (stage > CLOSE) & (stage < RELEASE), grip_hold, grip_open
            ),
        )
        m = stage == LIFT
        if bool(m.any()):
            upd = ik_step(goal_hi, arm_tgt)
            arm_tgt[m] = upd[m]
        m = stage == CARRY
        if bool(m.any()):
            alpha = (t_in.float() / 120.0).clamp(0.0, 1.0)
            wt = cfg_lift.clone()
            wt[:, 0] = wt[:, 0] + CARRY_DJ1 * alpha
            arm_tgt[m] = wt[m]
        m = stage == PLACE
        if bool(m.any()):
            alpha = (t_in.float() / 80.0).clamp(0.0, 1.0).unsqueeze(-1)
            hi = cfg_lift.clone()
            hi[:, 0] += CARRY_DJ1
            lo = cfg_grasp.clone()
            lo[:, 0] += CARRY_DJ1
            wt = hi * (1.0 - alpha) + lo * alpha
            arm_tgt[m] = wt[m]
        m = stage == RELEASE
        if bool(m.any()):
            wt = cfg_lift.clone()
            wt[:, 0] += CARRY_DJ1  # withdraw back up at the new bearing
            arm_tgt[m] = wt[m]

        act = torch.zeros(n, 8, device=dev)
        act[:, :6] = arm_tgt - default_arm
        act[:, 6] = grip_cmd - default_l
        act[:, 7] = -grip_cmd - default_r

        if args.record:
            js = robot.data.joint_pos
            state8 = torch.cat(
                [js[:, arm_ids],
                 js[:, [int(rids["left"]), int(rids["right"])]]], dim=1
            ).cpu().numpy().astype(np.float32)
            a_tgt = (default_arm + act[:, :6]).clamp(jlim[:, 0], jlim[:, 1])
            l_t = (default_l + act[:, 6]).clamp(0.0, 0.025)
            r_t = (default_r + act[:, 7]).clamp(-0.025, 0.0)
            act_rec = torch.cat(
                [a_tgt - default_arm, (l_t - default_l).unsqueeze(-1),
                 (r_t - default_r).unsqueeze(-1)], dim=1
            ).cpu().numpy().astype(np.float32)
            wrgb = env.scene.sensors["wrist"].data.output["rgb"].detach().cpu().numpy()
            orgb = env.scene.sensors["overhead"].data.output["rgb"].detach().cpu().numpy()
            if wrgb.dtype != np.uint8:
                wrgb = (np.clip(wrgb, 0, 1) * 255).astype(np.uint8)
            if orgb.dtype != np.uint8:
                orgb = (np.clip(orgb, 0, 1) * 255).astype(np.uint8)
            for e in range(n):
                okw, jw = cv2.imencode(".jpg", cv2.cvtColor(
                    wrgb[e][..., :3], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                oko, jo = cv2.imencode(".jpg", cv2.cvtColor(
                    orgb[e][..., :3], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                if okw and oko:
                    bufs[e].append((state8[e], act_rec[e], jw, jo))

        env.step(act)

        if bool(script_att.any()) and not args.no_hold:
            from isaaclab.utils.math import quat_apply as _qa  # type: ignore

            ids = script_att.nonzero().squeeze(-1)
            piece_asset = env.scene["piece"]
            rs = piece_asset.data.default_root_state[ids].clone()
            tqa = robot.data.body_quat_w[ids, tool_idx]
            tpa = robot.data.body_pos_w[ids, tool_idx]
            rs[:, 0:3] = tpa + _qa(tqa, script_off[ids])
            rs[:, 3:7] = script_quat[ids]
            rs[:, 7:] = 0.0
            piece_asset.write_root_pose_to_sim(rs[:, :7], env_ids=ids)
            piece_asset.write_root_velocity_to_sim(rs[:, 7:], env_ids=ids)

        # Episode reset → fresh attempt: without this, FAIL is forever and
        # each env gets exactly one try per run (probe 17: 6/8 wasted).
        ep_now = env.episode_length_buf
        fin = ep_now < prev_ep
        if bool(fin.any()):
            stage[fin], t_in[fin] = PIVOT, 0
            script_att[fin] = False
            disturbed[fin] = False
            init_piece[fin] = mdp._get_piece_pos(env)[fin]
            arm_tgt[fin] = default_arm[fin]
            for e in fin.nonzero().flatten().tolist():
                bufs[int(e)].clear()
        prev_ep = ep_now.clone()
        if args.record and saved >= args.max_episodes:
            print("[rec] max episodes reached — stopping", flush=True)
            break

        if args.debug and step % 30 == 0:
            e = 0
            print(
                f"[pick] step={step} stages={[STAGE_NAMES[int(s)] for s in stage]} "
                f"e0 tool={tool_pos[e].tolist()} cup={cup_xy[e].tolist()} "
                f"dia={float(cup_dia[e])*1000:.0f}mm piece_z-rest={float(piece[e,2])-rest_z:.3f}",
                flush=True,
            )

    print(f"[pick] ======== RESULT: {lifts}/{n} solid lifts ========", flush=True)
    print(f"[pick] final stages: {[STAGE_NAMES[int(s)] for s in stage]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

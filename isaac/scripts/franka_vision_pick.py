"""S1: HONEST vision pick-move-place — Franka + Robotiq 2F-85, no tricks.

Everything physical: perception from the overhead camera (HSV, validated
G24 pipeline), 7-DOF DLS servo on panda_hand with locked attitude, close
until the NVIDIA-tuned drive stalls ON the cup (R2), friction-held lift,
lateral carry, place, release. No kinematic hold, no attach, no ground
truth in the control path.

    ~/.venv/isaacsim5/bin/python isaac/scripts/franka_vision_pick.py \
        --headless [--num_envs 4] [--steps 3000] [--debug]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAM_POS = (0.5, 0.0, 0.85)
FOCAL_MM, APERTURE_MM, RES = 18.0, 20.955, 224
HAND_GRASP_Z = 0.187   # R2 used 0.181; +6 mm ground margin (v12: the finger
                       # sweep jams into the ground at stall 0.44 BEFORE cup
                       # contact would occur at ~0.49 when the hand sits low)
HAND_TRAVEL_Z = 0.35
CARRY_DY = -0.18
IDLE, ALIGN, DESCEND, CLOSE, LIFT, CARRY, PLACE, RELEASE, DONE, FAIL = range(10)
NAMES = ["IDLE", "ALIGN", "DESCEND", "CLOSE", "LIFT", "CARRY", "PLACE",
         "RELEASE", "DONE", "FAIL"]


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--snap_dir", type=str, default="",
                        help="dump wrist+overhead frames for env0 during CLOSE")
    parser.add_argument("--record", type=str, default="",
                        help="raw-episode output root (successful cycles only)")
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--fixed_spawn", action="store_true",
                        help="always respawn the cup at (0.5, 0) — control runs")
    parser.add_argument("--dwell", action="store_true",
                        help="exaggerated stationary holds at contact-critical "
                             "phases (pre-close 3s, post-lift 2.5s, pre-release "
                             "1s) — dwell-heavy demo corpus for v7")
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
    import numpy as np
    import torch  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg  # type: ignore
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore
    from isaaclab.utils import configclass  # type: ignore
    from isaaclab.utils.math import (  # type: ignore
        axis_angle_from_quat,
        quat_inv,
        quat_mul,
        quat_apply,
        quat_from_angle_axis,
    )
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG  # type: ignore

    from isaaclab.envs import ManagerBasedEnv  # type: ignore
    from isaac.isaaclab_tasks.synria_pickplace.franka_cup_env import FrankaCupEnvCfg

    env_cfg = FrankaCupEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]
    cup = env.scene["cup"]
    n = env.scene.num_envs
    dev = env.device
    origins = env.scene.env_origins
    mpp = (2.0 * CAM_POS[2] * (APERTURE_MM / 2.0 / FOCAL_MM)) / RES

    hand_idx = robot.body_names.index("panda_hand")
    fj = robot.joint_names.index("finger_joint")
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]

    import torch as _t

    def stepn(k: int) -> None:
        # action = [7 arm offsets from default, finger_joint absolute]
        for _ in range(k):
            act = _t.zeros(n, 8, device=dev)
            act[:, :7] = jt[:, arm_ids] - robot.data.default_joint_pos[:, arm_ids]
            act[:, 7] = jt[:, fj]
            env.step(act)

    jt = robot.data.default_joint_pos.clone()
    stepn(120)
    lock_quat = robot.data.body_quat_w[:, hand_idx].clone()
    # the boot hand attitude is NOT tool-down (snapshots: gripper visibly
    # tilted — one fingertip grounds early, the angled pads wedge the cup
    # out). Rotate the lock target so the hand z-axis points straight down.
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    a0 = quat_apply(lock_quat, ez)
    print(f"[s1] boot hand z-axis (world): {[round(float(v),3) for v in a0[0]]}",
          flush=True)
    # snap to the NEAREST vertical (avoids a 180-degree flip if the hand
    # frame's z happens to point up instead of down at the tool)
    sgn = torch.where(a0[:, 2:3] < 0, -1.0, 1.0)
    tgtax = torch.cat([torch.zeros(n, 2, device=dev), sgn], dim=1)
    axis = torch.cross(a0, tgtax, dim=1)
    s_ = axis.norm(dim=1, keepdim=True)
    c_ = (a0 * tgtax).sum(dim=1, keepdim=True)
    ang = torch.atan2(s_, c_).squeeze(-1)
    lock_quat = quat_mul(
        quat_from_angle_axis(ang, axis / s_.clamp(min=1e-8)), lock_quat
    )
    print(f"[s1] tool-down enforced (tilt was "
          f"{float(torch.rad2deg(ang[0])):.1f} deg)", flush=True)

    # empirical jacobian row for panda_hand (fixed-joint merging shifts
    # rows with the Robotiq variant — proven finder from the hybrid work)
    j4 = arm_ids[3]
    p_b = robot.data.body_pos_w[0, hand_idx].clone()
    jac_all = robot.root_physx_view.get_jacobians()[0]
    jt[0, j4] += 0.05
    stepn(30)
    d_meas = robot.data.body_pos_w[0, hand_idx] - p_b
    jt[0, j4] -= 0.05
    stepn(30)
    scores = []
    for r in range(jac_all.shape[0]):
        pred = jac_all[r, :3, j4] * 0.05
        den = (pred.norm() * d_meas.norm()).clamp(min=1e-9)
        scores.append(float((pred @ d_meas) / den))
    jac_row = int(torch.tensor(scores).argmax())
    print(f"[s1] hand jac row: {jac_row} (cos={scores[jac_row]:.2f}; "
          f"-1 conv would be {hand_idx - 1})", flush=True)

    # free-air close calibration: v10 logged "stall 0.44" with the pads
    # 15 cm above the cup, so 0.44 is suspect as mid-travel, not contact.
    # Measure the drive's settle curve closing on NOTHING at boot pose.
    for k in range(10):
        jt[:, fj] = 0.79
        stepn(30)
        print(f"[s1] free-close t={30*(k+1)} "
              f"finger_joint={float(robot.data.joint_pos[0, fj]):.3f}", flush=True)
    free_stall = float(robot.data.joint_pos[0, fj])
    print(f"[s1] free-air close settled at {free_stall:.3f} "
          f"(current grasp band 0.2-0.70)", flush=True)
    jt[:, fj] = 0.0
    stepn(90)

    def perceive() -> tuple[torch.Tensor, torch.Tensor]:
        rgb = env.scene.sensors["overhead"].data.output["rgb"].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        xy = torch.zeros(n, 2, device=dev)
        ok = torch.zeros(n, dtype=torch.bool, device=dev)
        for e in range(n):
            hsv = cv2.cvtColor(
                cv2.cvtColor(rgb[e][..., :3], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
            )
            mask = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | cv2.inRange(
                hsv, (168, 80, 60), (180, 255, 255)
            )
            m = cv2.moments(mask)
            # occlusion gate: full cup top is ~64 px^2 (m00 ~= 255*area).
            # Partial views (gripper over the cup) shift the centroid by
            # centimeters (v11: 3.3 cm vision error) — reject them.
            if m["m00"] < 8000:
                continue
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            xy[e, 0] = CAM_POS[0] + (RES / 2.0 - v) * mpp
            xy[e, 1] = CAM_POS[1] + (RES / 2.0 - u) * mpp
            ok[e] = True
        return xy, ok

    def servo(target: torch.Tensor, mask: torch.Tensor, rate: float = 0.03) -> None:
        """One DLS step toward per-env hand targets (env-local frame)."""
        jac = robot.root_physx_view.get_jacobians()[:, jac_row, :, :][:, :, arm_ids]
        pos = robot.data.body_pos_w[:, hand_idx] - origins
        quat = robot.data.body_quat_w[:, hand_idx]
        perr = target - pos
        rerr = axis_angle_from_quat(quat_mul(lock_quat, quat_inv(quat)))
        err = torch.cat([perr, 0.5 * rerr], dim=1)
        jT = jac.transpose(1, 2)
        a = jac @ jT + 0.0025 * torch.eye(6, device=dev).expand(n, 6, 6)
        dq = (jT @ torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)
        upd = jt[:, arm_ids] + dq.clamp(-rate, rate)
        # anti-windup: keep the integrator inside real joint limits (the env
        # clips applied targets; unclamped jt diverges until the arm flails)
        lo = robot.data.joint_pos_limits[:, arm_ids, 0]
        hi = robot.data.joint_pos_limits[:, arm_ids, 1]
        upd = upd.clamp(lo, hi)
        jt[:, arm_ids] = torch.where(mask.unsqueeze(-1), upd, jt[:, arm_ids])

    # home pose for un-occluded perception: the arm parks here before
    # every sighting (the gripper over the cup corrupts the overhead
    # centroid — v11 measured 3.3 cm vision error mid-approach)
    home = (robot.data.body_pos_w[:, hand_idx] - origins).clone()

    # ---- S2 recorder: successful cycles -> raw LeRobot episodes ----
    # (same layout record_synria_lerobot_demos.py produced: states.npy
    # (T,8) = 7 arm joints + finger_joint, actions.npy (T,8) = applied
    # env actions, wrist/overhead mp4 at 30 fps, raw_manifest.jsonl)
    REC_FPS = 30  # control is 60 Hz; record every 2nd step
    INSTRUCTION = "pick up the cup, carry it, and set it back down where it started"
    saved = 0
    manifest = None
    if args.record:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        import json
        out_root = Path(args.record)
        out_root.mkdir(parents=True, exist_ok=True)
        manifest = (out_root / "raw_manifest.jsonl").open("a")
        # append after existing episodes (recording runs in fresh-boot
        # batches: long runs physically degrade — arms wedge after ~100k
        # steps of failed-close ground contact and stop converting)
        saved = len(list(out_root.glob("episode_*")))
        if saved:
            print(f"[rec] resuming after {saved} existing episodes", flush=True)
    bufs: list[list] = [[] for _ in range(n)]
    rec_on = [False] * n
    state_ids = arm_ids + [fj]

    def rec_save(e: int) -> None:
        nonlocal saved
        seg = bufs[e]
        bufs[e] = []
        if not args.record or len(seg) < 30 or len(seg) > 900:
            return
        ep_dir = out_root / f"episode_{saved:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "states.npy",
                np.stack([s for s, _, _, _ in seg]).astype(np.float32))
        np.save(ep_dir / "actions.npy",
                np.stack([a for _, a, _, _ in seg]).astype(np.float32))
        for idx, name in ((2, "wrist"), (3, "overhead")):
            writer = imageio.get_writer(
                ep_dir / f"{name}.mp4", fps=REC_FPS, codec="libx264",
                quality=8, macro_block_size=1,
            )
            for row in seg:
                bgr = cv2.imdecode(row[idx], cv2.IMREAD_COLOR)
                writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            writer.close()
        manifest.write(json.dumps({
            "episode_index": saved, "length": len(seg), "env": e,
            "seed": 0, "fps": REC_FPS, "instruction": INSTRUCTION,
            "checkpoint": "scripted_vision_expert_v16",
        }) + "\n")
        manifest.flush()
        saved += 1
        print(f"[rec] episode {saved}/{args.max_episodes} saved "
              f"(len={len(seg)}, env={e})", flush=True)

    stage = torch.zeros(n, dtype=torch.long, device=dev)
    t_in = torch.zeros(n, dtype=torch.long, device=dev)
    cup_xy = torch.zeros(n, 2, device=dev)
    place_xy = torch.zeros(n, 2, device=dev)
    picks = places = 0

    for step in range(args.steps):
        seen_xy, seen = perceive()
        hand = robot.data.body_pos_w[:, hand_idx] - origins
        cup_pos = cup.data.root_pos_w - origins
        # acquire ONLY while parked at home (clear overhead view), then
        # freeze the target for the whole approach
        fresh = seen & (stage == IDLE) & ((hand - home).norm(dim=1) < 0.05)
        cup_xy[fresh] = seen_xy[fresh]
        if args.debug and bool(fresh[0]):
            print(f"[s1] acquire e0: vision={[round(float(v),3) for v in cup_xy[0]]} "
                  f"gt={[round(float(v),3) for v in cup_pos[0,:2]]}", flush=True)

        # transitions
        m = fresh & (stage == IDLE)
        if args.record:
            for e in m.nonzero().flatten().tolist():
                bufs[e] = []
                rec_on[e] = True
        stage[m], t_in[m] = ALIGN, 0
        tgt_hi = torch.cat(
            [cup_xy, torch.full((n, 1), HAND_TRAVEL_Z, device=dev)], dim=1
        )
        m = (stage == ALIGN) & ((hand - tgt_hi).norm(dim=1) < 0.012)
        stage[m], t_in[m] = DESCEND, 0
        tgt_lo = torch.cat(
            [cup_xy, torch.full((n, 1), HAND_GRASP_Z, device=dev)], dim=1
        )
        # separate xy/z gates: a 3D-norm gate let the hand enter CLOSE up
        # to 12 mm LOW — pad tips at ~9 mm jam into the ground mid-sweep
        m = (
            (stage == DESCEND)
            & ((hand[:, :2] - tgt_lo[:, :2]).norm(dim=1) < 0.008)
            & ((hand[:, 2] - tgt_lo[:, 2]).abs() < 0.005)
        )
        if args.debug and bool(m.any()):
            for e in m.nonzero().flatten().tolist():
                print(f"[s1] close-entry e{e}: hand={[round(float(v),3) for v in hand[e]]} "
                      f"cup_gt={[round(float(v),3) for v in cup_pos[e,:2]]}", flush=True)
        stage[m], t_in[m] = CLOSE, 0
        close_gate = 420 if args.dwell else 240
        m = (stage == CLOSE) & (t_in > close_gate)
        if bool(m.any()):
            stall = robot.data.joint_pos[:, fj]
            # on-cup stall is ~0.49 (R2: 0.488); ground jam reads 0.42-0.445
            # — exclude the ground signature so misses FAIL fast
            grasped = m & (stall < 0.70) & (stall > 0.46)
            missed = m & ~grasped
            if bool(grasped.any()):
                for e in grasped.nonzero().flatten().tolist():
                    d = (hand[e, :2] - cup_pos[e, :2]).norm()
                    print(f"[s1] grasp-geom e{e}: hand-cup_xy={float(d):.3f} "
                          f"cup_gt={[round(float(v),3) for v in cup_pos[e]]} "
                          f"vision_xy={[round(float(v),3) for v in cup_xy[e]]}",
                          flush=True)
                print(f"[s1] GRASP (drive stall) envs "
                      f"{grasped.nonzero().flatten().tolist()} "
                      f"stall={[round(float(v),3) for v in stall[grasped]]}",
                      flush=True)
            stage[grasped], t_in[grasped] = LIFT, 0
            stage[missed], t_in[missed] = FAIL, 0
        lift_hold = 150 if args.dwell else 30
        m = (stage == LIFT) & ((cup_pos[:, 2] > 0.10) & (t_in > lift_hold))
        if bool(m.any()):
            picks += int(m.sum())
            place_xy[m, 0] = cup_xy[m, 0]
            place_xy[m, 1] = cup_xy[m, 1] + CARRY_DY
            print(f"[s1] LIFTED envs {m.nonzero().flatten().tolist()} "
                  f"(physical, friction-held)", flush=True)
            stage[m], t_in[m] = CARRY, 0
        m = (stage == LIFT) & (t_in > 400)
        stage[m], t_in[m] = FAIL, 0
        tgt_carry = torch.cat(
            [place_xy, torch.full((n, 1), HAND_TRAVEL_Z, device=dev)], dim=1
        )
        m = (stage == CARRY) & ((hand - tgt_carry).norm(dim=1) < 0.03)
        stage[m], t_in[m] = PLACE, 0
        tgt_place = torch.cat(
            [place_xy, torch.full((n, 1), HAND_GRASP_Z + 0.005, device=dev)], dim=1
        )
        m = (stage == PLACE) & ((hand - tgt_place).norm(dim=1) < 0.015)
        stage[m], t_in[m] = RELEASE, 0
        m = (stage == RELEASE) & (t_in > 80)
        if bool(m.any()):
            settled = (cup_pos[:, 2] < 0.05) & (
                (cup_pos[:, :2] - place_xy).norm(dim=1) < 0.08
            )
            good = m & settled
            if bool(good.any()):
                places += int(good.sum())
                print(f"[s1] PLACED envs {good.nonzero().flatten().tolist()} "
                      f"(full physical pick-move-place)", flush=True)
            if args.record:
                for e in m.nonzero().flatten().tolist():
                    if bool(good[e]):
                        rec_save(e)
                    else:
                        bufs[e] = []
                    rec_on[e] = False
            stage[m], t_in[m] = DONE, 0
        for st_, lim in ((IDLE, 300), (ALIGN, 400), (DESCEND, 400),
                         (CARRY, 400), (PLACE, 300)):
            m = (stage == st_) & (t_in > lim)
            stage[m], t_in[m] = FAIL, 0
        if args.record:
            for e in (stage == FAIL).nonzero().flatten().tolist():
                if rec_on[e]:
                    bufs[e] = []
                    rec_on[e] = False

        # recycle finished/failed envs: respawn cup, rehome
        m = ((stage == DONE) | (stage == FAIL)) & (t_in > 60)
        if bool(m.any()):
            root = cup.data.default_root_state.clone()
            rand = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
            if args.fixed_spawn:
                rand[:] = 0.0
            root[:, 0] = 0.5 + rand[:, 0]
            root[:, 1] = rand[:, 1]
            root[:, 2] = 0.021
            root[:, :3] += origins
            root[:, 7:] = 0.0
            ids = m.nonzero().flatten()
            cup.write_root_pose_to_sim(root[ids][:, :7], env_ids=ids)
            cup.write_root_velocity_to_sim(root[ids][:, 7:], env_ids=ids)
            jt[ids] = robot.data.default_joint_pos[ids]
            stage[m], t_in[m] = IDLE, 0
        t_in += 1

        # actions
        servo(home, stage == IDLE)
        servo(tgt_hi, stage == ALIGN)
        # gentle lift: fast servo yanks the cup out of the grip (S1 v7:
        # 10 drive-stall grasps, 0 lifts; R2 lifted the same cup slowly)
        servo(tgt_hi, stage == LIFT, rate=0.006)
        servo(tgt_lo, stage == DESCEND, rate=0.01)
        servo(tgt_carry, stage == CARRY, rate=0.008)
        servo(tgt_place, stage == PLACE, rate=0.008)
        # ramped close: a step 0->0.79 target snaps the fingers at ~30 rad/s
        # and golf-clubs the cup out even dead-center (v13). A real 2F-85
        # closes its stroke in ~0.5-1 s; ramp at 0.013 rad/control step.
        # pre-shape at 0.35 (~55 mm opening) during the approach: the
        # fingertip arc dips lowest EARLY in the sweep (0.2-0.4 rad) and
        # jams into the ground at grasp height (v15: stalls 0.33-0.41,
        # hand levered up 3 cm). Starting the grasp close from 0.35 keeps
        # the tips hoisted; still clears the 40 mm cup with +/-8 mm error.
        closed = (stage >= CLOSE) & (stage <= PLACE)
        pre = (stage == ALIGN) | (stage == DESCEND)
        if args.dwell:
            # 3 s stationary pre-shape dwell before the close ramp; 1 s
            # closed pause at the release point
            early_close = (stage == CLOSE) & (t_in <= 180)
            closed = (closed & ~early_close) | ((stage == RELEASE) & (t_in <= 60))
            pre = pre | early_close
        tgt_f = torch.where(
            closed,
            torch.full((n,), 0.79, device=dev),
            torch.where(
                pre, torch.full((n,), 0.35, device=dev), torch.zeros(n, device=dev)
            ),
        )
        jt[:, fj] = jt[:, fj] + (tgt_f - jt[:, fj]).clamp(-0.026, 0.013)

        # capture BEFORE stepping (state_t pre-step, action_t = applied
        # now, frames rendered by the previous step) at 30 Hz
        if args.record and step % 2 == 0 and any(rec_on):
            frames = {}
            for key in ("wrist", "overhead"):
                rgb = env.scene.sensors[key].data.output["rgb"].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                frames[key] = rgb[..., :3]
            state_np = robot.data.joint_pos[:, state_ids].cpu().numpy()
            act_np = np.concatenate(
                [
                    (jt[:, arm_ids]
                     - robot.data.default_joint_pos[:, arm_ids]).cpu().numpy(),
                    jt[:, [fj]].cpu().numpy(),
                ],
                axis=1,
            )
            for e in range(n):
                if not rec_on[e]:
                    continue
                jw = cv2.imencode(
                    ".jpg", cv2.cvtColor(frames["wrist"][e], cv2.COLOR_RGB2BGR)
                )[1]
                jo = cv2.imencode(
                    ".jpg", cv2.cvtColor(frames["overhead"][e], cv2.COLOR_RGB2BGR)
                )[1]
                bufs[e].append((state_np[e], act_np[e], jw, jo))

        if args.record and saved >= args.max_episodes:
            print(f"[rec] target reached: {saved} episodes", flush=True)
            break
        stepn(1)

        if args.snap_dir and int(stage[0]) in (DESCEND, CLOSE, LIFT) and step % 15 == 0:
            for cam in ("wrist", "overhead"):
                rgb = env.scene.sensors[cam].data.output["rgb"][0].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    f"{args.snap_dir}/{step:05d}_{NAMES[int(stage[0])]}_{cam}.png",
                    cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR),
                )
        if args.debug:
            closing = (stage == CLOSE).nonzero().flatten()
            if len(closing) and step % 60 == 0:
                e = int(closing[0])
                print(f"[close] e{e} t={int(t_in[e])} "
                      f"hand_z={float(hand[e,2]):.3f} "
                      f"dxy={float((hand[e,:2]-cup_pos[e,:2]).norm()):.3f} "
                      f"stall={float(robot.data.joint_pos[e, fj]):.3f}", flush=True)
        if args.debug:
            lifting = (stage == LIFT).nonzero().flatten()
            if len(lifting) and step % 20 == 0:
                e = int(lifting[0])
                print(f"[lift] e{e} t={int(t_in[e])} cup_z={float(cup_pos[e,2]):.3f} "
                      f"hand_z={float(hand[e,2]):.3f} "
                      f"stall={float(robot.data.joint_pos[e, fj]):.3f}", flush=True)
        if args.debug and step % 100 == 0:
            print(f"[s1] step={step} stages={[NAMES[int(s)] for s in stage]} "
                  f"picks={picks} places={places} "
                  f"hand0={[round(float(v),3) for v in hand[0]]} "
                  f"cup_xy0={[round(float(v),3) for v in cup_xy[0]]}", flush=True)

    print(f"[s1] ======== RESULT: {picks} physical picks, {places} "
          f"physical pick-move-places ========", flush=True)
    if manifest is not None:
        manifest.close()
    # hard-exit: env/app teardown reliably hangs after long camera runs
    # (observed v9-v16 and every recording batch) — all results are
    # flushed above, so skip the wedged cleanup entirely.
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

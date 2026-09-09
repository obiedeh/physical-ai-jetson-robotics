"""Stage-level closed-loop evaluation of a GR00T checkpoint (Goal 1/2/4).

Extends eval_gr00t_franka.py with:
- a machine-readable per-episode ledger (JSONL) recording spawn pose,
  stage signals, first failed stage, and final outcome;
- a failure-taxonomy classifier driven by measured signals (hand-cup
  distance, finger closure timing, cup kinematics);
- ISOLATED SKILL TESTS via scripted takeover: the harness drives the
  arm with ground-truth-based scripted control through the setup phases,
  then hands control to the policy at --takeover_stage:
      none       policy controls the whole episode (full task)
      grasp      scripted to a verified pregrasp; policy grasps + lifts
      transport  scripted through a held lift; policy carries
      placement  scripted through the carry; policy places + releases
  GR00T N1.7 conditions only on the current observation (delta_indices
  [0]), so mid-episode handover is in-distribution at the frame level.

Server (gr00t venv):
    HF_TOKEN=$(cat ~/.cache/huggingface/token) ~/.venv/gr00t/bin/python \
      gr00t/eval/run_gr00t_server.py --model-path <ckpt> \
      --embodiment-tag new_embodiment --port 5594

Client (Isaac venv):
    ~/.venv/isaacsim5/bin/python isaac/scripts/evaluate_stage_pipeline.py \
      --headless --port 5594 --num_envs 4 --episodes 40 \
      --ledger reports/closed_loop_v6d.jsonl [--takeover_stage grasp]

Safety: arm targets are clamped inside joint limits every step; gripper
command clamped 0..0.79; simulation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_INSTRUCTION = (
    "pick up the cup, carry it, and set it back down where it started"
)
EXEC_HORIZON = 8
EP_STEPS = 900
HAND_GRASP_Z = 0.187
HAND_TRAVEL_Z = 0.35
CARRY_DY = -0.18


class MiniGrootClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack = msgpack
        self._mnp = mnp
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def get_action(self, observation: dict) -> dict:
        req = self._msgpack.packb(
            {"endpoint": "get_action", "data": {"observation": observation}},
            default=self._mnp.encode,
        )
        self.socket.send(req)
        msg = self.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("policy server error")
        resp = self._msgpack.unpackb(msg, object_hook=self._mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server: {resp['error']}")
        return resp


def classify_failure(sig: dict) -> str:
    """Map measured episode signals to one primary failure label."""
    if sig["outcome"] == "success":
        return "none"
    # never approached the cup region
    if sig["min_hand_cup_xy"] > 0.08:
        return "approach_misaligned"
    # closed the fingers while still far from the cup
    if sig["closed_far_from_cup"]:
        return "premature_gripper_close"
    # got close, closed near the cup, but the cup never rose
    if sig["max_cup_z"] < 0.05 and sig["closed_near_cup"]:
        return "failed_grasp"
    # partial rise then early drop
    if 0.05 <= sig["max_cup_z"] <= 0.10:
        return "unstable_grasp"
    if sig["max_cup_z"] > 0.10 and sig["dropped_mid_air"]:
        return "object_slip"
    if sig["max_cup_z"] > 0.10 and not sig["reached_target_region"]:
        return "transport_drift"
    if sig["max_cup_z"] > 0.10 and sig["reached_target_region"]:
        return "placement_misaligned"
    if not sig["closed_near_cup"] and sig["min_hand_cup_xy"] <= 0.08:
        return "incorrect_pregrasp"
    return "unknown"


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5594)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed_spawn", action="store_true")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--takeover_stage",
        choices=("none", "grasp", "transport", "placement"),
        default="none",
    )
    parser.add_argument("--ledger", type=str, required=True)
    parser.add_argument("--exec_horizon", type=int, default=EXEC_HORIZON,
                        help="steps of each 16-step chunk executed before replanning")
    parser.add_argument(
        "--gripper_latch",
        action="store_true",
        help="diagnostic shim: once a genuine stall (finger 0.46-0.70) is "
        "measured with the cup lifted, override the policy's gripper "
        "command to 0.79 until the hand reaches the place region — "
        "quantifies the ceiling with the gripper channel fixed",
    )
    parser.add_argument(
        "--gripper_override",
        action="store_true",
        help="diagnostic ceiling test: gripper channel FULLY scripted from "
        "measured geometry (pre-shape -> close when aligned low over the "
        "cup -> hold -> open at the place region); arm remains 100%% "
        "policy-driven. Splits arm-channel vs gripper-channel blame.",
    )
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
        import os

        os._exit(1)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch  # type: ignore[import-not-found]

    from isaaclab.envs import ManagerBasedEnv  # type: ignore[import-not-found]
    from isaaclab.utils.math import (  # type: ignore[import-not-found]
        axis_angle_from_quat,
        quat_apply,
        quat_from_angle_axis,
        quat_inv,
        quat_mul,
    )

    from isaac.isaaclab_tasks.synria_pickplace.franka_cup_env import (
        FrankaCupEnvCfg,
    )

    torch.manual_seed(args.seed)
    cfg = FrankaCupEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg)
    env.reset()
    robot = env.scene["robot"]
    cup = env.scene["cup"]
    n = env.scene.num_envs
    dev = env.device
    origins = env.scene.env_origins

    hand_idx = robot.body_names.index("panda_hand")
    fj = robot.joint_names.index("finger_joint")
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    lo = robot.data.joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.joint_pos_limits[:, arm_ids, 1]
    default_arm = robot.data.default_joint_pos[:, arm_ids]

    jt = robot.data.default_joint_pos.clone()

    def stepn(k: int) -> None:
        for _ in range(k):
            act = torch.zeros(n, 8, device=dev)
            act[:, :7] = jt[:, arm_ids] - default_arm
            act[:, 7] = jt[:, fj]
            env.step(act)

    stepn(120)

    # tool-down attitude lock (S1 v15)
    lock_quat = robot.data.body_quat_w[:, hand_idx].clone()
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    a0 = quat_apply(lock_quat, ez)
    sgn = torch.where(a0[:, 2:3] < 0, -1.0, 1.0)
    tgtax = torch.cat([torch.zeros(n, 2, device=dev), sgn], dim=1)
    axis = torch.cross(a0, tgtax, dim=1)
    s_ = axis.norm(dim=1, keepdim=True)
    c_ = (a0 * tgtax).sum(dim=1, keepdim=True)
    ang = torch.atan2(s_, c_).squeeze(-1)
    lock_quat = quat_mul(
        quat_from_angle_axis(ang, axis / s_.clamp(min=1e-8)), lock_quat
    )

    # empirical jacobian row (S1)
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
    print(f"[stage-eval] jac row {jac_row} (cos={scores[jac_row]:.2f})", flush=True)

    def servo_step(target: torch.Tensor, rate: float = 0.03) -> None:
        jac = robot.root_physx_view.get_jacobians()[:, jac_row, :, :][:, :, arm_ids]
        pos = robot.data.body_pos_w[:, hand_idx] - origins
        quat = robot.data.body_quat_w[:, hand_idx]
        perr = target - pos
        rerr = axis_angle_from_quat(quat_mul(lock_quat, quat_inv(quat)))
        err = torch.cat([perr, 0.5 * rerr], dim=1)
        jT = jac.transpose(1, 2)
        a = jac @ jT + 0.0025 * torch.eye(6, device=dev).expand(n, 6, 6)
        dq = (jT @ torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)
        upd = (jt[:, arm_ids] + dq.clamp(-rate, rate)).clamp(lo, hi)
        jt[:, arm_ids] = upd

    def servo_until(target_fn, tol: float, max_steps: int, rate: float = 0.03) -> bool:
        for _ in range(max_steps):
            tgt = target_fn()
            servo_step(tgt, rate)
            stepn(1)
            hand = robot.data.body_pos_w[:, hand_idx] - origins
            if bool(((hand - tgt).norm(dim=1) < tol).all()):
                return True
        return False

    def ramp_finger(target: float, steps: int) -> None:
        for _ in range(steps):
            jt[:, fj] = jt[:, fj] + float(
                np.clip(target - float(jt[0, fj]), -0.026, 0.013)
            )
            stepn(1)

    client = MiniGrootClient(args.host, args.port)
    print(f"[stage-eval] policy server {args.host}:{args.port}", flush=True)

    def frames() -> tuple[np.ndarray, np.ndarray]:
        outs = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            outs.append(rgb[..., :3])
        return outs[0], outs[1]

    def respawn() -> torch.Tensor:
        root = cup.data.default_root_state.clone()
        rand = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
        if args.fixed_spawn:
            rand[:] = 0.0
        root[:, 0] = 0.5 + rand[:, 0]
        root[:, 1] = rand[:, 1]
        root[:, 2] = 0.021
        root[:, :3] += origins
        root[:, 7:] = 0.0
        ids = torch.arange(n, device=dev)
        cup.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        cup.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        jt.copy_(robot.data.default_joint_pos)
        return root[:, :2] - origins[:, :2]

    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = ledger_path.open("a")

    episodes_done = 0
    successes = 0
    window_idx = 0

    while episodes_done < args.episodes:
        spawn_xy = respawn()
        stepn(60)
        cup_gt = cup.data.root_pos_w - origins
        target_xy = spawn_xy.clone()
        target_xy[:, 1] += CARRY_DY

        # ---- scripted setup for takeover modes (GT-based test harness) ----
        setup_ok = torch.ones(n, dtype=torch.bool, device=dev)
        if args.takeover_stage != "none":
            cxy = (cup.data.root_pos_w - origins)[:, :2]
            hi_t = torch.cat(
                [cxy, torch.full((n, 1), HAND_TRAVEL_Z, device=dev)], dim=1
            )
            ramp_finger(0.35, 40)
            servo_until(lambda: hi_t, 0.012, 500)
            lo_t = torch.cat(
                [cxy, torch.full((n, 1), HAND_GRASP_Z, device=dev)], dim=1
            )
            servo_until(lambda: lo_t, 0.010, 500, rate=0.01)
            if args.takeover_stage in ("transport", "placement"):
                ramp_finger(0.79, 120)
                stall = robot.data.joint_pos[:, fj]
                setup_ok = (stall > 0.46) & (stall < 0.70)
                lift_t = torch.cat(
                    [cxy, torch.full((n, 1), HAND_TRAVEL_Z, device=dev)], dim=1
                )
                for _ in range(400):
                    servo_step(lift_t, rate=0.006)
                    stepn(1)
                    if bool(((cup.data.root_pos_w - origins)[:, 2] > 0.10).all()):
                        break
                setup_ok &= (cup.data.root_pos_w - origins)[:, 2] > 0.10
            if args.takeover_stage == "placement":
                carry_t = torch.cat(
                    [target_xy, torch.full((n, 1), HAND_TRAVEL_Z, device=dev)],
                    dim=1,
                )
                for _ in range(400):
                    servo_step(carry_t, rate=0.008)
                    stepn(1)
                    hand = robot.data.body_pos_w[:, hand_idx] - origins
                    if bool(((hand - carry_t).norm(dim=1) < 0.03).all()):
                        break
                setup_ok &= (cup.data.root_pos_w - origins)[:, 2] > 0.10

        # ---- policy control window with signal tracking ----
        min_hand_cup = torch.full((n,), 1e9, device=dev)
        min_dxy_at_gh = torch.full((n,), 1e9, device=dev)  # at grasp height
        min_hand_z = torch.full((n,), 1e9, device=dev)
        max_cup_z = torch.zeros(n, device=dev)
        closed_far = torch.zeros(n, dtype=torch.bool, device=dev)
        closed_near = torch.zeros(n, dtype=torch.bool, device=dev)
        dropped_mid = torch.zeros(n, dtype=torch.bool, device=dev)
        was_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        first_close_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        chunk = None
        chunk_i = 0
        latched = torch.zeros(n, dtype=torch.bool, device=dev)
        # gripper-override state machine: 0 preshape, 1 closing, 2 hold,
        # 3 released
        gphase = torch.zeros(n, dtype=torch.long, device=dev)
        gcmd = torch.full((n,), 0.35, device=dev)
        policy_steps = EP_STEPS if args.takeover_stage == "none" else 400

        for t in range(policy_steps):
            if chunk is None or chunk_i >= args.exec_horizon:
                js = robot.data.joint_pos
                wrist_f, over_f = frames()
                obs = {
                    "video": {"wrist": wrist_f[:, None], "overhead": over_f[:, None]},
                    "state": {
                        "arm": js[:, arm_ids].cpu().numpy().astype(np.float32)[:, None],
                        "gripper": js[:, [fj]].cpu().numpy().astype(np.float32)[:, None],
                    },
                    "language": {
                        "annotation.human.task_description": [[args.instruction]] * n,
                    },
                }
                resp = client.get_action(obs)
                if isinstance(resp, (list, tuple)):
                    resp = resp[0]
                arm_c = np.asarray(resp["arm"], dtype=np.float32)
                grip_c = np.asarray(resp["gripper"], dtype=np.float32)
                if arm_c.ndim == 2:
                    arm_c, grip_c = arm_c[None], grip_c[None]
                chunk = np.concatenate([arm_c, grip_c], axis=-1)
                chunk_i = 0
            act = torch.from_numpy(chunk[:, min(chunk_i, chunk.shape[1] - 1)]).to(dev)
            chunk_i += 1
            tgt = (default_arm + act[:, :7]).clamp(lo, hi)
            grip_cmd = act[:, 7:8].clamp(0.0, 0.79)
            if args.gripper_override:
                hand_now = robot.data.body_pos_w[:, hand_idx] - origins
                cpos_now = cup.data.root_pos_w - origins
                dxy_now = (hand_now[:, :2] - cpos_now[:, :2]).norm(dim=1)
                aligned = (
                    (dxy_now < 0.045)
                    & (hand_now[:, 2] > 0.16)
                    & (hand_now[:, 2] < 0.22)
                )
                meas = robot.data.joint_pos[:, fj]
                cz = cpos_now[:, 2]
                hxy_t = (hand_now[:, :2] - target_xy).norm(dim=1)
                start_close = (gphase == 0) & aligned
                gphase[start_close] = 1
                to_hold = (gphase == 1) & (meas > 0.46)
                gphase[to_hold] = 2
                release = (gphase == 2) & (hxy_t < 0.05) & (cz > 0.05)
                gphase[release] = 3
                tgt_f = torch.where(
                    gphase == 0,
                    torch.full((n,), 0.35, device=dev),
                    torch.where(
                        gphase == 3,
                        torch.zeros(n, device=dev),
                        torch.full((n,), 0.79, device=dev),
                    ),
                )
                gcmd = gcmd + (tgt_f - gcmd).clamp(-0.026, 0.013)
                grip_cmd = gcmd.unsqueeze(-1)
            elif args.gripper_latch:
                meas = robot.data.joint_pos[:, fj]
                cz = (cup.data.root_pos_w - origins)[:, 2]
                hxy = (robot.data.body_pos_w[:, hand_idx] - origins)[:, :2]
                latched |= (meas > 0.46) & (meas < 0.70) & (cz > 0.08)
                at_place = (hxy - target_xy).norm(dim=1) < 0.04
                latched &= ~at_place  # release allowed at the target
                grip_cmd = torch.where(
                    latched.unsqueeze(-1),
                    torch.full_like(grip_cmd, 0.79),
                    grip_cmd,
                )
            act8 = torch.cat([tgt - default_arm, grip_cmd], dim=1)
            env.step(act8)

            hand = robot.data.body_pos_w[:, hand_idx] - origins
            cpos = cup.data.root_pos_w - origins
            dxy = (hand[:, :2] - cpos[:, :2]).norm(dim=1)
            min_hand_cup = torch.minimum(min_hand_cup, dxy)
            at_gh = (hand[:, 2] > 0.15) & (hand[:, 2] < 0.23)
            min_dxy_at_gh = torch.where(
                at_gh, torch.minimum(min_dxy_at_gh, dxy), min_dxy_at_gh
            )
            min_hand_z = torch.minimum(min_hand_z, hand[:, 2])
            max_cup_z = torch.maximum(max_cup_z, cpos[:, 2])
            was_lifted |= cpos[:, 2] > 0.10
            dropped_mid |= was_lifted & (cpos[:, 2] < 0.05) & (
                (cpos[:, :2] - target_xy).norm(dim=1) > 0.08
            )
            closing = robot.data.joint_pos[:, fj] > 0.46
            newly = closing & (first_close_step < 0)
            first_close_step[newly] = t
            closed_far |= newly & (dxy > 0.05)
            closed_near |= newly & (dxy <= 0.05)

        # ---- outcome + ledger ----
        cpos = cup.data.root_pos_w - origins
        hand = robot.data.body_pos_w[:, hand_idx] - origins
        on_ground = cpos[:, 2] < 0.05
        near_target = (cpos[:, :2] - target_xy).norm(dim=1) < 0.08
        if args.takeover_stage == "none":
            success = was_lifted & on_ground & near_target
        elif args.takeover_stage == "grasp":
            success = was_lifted
        elif args.takeover_stage == "transport":
            success = cpos[:, 2] > 0.05  # retained through the window
        else:  # placement
            success = on_ground & near_target

        for e in range(n):
            sig = {
                "min_hand_cup_xy": float(min_hand_cup[e]),
                "min_dxy_at_grasp_height": round(
                    min(float(min_dxy_at_gh[e]), 9.9), 4
                ),
                "min_hand_z": round(float(min_hand_z[e]), 4),
                "max_cup_z": float(max_cup_z[e]),
                "closed_far_from_cup": bool(closed_far[e]),
                "closed_near_cup": bool(closed_near[e]),
                "dropped_mid_air": bool(dropped_mid[e]),
                "reached_target_region": bool(near_target[e]),
                "outcome": "success" if bool(success[e]) else "failure",
            }
            rec = {
                "episode": episodes_done + e,
                "window": window_idx,
                "env": e,
                "takeover_stage": args.takeover_stage,
                "instruction": args.instruction,
                "fixed_spawn": bool(args.fixed_spawn),
                "setup_ok": bool(setup_ok[e]),
                "spawn_xy": [round(float(v), 4) for v in spawn_xy[e]],
                "target_xy": [round(float(v), 4) for v in target_xy[e]],
                "final_cup": [round(float(v), 4) for v in cpos[e]],
                "final_hand": [round(float(v), 4) for v in hand[e]],
                "first_close_step": int(first_close_step[e]),
                "final_finger": round(float(robot.data.joint_pos[e, fj]), 3),
                **sig,
                "first_failed_stage": classify_failure(sig),
                "safety_intervention": False,
            }
            ledger.write(json.dumps(rec) + "\n")
        ledger.flush()
        successes += int(success.sum())
        episodes_done += n
        window_idx += 1
        print(
            f"[stage-eval] window {window_idx}: {int(success.sum())}/{n} | "
            f"totals {successes}/{episodes_done}",
            flush=True,
        )

    print(
        f"[stage-eval] ======== RESULT ({args.takeover_stage}): "
        f"{successes}/{episodes_done} ========",
        flush=True,
    )
    ledger.close()
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

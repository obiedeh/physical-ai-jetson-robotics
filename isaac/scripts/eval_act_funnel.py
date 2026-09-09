#!/usr/bin/env python3
"""ACT under the FUNNEL harness — the missing row of the three-arm table.

Same trial-manager funnel counting as eval_gr00t_sequence.py (grasp,
carry5s, dock, cycle per episode-equivalent), same scene/seed convention,
but actions come from the ACT policy bridge (pickle-over-TCP, Thor :5599).
No retry supervisor — this is the bare-policy row, directly comparable to
the GR00T arms. num_envs is fixed at 1 (ACT temporal ensembling is
batch-global).

    ~/.venv/isaacsim5/bin/python isaac/scripts/eval_act_funnel.py \
        --host 192.168.1.170 --port 5599 --steps 14400 --headless
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TASK = "pick up the cup, carry it to the plate, and set it down gently"


def send_msg(conn, obj):
    buf = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(buf)) + buf)


def recv_msg(conn):
    hdr = conn.recv(8, socket.MSG_WAITALL)
    (size,) = struct.unpack("<Q", hdr)
    return pickle.loads(conn.recv(size, socket.MSG_WAITALL))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.1.170")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--steps", type=int, default=14400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--staged", action="store_true",
                        help="demo-distribution starts (80% zone surface)")
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
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
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z

    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.8 if args.staged else 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg(
        "Synria-Chess-PickPlace-v0", device="cuda", num_envs=1
    )
    env_cfg.seed = args.seed
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    add_recorder_cameras(env_cfg)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    env.reset()
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    lf, rf = int(rids["left"]), int(rids["right"])
    default_l = float(robot.data.default_joint_pos[0, lf])
    mgr = mdp._get_trial_mgr(env)

    conn = socket.create_connection((args.host, args.port))
    print(f"[act-funnel] connected to {args.host}:{args.port}", flush=True)

    def frames():
        out = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"][0]
            fr = rgb.detach().cpu().numpy()
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
            out.append(fr[..., :3])
        return out

    totals = {"grasp": 0, "carry5s": 0, "dock": 0, "cycle": 0}
    prev_cycles = mgr.cycles.clone()
    prev_f = {"carry5s": mgr.f_carry5s.clone(), "dock": mgr.f_dock.clone()}
    prev_phase = mgr.phase.clone()
    prev_ep = int(env.episode_length_buf[0])
    need_reset = True

    for step in range(args.steps):
        js = robot.data.joint_pos
        state8 = torch.cat([js[:, arm_ids], js[:, [lf, rf]]],
                           dim=1)[0].cpu().numpy().astype(np.float32)
        wr, ov = frames()
        send_msg(conn, {"state": state8, "wrist": wr, "overhead": ov,
                        "reset": need_reset, "task": TASK})
        need_reset = False
        act8 = np.asarray(recv_msg(conn)["action"], dtype=np.float32)
        action = torch.zeros(1, 7, device=env.device)
        action[0, :6] = torch.from_numpy(act8[:6]).to(env.device)
        l_target = default_l + float(act8[6])
        action[0, 6] = 1.0 if l_target > 0.020 else -1.0
        env.step(action)

        totals["grasp"] += int(((prev_phase == 0) & (mgr.phase == 1)).sum())
        for k in ("carry5s", "dock"):
            cur = getattr(mgr, f"f_{k}")
            totals[k] += int((cur & ~prev_f[k]).sum())
            prev_f[k] = cur.clone()
        totals["cycle"] += int(((mgr.cycles - prev_cycles) > 0).sum())
        prev_cycles = mgr.cycles.clone()
        prev_phase = mgr.phase.clone()

        cur_ep = int(env.episode_length_buf[0])
        if cur_ep < prev_ep:
            need_reset = True  # episode rolled: fresh server chunk queue
        prev_ep = cur_ep

        if step % 1200 == 0:
            print(f"[act-funnel] step={step} totals={totals}", flush=True)

    ep_len = int(env.max_episode_length)
    ep_equiv = args.steps / ep_len
    print("[act-funnel] ======== ACT FUNNEL RESULT ========", flush=True)
    print(f"[act-funnel] episode-equivalents: {ep_equiv:.1f}", flush=True)
    for k, v in totals.items():
        print(f"[act-funnel] {k}: {v} total, {v / ep_equiv:.3f}/episode",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

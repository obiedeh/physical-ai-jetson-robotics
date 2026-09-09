"""Closed-loop evaluation of the ACT policy via the policy server.

Isaac venv side of the bridge: builds the RECORDER env variant (wrist +
overhead cameras — the observations ACT was trained on), streams
state+frames to serve_act_policy.py on localhost, applies the returned
8-dim corpus-convention actions through the env's 7-dim interface, and
scores the same latched stages as eval_synria_sequence.py so the numbers
sit directly beside the E26 baseline (place 0.132, full 0.176 on s42).

Action mapping (corpus convention -> env):
    dims 0..5 : arm joint offsets from default  -> env action[:6] unchanged
    dims 6,7  : finger-target offsets           -> binary gripper: the corpus
                encodes open as left target 0.025 and close as 0.0155, so
                left_target > 0.020 -> +1 (open), else -1 (close)

num_envs=1 by design: ACT's temporal-ensembling queue is batch-global, so
mixing envs with different episode boundaries corrupts the chunk blending.
Serial episodes are slower and correct.

    ~/.venv/isaacsim5/bin/python isaac/scripts/eval_act_synria.py --headless \
        [--episodes 40] [--port 5599] [--seed 123] [--out reports/eval/act_v1.json]

Start the server FIRST (lerobot venv, see serve_act_policy.py).
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stats import wilson_interval  # noqa: E402

REACH_R = 0.10
ALIGN_XY = 0.025
ALIGN_Z = 0.02
UPRIGHT_COS = 0.966
LIFT_STEPS = 10
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
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--host", default="127.0.0.1",
                        help="policy-server address — point at the AGX Thor "
                             "to evaluate a Thor-served policy in this sim")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=0,
                        help="RETRY SUPERVISOR: up to N scripted grasp-recovery "
                             "interventions per episode. Detects an empty pinch "
                             "(width < 33mm in PICK) or a drop, opens the "
                             "gripper, lifts the arm back toward its ready "
                             "altitude, RESETS the policy's chunk queue, and "
                             "returns control. Compound odds: 0.325/attempt "
                             "-> ~0.69 at three attempts.")
    parser.add_argument("--horizon_scale", type=float, default=1.0,
                        help="episode length multiplier (retries need room; "
                             "A/B arms must use the SAME value)")
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
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
        add_recorder_cameras,
    )
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg("Synria-Chess-PickPlace-v0", device="cuda", num_envs=1)
    if args.horizon_scale != 1.0:
        env_cfg.episode_length_s = env_cfg.episode_length_s * args.horizon_scale
    env_cfg.seed = args.seed
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    add_recorder_cameras(env_cfg)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    lf, rf = int(rids["left"]), int(rids["right"])
    mgr = mdp._get_trial_mgr(env)
    default_arm = robot.data.default_joint_pos[:, arm_ids].clone()
    default_l = float(robot.data.default_joint_pos[0, lf])
    rest_z = TABLE_SURFACE_Z + 0.05 / 2.0

    conn = socket.create_connection((args.host, args.port))
    print(f"[eval-act] connected to policy server {args.host}:{args.port}", flush=True)

    env.reset()

    stages = ("reach", "align", "grasp", "grasp_upright", "lift", "transport",
              "at_plate", "at_plate_upright", "place", "upright", "full")
    # --- retry supervisor state ---
    RECOVERING, HANDBACK = 1, 2
    sup_mode = 0            # 0 = policy driving
    sup_t = 0
    attempts_left = args.retries
    retry_events = 0
    pinch_run = 0
    was_lifted = False
    default_arm_t = default_arm.clone()
    totals = {k: 0 for k in stages}
    episodes_done = 0
    f = {k: False for k in stages}
    lift_run = 0
    prev_phase = int(mgr.phase[0])
    prev_cycles = int(mgr.cycles[0])
    prev_ep = int(env.episode_length_buf[0])
    need_reset = True

    def frames():
        out = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"][0]
            fr = rgb.detach().cpu().numpy()
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
            out.append(fr[..., :3])
        return out

    while episodes_done < args.episodes:
        js = robot.data.joint_pos
        state8 = torch.cat([js[:, arm_ids], js[:, [lf, rf]]],
                           dim=1)[0].cpu().numpy().astype(np.float32)
        wr, ov = frames()
        if sup_mode == 0:
            send_msg(conn, {"state": state8, "wrist": wr, "overhead": ov,
                            "reset": need_reset, "task": TASK})
            need_reset = False
            act8 = np.asarray(recv_msg(conn)["action"], dtype=np.float32)
            action = torch.zeros(1, 7, device=env.device)
            action[0, :6] = torch.from_numpy(act8[:6]).to(env.device)
            l_target = default_l + float(act8[6])
            action[0, 6] = 1.0 if l_target > 0.020 else -1.0
        else:
            # SCRIPTED RECOVERY: gripper open, raise shoulder/elbow toward
            # their ready-pose values (lifts the hand clear; yaw untouched so
            # the hand stays in the cup's sector — the policy restarts from a
            # hover state it has seen thousands of in training)
            cur = robot.data.joint_pos[:, arm_ids].clone()
            tgt = cur.clone()
            tgt[0, 1] = default_arm_t[0, 1]
            tgt[0, 2] = default_arm_t[0, 2]
            step_q = cur + (tgt - cur) * 0.06
            action = torch.zeros(1, 7, device=env.device)
            action[0, :6] = step_q[0] - default_arm_t[0]
            action[0, 6] = 1.0                       # open
            sup_t += 1
            lifted_enough = float(mdp._grasp_center_local(env)[0, 2]) >                 float(TABLE_SURFACE_Z) + 0.11
            if sup_t > 140 or (sup_t > 40 and lifted_enough):
                sup_mode, sup_t = 0, 0
                need_reset = True                    # fresh chunk queue
        env.step(action)

        piece = mdp._get_piece_pos(env)[0]
        gc = mdp._grasp_center_local(env)[0]
        width = float(mdp._gripper_width_m(env).squeeze(-1)[0])
        phase = int(mgr.phase[0])
        lifted = bool(mdp._piece_lifted_mask(env)[0])
        try:
            q = env.scene["piece"].data.root_quat_w[0]
            upright_now = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)) >= UPRIGHT_COS
        except (KeyError, AttributeError):
            upright_now = True

        d3 = float(torch.norm(gc - piece))
        dxy = float(torch.norm(gc[:2] - piece[:2]))
        f["reach"] |= d3 <= REACH_R
        f["align"] |= (dxy <= ALIGN_XY
                       and abs(float(gc[2] - piece[2])) <= ALIGN_Z
                       and width >= 0.04)
        grasp_now = (prev_phase == int(TaskPhase.PICK_FROM_BOARD)
                     and phase == int(TaskPhase.PLACE_ON_ZONE))
        f["grasp"] |= grasp_now
        if grasp_now:
            f["grasp_upright"] |= upright_now
        lift_run = lift_run + 1 if lifted else 0
        f["lift"] |= lift_run >= LIFT_STEPS
        f["transport"] |= (prev_phase == int(TaskPhase.PLACE_ON_ZONE)
                           and phase == int(TaskPhase.PICK_FROM_ZONE))
        at_plate_now = (phase == int(TaskPhase.PICK_FROM_ZONE) and lifted
                        and float(torch.norm(piece[:2] - mgr.zone_xy[0])) <= 0.15)
        f["at_plate"] |= at_plate_now
        if at_plate_now:
            f["at_plate_upright"] |= upright_now
        place_now = (prev_phase == int(TaskPhase.PICK_FROM_ZONE)
                     and phase == int(TaskPhase.RETURN_TO_BOARD))
        f["place"] |= place_now
        if place_now:
            f["upright"] |= upright_now
        cyc = int(mgr.cycles[0])
        f["full"] |= cyc > prev_cycles
        prev_phase, prev_cycles = phase, cyc

        # --- retry supervisor detection (policy-driving only) -------------
        if args.retries and sup_mode == 0 and attempts_left > 0:
            in_pick = phase == int(TaskPhase.PICK_FROM_BOARD)
            pinch_run = pinch_run + 1 if (in_pick and width < 0.033) else 0
            dropped = (was_lifted and not lifted
                       and float(piece[2]) < rest_z + 0.008 and width < 0.035)
            if pinch_run > 12 or dropped:
                sup_mode, sup_t = RECOVERING, 0
                attempts_left -= 1
                retry_events += 1
                pinch_run = 0
        was_lifted = lifted

        ep = int(env.episode_length_buf[0])
        if ep < prev_ep:                      # episode boundary
            episodes_done += 1
            for k in stages:
                totals[k] += int(f[k])
                f[k] = False
            lift_run = 0
            need_reset = True                 # clear the ACT chunk queue
            sup_mode, sup_t = 0, 0
            attempts_left = args.retries
            pinch_run = 0
            was_lifted = False
            if episodes_done % 5 == 0:
                print(f"[eval-act] {episodes_done}/{args.episodes} episodes | "
                      + " ".join(f"{k}={totals[k]}" for k in
                                 ("grasp", "at_plate", "place", "full")),
                      flush=True)
        prev_ep = ep

    print("[eval-act] ======== RESULT ========", flush=True)
    report = {"episodes": episodes_done, "eval_seed": args.seed,
              "port": args.port, "retries": args.retries,
              "horizon_scale": args.horizon_scale,
              "retry_events": retry_events, "stages": {}}
    print(f"[eval-act] retry supervisor: budget={args.retries}/ep, "
          f"interventions fired={retry_events}", flush=True)
    for k in stages:
        iv = wilson_interval(totals[k], episodes_done)
        report["stages"][k] = {"successes": totals[k], "trials": episodes_done,
                               "rate": iv.rate, "ci95_low": iv.low,
                               "ci95_high": iv.high}
        print(f"[eval-act] stage {k}: {totals[k]}/{episodes_done} = "
              f"{iv.rate:.3f}  95% CI [{iv.low:.3f}, {iv.high:.3f}]", flush=True)
    out = args.out or (_REPO_ROOT / "reports/eval/act_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[eval-act] report: {out}", flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

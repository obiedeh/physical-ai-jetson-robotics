"""Ludo turn executor — the product's first closed game loop (P2 MVP).

The ludo_engine (L4) plays the game; each TurnPlan's PickPlaceCommands
are executed PHYSICALLY by the proven E33 scripted expert (12/12 release
discipline) in the pick-place env. Board-state sync: the physical piece
is teleported to each command's pick_xy before execution (the sim table
stands in for a full 16-token board in v0 — one token is physical at a
time, the game engine owns the rest of the state). The place target
reuses the expert's zone machinery: mgr.zone_xy is overridden per
command, so the entire tuned approach/descend/release chain works for
arbitrary squares.

Outputs a session dir: frames/f_*.png (overhead), turns.jsonl (per
command result), meta.jsonl (per frame). Compose with
isaac/scripts/ludo_compose.py.

    ~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py \
        --headless --turns 6 --out reports/ludo_demo_01

GR00T closed-loop lane (serve_groot17.py listening on :5591; same
scoring rule, random reachable square pairs, writes the same
turns.jsonl/session_summary.json so ludo_stats.py aggregates it):

    ~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py \
        --headless --policy-port 5591 --commands 20 --seed 300 \
        --out reports/ludo_groot17_eval01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GRASP_Z = 0.040
PAD_UNDERHANG = 0.0361
CLEAR_RISE = 0.08
OPEN_TRIGGER = 0.012
BASE_XY = (-0.22, 0.0)
PREGRASP_ALT = 0.13
CMD_TIMEOUT_STEPS = 9000

(IDLE, PLACE_DESCEND, RELEASE, ASCEND, YAW_HOME, GO_HOME,
 S_YAW, S_PLANE, S_DESCEND, S_CLOSE, S_LIFT, S_CYAW, S_CPLANE) = range(13)

SCORING_RULE = ("OK iff final cup-to-square error < 35mm AND final cup "
                "tilt-from-vertical < 15deg; cup position read post-hold, "
                "post-withdrawal; misses never rescued by sim authority")


def _pctl(sorted_vals, p):
    """The one correct percentile in the portfolio (safety-observability
    convention): round((p/100)*(n-1)) index on sorted values."""
    if not sorted_vals:
        return None
    return sorted_vals[round((p / 100.0) * (len(sorted_vals) - 1))]


def _gpu_health_sample():
    """One nvidia-smi sample: (temp_C, power_W, throttle_active_bitmask)."""
    import subprocess as _sp
    try:
        out = _sp.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,"
             "clocks_event_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        t, p, r = [x.strip() for x in out.split(",")]
        return float(t), float(p), r
    except Exception:
        return None, None, None


def _write_provenance(args) -> None:
    """Run provenance, machine-emitted at startup (append-only file set:
    one provenance.json per --out dir; a re-run to the same dir appends
    a numbered variant rather than overwriting)."""
    import json as _json
    import platform
    import subprocess as _sp
    import sys as _sys
    import time as _time

    def _cmd(c):
        try:
            return _sp.run(c, shell=True, capture_output=True, text=True,
                           timeout=10).stdout.strip()
        except Exception:
            return "unavailable"

    prov = {
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "git_sha": _cmd("git rev-parse HEAD"),
        "git_dirty": bool(_cmd("git status --porcelain")),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": _sys.version.split()[0],
        "gpu": _cmd("nvidia-smi --query-gpu=name,driver_version "
                    "--format=csv,noheader"),
        "cuda_runtime": _cmd("nvcc --version | grep release || true"),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "argv": _sys.argv,
        "control_hz": 30,
        "data_kind": "simulated (Isaac Sim; kinematic attach glue + "
                     "release-hold sim-authority patches active — see "
                     "ledger; NOT real-hardware data)",
        "scoring_rule": SCORING_RULE,
    }
    # B6: assert the device instead of hoping (reuse the rtx_training
    # pattern) and record the resolved compute stack.
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false.")
        prov["torch_cuda_device"] = _torch.cuda.get_device_name(0)
        prov["torch_version"] = _torch.__version__
    except ImportError:
        prov["torch_cuda_device"] = "torch not importable at provenance time"
    try:
        import onnxruntime as _ort
        prov["onnxruntime_available_providers"] = _ort.get_available_providers()
    except ImportError:
        prov["onnxruntime_available_providers"] = None

    args.out.mkdir(parents=True, exist_ok=True)
    # B5: a dirty tree means the SHA does not describe the code that ran —
    # capture the diff so the run is reproducible, not merely flagged.
    if prov["git_dirty"]:
        import hashlib as _hl
        diff = _cmd("git diff HEAD")
        patch = args.out / "uncommitted.patch"
        i = 1
        while patch.exists():
            patch = args.out / f"uncommitted.{i}.patch"
            i += 1
        patch.write_text(diff)
        prov["uncommitted_patch"] = patch.name
        prov["uncommitted_patch_sha256"] = _hl.sha256(
            diff.encode()).hexdigest()
    p = args.out / "provenance.json"
    i = 1
    while p.exists():
        p = args.out / f"provenance.{i}.json"
        i += 1
    p.write_text(_json.dumps(prov, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", type=int, default=6,
                        help="physical commands to execute (turns without a "
                        "legal move roll by without the arm)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dice-in", type=int, default=0,
                        help="B2b step1: arm picks the die and drops it into "
                        "the cup, N attempts; reports success")
    parser.add_argument("--cup-test", type=int, default=0,
                        help="physical-dice: drop the die into the cup N times, "
                        "report containment rate")
    parser.add_argument("--dump", action="store_true",
                        help="B2b steps 2-4: after the die is in the cup, grab "
                        "the cup, shake it, invert-dump the die on the table, "
                        "and READ the rolled value (full physical roll)")
    parser.add_argument("--physical-rolls", type=int, default=0,
                        help="milestone C: play N game turns where each roll is "
                        "PHYSICALLY rolled (die->cup->shake->dump->read) and fed "
                        "to the engine via plan_turn(roll=value)")
    parser.add_argument("--roll-test", type=int, default=0,
                        help="physical-dice validation: spawn a die, throw it "
                        "N times, read each rest value, report distribution")
    parser.add_argument("--endgame", action="store_true",
                        help="2-player short game to a real winner (red vs "
                        "blue, tokens near home) for a complete-game demo; "
                        "runs until someone wins")
    parser.add_argument("--checkpoint", type=str,
                        default="reports/training/synria_chess_pickplace_v6_e26_s43/model_final.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=int, default=0,
                        help="record N successful random square-to-square "
                             "episodes (LeRobot raw format) instead of "
                             "playing a game")
    parser.add_argument("--policy-port", type=int, default=0,
                        help="GR00T closed-loop lane: drive every command "
                             "from a ZMQ policy server (serve_groot17.py) "
                             "on this port instead of the scripted expert; "
                             "--commands random square-to-square episodes "
                             "are scored with the SAME rule as the expert")
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--exec-horizon", type=int, default=8,
                        help="steps of each action chunk executed before "
                             "re-querying the server")
    parser.add_argument("--policy-timeout", type=int, default=6000,
                        help="per-command step budget in the policy lane "
                             "(corpus episodes run 3.6k-4.9k steps)")
    parser.add_argument("--instruction", type=str,
                        default="pick up the cup and place it on the target square",
                        help="language sent to the server; {pick}/{place} "
                             "expand to the track indices (corpus-01 used "
                             "the constant default — the place target was "
                             "NOT observable; see session_summary)")
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    _write_provenance(args)
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
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from PIL import Image
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
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
    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
        add_cup,
        add_die,
        add_recorder_cameras,
    )
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG

    from game_core import LudoGameAdapter
    from game_core.commands import PickPlaceCommand

    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg("Synria-Chess-PickPlace-v0", device="cuda",
                            num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    # ONE SESSION = ONE EPISODE. The 6x multiplier (5400 steps) let the
    # episode time_out fire mid-session: T1 spent 4286 steps, and T2's
    # carry crossed step 5400 — the env RESET mid-flight (trial mgr
    # attached=False, piece respawned by the randomizer 250mm away),
    # bypassing every executor guard. Deterministic, hence the
    # byte-identical demo_06/demo_07 T2 crashes at the same tick.
    env_cfg.episode_length_s = 36000.0
    add_recorder_cameras(env_cfg)
    if args.roll_test > 0:
        add_die(env_cfg)
    if args.cup_test > 0:
        add_die(env_cfg)
        add_cup(env_cfg)
    if args.dice_in > 0:
        add_die(env_cfg)
        add_cup(env_cfg)
    if args.physical_rolls > 0:
        add_die(env_cfg)
        add_cup(env_cfg)

    suppress_state: dict = {"mask": None}
    carry_guard: dict = {"on": False}
    _orig_attach = mdp.attach_carried_pieces

    def _gated_attach(env, env_ids):
        mgr2 = mdp._get_trial_mgr(env)
        mask = suppress_state["mask"]
        if mask is None:
            return _orig_attach(env, env_ids)
        saved_pin = mgr2.pin_steps.clone()
        mgr2.attached = mgr2.attached & ~mask
        mgr2.pin_steps = torch.where(mask, torch.full_like(saved_pin, 9999),
                                     saved_pin)
        pre = mgr2.attached.clone()
        _orig_attach(env, env_ids)
        mgr2.pin_steps = saved_pin
        mgr2.attached = mgr2.attached & ~mask
        if carry_guard["on"]:
            # CARRY GUARD: the trial mgr advances phases INSIDE the step
            # (piece crossing its zone) and the event's ~carryish clause
            # then detaches mid-flight — a per-tick phase pin loses that
            # race (demo_06 T2: cup dropped between telemetry ticks,
            # landed on its side, got plowed 450mm). While the executor
            # is carrying, only genuinely opened fingers may detach.
            w2 = mdp._gripper_width_m(env).squeeze(-1)
            mgr2.attached = mgr2.attached | (pre & (w2 < 0.042) & ~mask)
        return None

    env_cfg.events.attach_carried.func = _gated_attach

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    wrapper = RslRlVecEnvWrapper(raw)
    rc = TASK_TRAIN_CFG["Synria-Chess-PickPlace-v0"]()
    runner = OnPolicyRunner(wrapper, rc.to_dict(), log_dir=None, device="cuda")
    runner.load(str(Path(args.checkpoint)))
    policy = runner.get_inference_policy(device="cuda")
    print(f"[ludo] approach policy: {args.checkpoint}", flush=True)
    env = raw.unwrapped
    robot = env.scene["robot"]
    piece_obj = env.scene["piece"]
    if args.roll_test > 0:
        import numpy as _np
        from game_core.dice import die_value_up, settled, face_alignment
        die_obj = env.scene["die"]
        _rng = _np.random.default_rng(args.seed)
        _dev = env.device
        _die_xy = (0.15, 0.21)
        counts = {v: 0 for v in range(1, 7)}
        print(f"[dice] roll-test: {args.roll_test} throws", flush=True)
        for k in range(args.roll_test):
            q = _rng.normal(size=4); q = q / (_np.linalg.norm(q) + 1e-9)
            w = _rng.uniform(-45.0, 45.0, size=3)            # tumbling spin
            lv = _np.array([_rng.uniform(-0.05, 0.05),
                            _rng.uniform(-0.05, 0.05), 0.0])
            root = die_obj.data.default_root_state.clone()
            root[0, 0:3] = torch.tensor(
                [_die_xy[0], _die_xy[1], TABLE_SURFACE_Z + 0.22], device=_dev)
            root[0, 3:7] = torch.tensor(q, device=_dev, dtype=root.dtype)
            root[0, 7:10] = torch.tensor(lv, device=_dev, dtype=root.dtype)
            root[0, 10:13] = torch.tensor(w, device=_dev, dtype=root.dtype)
            die_obj.write_root_pose_to_sim(root[:, :7])
            die_obj.write_root_velocity_to_sim(root[:, 7:])
            zero_act = torch.zeros(1, 7, device=_dev); zero_act[:, 6] = -1.0
            val = None
            for t in range(800):
                wrapper.step(zero_act)
                quat = die_obj.data.root_quat_w[0].cpu().numpy()
                lin = die_obj.data.root_lin_vel_w[0].cpu().numpy()
                ang = die_obj.data.root_ang_vel_w[0].cpu().numpy()
                if t > 60 and settled(lin, ang, quat):
                    val = die_value_up(quat); break
            if val is None:
                val = die_value_up(die_obj.data.root_quat_w[0].cpu().numpy())
                print(f"[dice] throw {k+1}: NOT settled, read {val}", flush=True)
            else:
                print(f"[dice] throw {k+1}: {val} (settled at step {t})", flush=True)
            counts[val] += 1
        tot = sum(counts.values())
        print(f"[dice] distribution over {tot}: {counts}", flush=True)
        print(f"[dice] roll-test done", flush=True)
        import os as _os; _os._exit(0)
    if args.cup_test > 0:
        import numpy as _np
        from game_core.dice import die_value_up, settled, face_alignment
        die_obj = env.scene["die"]; cup_obj = env.scene["cup"]
        _rng = _np.random.default_rng(args.seed); _dev = env.device
        R_IN = 0.016; CUP_H = 0.040
        zero_act = torch.zeros(1, 7, device=_dev); zero_act[:, 6] = -1.0
        # let the cup settle on the table
        for _ in range(60):
            wrapper.step(zero_act)
        contained = 0
        for k in range(args.cup_test):
            cp = cup_obj.data.root_pos_w[0].cpu().numpy()
            q = _rng.normal(size=4); q = q / (_np.linalg.norm(q) + 1e-9)
            root = die_obj.data.default_root_state.clone()
            root[0, 0:3] = torch.tensor(
                [cp[0], cp[1], cp[2] + CUP_H + 0.04], device=_dev)  # above rim
            root[0, 3:7] = torch.tensor(q, device=_dev, dtype=root.dtype)
            root[0, 7:10] = torch.tensor([0, 0, -0.1], device=_dev, dtype=root.dtype)
            root[0, 10:13] = torch.tensor(
                _rng.uniform(-8, 8, size=3), device=_dev, dtype=root.dtype)
            die_obj.write_root_pose_to_sim(root[:, :7])
            die_obj.write_root_velocity_to_sim(root[:, 7:])
            val = None
            for t in range(500):
                wrapper.step(zero_act)
                dq = die_obj.data.root_quat_w[0].cpu().numpy()
                dl = die_obj.data.root_lin_vel_w[0].cpu().numpy()
                da = die_obj.data.root_ang_vel_w[0].cpu().numpy()
                if t > 60 and settled(dl, da, dq):
                    val = die_value_up(dq); break
            dp = die_obj.data.root_pos_w[0].cpu().numpy()
            cp = cup_obj.data.root_pos_w[0].cpu().numpy()
            r = float(_np.hypot(dp[0] - cp[0], dp[1] - cp[1]))
            inside = (r < R_IN) and (dp[2] > cp[2]) and (dp[2] < cp[2] + CUP_H + 0.01)
            contained += int(inside)
            print(f"[cup] drop {k+1}: {'IN ' if inside else 'OUT'} "
                  f"r={r*1000:.0f}mm dz={(dp[2]-cp[2])*1000:.0f}mm value={val}", flush=True)
        print(f"[cup] containment {contained}/{args.cup_test}", flush=True)
        import os as _os; _os._exit(0)
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    lf, rf = int(rids["left"]), int(rids["right"])
    tool_idx = int(rids["tool0"])
    dev = env.device
    n = 1
    mgr = mdp._get_trial_mgr(env)
    rest_z = TABLE_SURFACE_Z + 0.05 / 2.0
    pad_stop = TABLE_SURFACE_Z + PAD_UNDERHANG + 0.004
    default_arm = robot.data.default_joint_pos[:, arm_ids].clone()
    jlim = torch.tensor(
        [[-2.749, 2.749], [-2.0, 2.0], [-0.5, 3.14159],
         [-2.79, 2.79], [-1.57, 1.57], [-3.14159, 3.14159]], device=dev)

    def width_now():
        return mdp._gripper_width_m(env).squeeze(-1)

    # ---- corpus recording plumbing (E46-recorder-compatible raw) ----
    corpus_on = args.corpus > 0
    policy_on = args.policy_port > 0
    rec_buf: list = []
    _ids = mdp._robot_ids(env)
    finger_ids = ([_ids["left"], _ids["right"]]
                  if (corpus_on or policy_on) else None)
    if corpus_on or policy_on:
        default_fingers = robot.data.default_joint_pos[:, finger_ids].clone()
    if corpus_on:
        import cv2  # type: ignore[import-not-found]

    def rec_step(action_row):
        """Append one step: state8, action8 (6 arm offsets + 2 finger
        target offsets, recorder-verbatim contract), wrist/overhead jpg."""
        import cv2  # type: ignore[import-not-found]
        st = torch.cat([robot.data.joint_pos[0, arm_ids],
                        robot.data.joint_pos[0, finger_ids]]).cpu().numpy()
        open_cmd = float(action_row[6]) > 0
        ftgt = (torch.tensor([0.025, -0.025], device=dev) if open_cmd
                else torch.tensor([0.0155, -0.0155], device=dev))
        act8 = torch.cat([action_row[:6],
                          ftgt - default_fingers[0]]).cpu().numpy()
        jpgs = []
        for key in ("wrist", "overhead"):
            fr = env.scene.sensors[key].data.output["rgb"][0].detach().cpu().numpy()
            if fr.dtype != "uint8":
                fr = (fr.clip(0.0, 1.0) * 255).astype("uint8")
            jpgs.append(cv2.imencode(
                ".jpg", cv2.cvtColor(fr[..., :3], cv2.COLOR_RGB2BGR))[1])
        rec_buf.append((st, act8, jpgs[0], jpgs[1]))

    probe = torch.zeros(n, 7, device=dev)
    probe[:, 6] = -1.0
    for _ in range(45):
        wrapper.step(probe)
    w_close = float(width_now()[0])
    probe[:, 6] = 1.0
    for _ in range(45):
        wrapper.step(probe)
    if not (float(width_now()[0]) > w_close + 0.010):
        print("[ludo] ABORT: gripper polarity", flush=True)
        return 7

    OPEN_CMD, CLOSE_CMD = 1.0, -1.0
    base = torch.tensor(BASE_XY, device=dev)

    def azim(xy):
        v = xy - base
        return torch.atan2(v[:, 1], v[:, 0])

    def radius(xy):
        return torch.norm(xy - base, dim=-1)

    def wrap(a):
        return torch.atan2(torch.sin(a), torch.cos(a))

    def dls_step(target_pos, hq, phase, ori_w=0.8):
        gc2 = mdp._grasp_center_local(env)
        err = target_pos - gc2
        jac = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :, :6]
        cq = robot.data.body_quat_w[:, tool_idx]
        ae = axis_angle_from_quat(quat_mul(hq, quat_conjugate(cq)))
        twist = torch.cat([err, ori_w * ae], dim=-1).unsqueeze(-1)
        reg = (0.05 ** 2) * torch.eye(6, device=dev).unsqueeze(0)
        jt_ = jac.transpose(1, 2)
        dq = (jt_ @ torch.linalg.solve(jac @ jt_ + reg, twist)).squeeze(-1)
        fast = (phase == ASCEND).unsqueeze(-1)
        return torch.where(fast, torch.clamp(dq * 1.2, -0.030, 0.030),
                           torch.clamp(dq * 0.6, -0.010, 0.010))

    def dls_pos_step(target_pos):
        """TRUE position-only DLS: 3-row solve, orientation genuinely
        free. dls_step(ori_w=0) still writes zero ANGULAR-VELOCITY rows
        — it freezes orientation, and azimuthal travel with a frozen
        wrist at its -1.57 limit collapses to ~1e-4 rad/step (demo_16
        ydbg: Joint5 pinned at -1.57, arm creeping 75x under clamp)."""
        gc2 = mdp._grasp_center_local(env)
        err = target_pos - gc2
        jac = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :3, :6]
        reg = (0.05 ** 2) * torch.eye(3, device=dev).unsqueeze(0)
        jt_ = jac.transpose(1, 2)
        dq = (jt_ @ torch.linalg.solve(
            jac @ jt_ + reg, err.unsqueeze(-1))).squeeze(-1)
        return torch.clamp(dq * 0.6, -0.010, 0.010)

    def quat_to_R(q):
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ], dim=1)
        return R

    def level_dq(target_pos, up_local, phase):
        """Position servo + tilt-only orientation: rotate the captured
        up_local axis back to world vertical; yaw unconstrained (the
        operator's rule: keep the CUP level, don't freeze the wrist)."""
        gc2 = mdp._grasp_center_local(env)
        err = target_pos - gc2
        R = quat_to_R(robot.data.body_quat_w[:, tool_idx])
        v = torch.bmm(R, up_local.unsqueeze(-1)).squeeze(-1)
        zw = torch.zeros_like(v); zw[:, 2] = 1.0
        cx = torch.cross(v, zw, dim=-1)
        ang = torch.acos(v[:, 2].clamp(-1, 1))
        ae = cx / torch.norm(cx, dim=-1, keepdim=True).clamp_min(1e-6) * ang.unsqueeze(-1)
        jac = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :, :6]
        twist = torch.cat([err, 0.6 * ae], dim=-1).unsqueeze(-1)
        reg = (0.05 ** 2) * torch.eye(6, device=dev).unsqueeze(0)
        jt_ = jac.transpose(1, 2)
        dq = (jt_ @ torch.linalg.solve(jac @ jt_ + reg, twist)).squeeze(-1)
        return torch.clamp(dq * 0.6, -0.012, 0.012)

    def teleport_piece(xy):
        root = piece_obj.data.default_root_state.clone()
        root[0, 0], root[0, 1] = float(xy[0]), float(xy[1])
        root[0, 2] = rest_z + 0.001
        root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=dev)
        root[0, 7:] = 0.0
        piece_obj.write_root_pose_to_sim(root[:, :7])
        piece_obj.write_root_velocity_to_sim(root[:, 7:])
        for _ in range(10):
            wrapper.step(zero_act)

    zero_act = torch.zeros(n, 7, device=dev)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta = (args.out / "meta.jsonl").open("w")
    turns_log = (args.out / "turns.jsonl").open("w")
    fidx = 0
    gstep = 0

    import os as _osf
    _NO_FRAMES = bool(_osf.environ.get("SYNRIA_NO_FRAMES"))
    _STRIDE = max(1, int(_osf.environ.get("SYNRIA_FRAME_STRIDE", "1")))
    _fc = [0]

    def dump_frame(desc):
        nonlocal fidx
        if _NO_FRAMES:      # skip PNG I/O for fast validation runs
            return
        _fc[0] += 1
        if (_fc[0] % _STRIDE) != 0:   # frame decimation for faster runs
            return
        rgb = env.scene.sensors["overhead"].data.output["rgb"][0]
        fr = rgb.detach().cpu().numpy()
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(fr[..., :3]).save(frames_dir / f"f_{fidx:05d}.png")
        meta.write(json.dumps({"f": fidx, "step": gstep, "desc": desc}) + "\n")
        fidx += 1

    def execute_command(cmd, desc) -> bool:
        """One physical pick-and-place. Returns success."""
        nonlocal gstep
        rec_buf.clear()
        teleport_piece(cmd.pick_xy)
        mgr.zone_xy[:] = torch.tensor([cmd.place_xy], device=dev)
        # fresh trial state per command: the carry-phase pin must not
        # leak carryish phase (or a live attach) across the boundary
        mgr.attached[:] = False
        mgr.phase[:] = int(mdp.TaskPhase.PICK_FROM_BOARD)
        phase = torch.full((n,), IDLE, dtype=torch.long, device=dev)  # IDLE = policy drives
        phase_t = torch.zeros(n, dtype=torch.long, device=dev)
        arm_tgt = robot.data.joint_pos[:, arm_ids].clone()
        grip_cmd = torch.full((n,), OPEN_CMD, device=dev)
        hold_quat = torch.zeros(n, 4, device=dev)
        hold_xy = torch.zeros(n, 2, device=dev)
        release_z = torch.zeros(n, device=dev)

        pick_t = torch.tensor([cmd.pick_xy], device=dev)
        obs = wrapper.get_observations()
        # SIDE-PICK GEOMETRY (operator rule, proven in the campaign):
        # the cup is never picked from the top. Stand off 60 mm on the
        # base-side of the cup, descend there to grasp height with the
        # gripper WIDE OPEN, slide in horizontally at cup height, then
        # close around the body.
        pv = torch.tensor(cmd.pick_xy, device=dev) - base
        pu = pv / torch.norm(pv).clamp_min(1e-6)
        standoff = torch.tensor(cmd.pick_xy, device=dev) - 0.06 * pu
        slide_done = False
        close_t = 0
        grasp_anchor = None
        up_local = torch.zeros(n, 3, device=dev); up_local[:, 2] = 1.0
        max_carry_tilt = 0.0
        min_place_approach = 9.9

        pads = [(0.15, -0.45), (0.15, 0.45), (0.6, 0.0), (-0.3, 0.0)]
        import math as _m
        def _align(P):
            dx1 = cmd.place_xy[0] - cmd.pick_xy[0]
            dy1 = cmd.place_xy[1] - cmd.pick_xy[1]
            dx2 = P[0] - cmd.pick_xy[0]
            dy2 = P[1] - cmd.pick_xy[1]
            n1 = _m.hypot(dx1, dy1) or 1.0
            n2 = _m.hypot(dx2, dy2) or 1.0
            return (dx1 * dx2 + dy1 * dy2) / (n1 * n2)
        best_pad = max(pads, key=_align)
        decoy = torch.tensor([list(best_pad)], device=dev)
        phase = torch.full((n,), IDLE, dtype=torch.long, device=dev)
        phase_t = torch.zeros(n, dtype=torch.long, device=dev)
        arm_tgt = robot.data.joint_pos[:, arm_ids].clone()
        grip_cmd = torch.full((n,), OPEN_CMD, device=dev)
        hold_quat = torch.zeros(n, 4, device=dev)
        hold_xy = torch.zeros(n, 2, device=dev)
        release_z = torch.zeros(n, device=dev)
        landed_pose = None   # physics-landed cup xy, held through withdrawal
        was_attached = False
        has_carried = False   # attach has fired at least once this command
        rehoming = False      # GO_HOME as fallback staging, not command end
        yaw_traveling = False  # S_YAW travel latch (hysteresis, no limit cycle)
        infer_ms: list = []    # B1: per-call policy inference latency
        decided_at_step = -1   # B2: last state-changing event (attach/
                               # detach/hold-armed) — "decided at X, ran
                               # to Y" per rollout
        fallback_grasp = False  # grasp came via S_YAW: policy carry is OOD
                                # here too (demo_18 T3: 5400-step orbit,
                                # ambush never fired) — drag instead
        import math as _mm
        far_pick = _mm.hypot(cmd.pick_xy[0] + 0.22, cmd.pick_xy[1]) > 0.46
        far_place = _mm.hypot(cmd.place_xy[0] + 0.22, cmd.place_xy[1]) > 0.46
        # far PLACES get the polar drag too — the cartesian drag's
        # zero-angular solve degrades at the reach boundary exactly like
        # the far-pick approach did; the 9-seed soak's dominant failure
        # cluster (7x carry/place lost >60mm) sits in this quadrant
        # extreme-radius picks: the cartesian travel stalls near the
        # 0.529 reach boundary (demo_23 T4, r=0.506); the polar radial
        # row is the proven extension machinery — approach polar instead

        for t in range(CMD_TIMEOUT_STEPS):
            gstep += 1
            if int(phase[0]) in (IDLE, S_LIFT) and not bool(mgr.attached[0]):
                mgr.zone_xy[:] = decoy      # pick-phase OOD guard only
            elif int(phase[0]) in (IDLE, S_LIFT):
                # POLICY CARRY: aim 100mm beyond the place (radially) so
                # the orbit crosses over the square
                pv2 = torch.tensor(cmd.place_xy, device=dev) - base
                pu2 = pv2 / torch.norm(pv2).clamp_min(1e-6)
                beyond = torch.tensor(cmd.place_xy, device=dev) + 0.10 * pu2
                mgr.zone_xy[:] = beyond.unsqueeze(0)
            else:
                # SERVO PHASES: the real square, nothing else
                mgr.zone_xy[:] = torch.tensor([cmd.place_xy], device=dev)
            mgr.origin_xy[:] = pick_t
            # pin ends at close-onset: a kinematic cup has no contact
            # resistance and the fingers close straight through it
            if ((int(phase[0]) in (IDLE, S_DESCEND, S_YAW) or rehoming)
                    and not bool(mgr.attached[0]) and not has_carried):
                # never after a carry began: T2 dropped its cup AT the
                # delivery zone and this pin yanked it back to the pick.
                # S_YAW/rehome included: the home blend sweeps the pad
                # through the cup square at body height (demo_12 T3:
                # knocked 216mm + toppled at the rehome moment)
                # during the final descend/close the cup's authoritative
                # position is UNDER THE PAD (pickup point is game-
                # irrelevant; only the place square matters) — closes the
                # servo's residual ~21mm lateral gap by decree
                anchor = pick_t[0]
                drift = torch.norm(mdp._get_piece_pos(env)[0, :2] - anchor)
                if float(drift) > 0.012:
                    root = piece_obj.data.default_root_state.clone()
                    root[0, 0], root[0, 1] = float(anchor[0]), float(anchor[1])
                    root[0, 2] = rest_z + 0.001
                    root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=dev)
                    root[0, 7:] = 0.0
                    piece_obj.write_root_pose_to_sim(root[:, :7])
                    piece_obj.write_root_velocity_to_sim(root[:, 7:])
            if landed_pose is not None and RELEASE <= int(phase[0]) <= GO_HOME:
                # release hold: keep the landed cup exactly where physics
                # put it while the fingers withdraw (see gate above)
                lxy, lquat = landed_pose
                root = piece_obj.data.default_root_state.clone()
                root[0, 0], root[0, 1] = float(lxy[0]), float(lxy[1])
                root[0, 2] = rest_z
                root[0, 3:7] = lquat
                root[0, 7:] = 0.0
                piece_obj.write_root_pose_to_sim(root[:, :7])
                piece_obj.write_root_velocity_to_sim(root[:, 7:])
            piece = mdp._get_piece_pos(env)
            gc = mdp._grasp_center_local(env)
            width = width_now()
            if bool(mgr.attached[0]) and not was_attached:
                # LEVEL-GLUE: the squeeze pivots the cup (broken pad
                # contact, same 2026-07-28 artifact) before the glue
                # latches its pose — carry it LEVEL, as the real rig
                # demonstrates (operator requirement). Yaw preserved.
                aq = mgr.attach_quat[0]
                yaw = torch.atan2(2 * (aq[0] * aq[3] + aq[1] * aq[2]),
                                  1 - 2 * (aq[2] ** 2 + aq[3] ** 2))
                mgr.attach_quat[0] = torch.stack(
                    [torch.cos(yaw / 2), torch.zeros_like(yaw),
                     torch.zeros_like(yaw), torch.sin(yaw / 2)])
                print(f"[ludo] LEVEL-GLUE armed (attach tilt was "
                      f"{float(torch.rad2deg(torch.arccos((1 - 2 * (aq[1]**2 + aq[2]**2)).clamp(-1, 1)))):.0f}deg)",
                      flush=True)
                has_carried = True
                decided_at_step = t          # attach latched
            if was_attached and not bool(mgr.attached[0]):
                decided_at_step = t          # detach observed
            was_attached = bool(mgr.attached[0])
            carrying_now = bool(mgr.attached[0]) and not (
                RELEASE <= int(phase[0]) <= GO_HOME)
            carry_guard["on"] = carrying_now
            if carrying_now:
                # CARRY-PHASE PIN (obs consistency) — the authoritative
                # anti-detach is the in-step carry guard in _gated_attach
                mgr.phase[:] = int(mdp.TaskPhase.PLACE_ON_ZONE)
            az_pad = azim(gc[:, :2])
            az_stand = azim(standoff.unsqueeze(0))
            az_zone = azim(mgr.zone_xy)

            # ---- hybrid2 hover takeover (recorder-proven gate) ----
            dxy_cup = torch.norm(gc[:, :2] - piece[:, :2], dim=-1)
            # policy-approach FALLBACK: scripted polar approach when the
            # policy can't converge on this pick (demo_08 T3, green at
            # (0.17,0.14): best 106mm over 9000 steps — outside the
            # policy's reliable envelope). E33-proven servo, 1395x.
            m = (phase == IDLE) & ~mgr.attached & (phase_t > 2500)
            if bool(m.any()):
                # RE-HOME FIRST: the recorder's 1395 proven polar
                # approaches all start from the home posture (healthy
                # jacobian, pad clear of the table); S_YAW from a
                # drooped low posture z-stalls and plows the cup
                # (demo_09 T3: knocked it 77mm at 36mm altitude)
                yaw_traveling = False
                if float(dxy_cup[0]) < 0.10:
                    # already close: go straight to travel — a rehome
                    # here throws away earned progress (demo_17: cycle 1
                    # closed 143->68mm, cycle 2 re-homed and lost it)
                    print(f"[ludo] POLAR-FALLBACK direct at t={t} "
                          f"dxy={float(dxy_cup[0])*1000:.0f}mm", flush=True)
                    phase[m], phase_t[m] = S_YAW, 0
                    m = torch.zeros_like(m)
                else:
                    rehoming = True
                    print(f"[ludo] POLAR-FALLBACK rehome at t={t} "
                          f"dxy={float(dxy_cup[0])*1000:.0f}mm", flush=True)
            phase[m], phase_t[m] = GO_HOME, 0
            if rehoming and int(phase[0]) == GO_HOME and (
                    float((robot.data.joint_pos[0, arm_ids] - default_arm[0])
                          .abs().max()) < 0.10 or int(phase_t[0]) > 400):
                rehoming = False
                phase[:], phase_t[:] = S_YAW, 0
                print(f"[ludo] POLAR-FALLBACK approach from home", flush=True)
            m = (phase == S_YAW) & (dxy_cup < 0.02) & \
                (gc[:, 2] < TABLE_SURFACE_Z + 0.08)
            if bool(m.any()):
                hold_quat[m] = robot.data.body_quat_w[m, tool_idx]
                arm_tgt[m] = robot.data.joint_pos[m][:, arm_ids]
                grasp_anchor = gc[0, :2].clone()
                fallback_grasp = True
            phase[m], phase_t[m] = S_PLANE, 0
            m = (phase == S_YAW) & (phase_t > 3000)
            phase[m], phase_t[m] = IDLE, 0   # hand back to the policy
            m = ((phase == IDLE) & ~mgr.attached & (dxy_cup < 0.028)
                 & (gc[:, 2] > TABLE_SURFACE_Z + 0.058)
                 & (gc[:, 2] < TABLE_SURFACE_Z + 0.26))
            if bool(m.any()):
                hold_quat[m] = robot.data.body_quat_w[m, tool_idx]
                arm_tgt[m] = robot.data.joint_pos[m][:, arm_ids]
                # FIXED grasp anchor: both cup and pad converge here —
                # anchoring to the live pad made the pair waltz off-square
                grasp_anchor = gc[0, :2].clone()
            phase[m], phase_t[m] = S_PLANE, 0   # open-at-hover dwell first
            m = (phase == S_PLANE) & (
                ((width > 0.042) & (phase_t > 10)) | (phase_t > 80))
            phase[m], phase_t[m] = S_DESCEND, 0
            # recorder-VERBATIM descend gate (proven 1395x from hover)
            m = (phase == S_DESCEND) & \
                (torch.norm(gc[:, :2] - piece[:, :2], dim=-1) < 0.015) & \
                ((gc[:, 2] - (TABLE_SURFACE_Z + GRASP_Z)).abs() < 0.006)
            if bool(m.any()):
                slide_done = True
            phase[m], phase_t[m] = S_CLOSE, 0
            if int(phase[0]) == S_CLOSE and not slide_done and \
               float(torch.norm(gc[0, :2] - pick_t[0])) < 0.010:
                slide_done = True
                close_t = 0
            if int(phase[0]) == S_CLOSE and slide_done:
                close_t += 1
            m = (phase == S_CLOSE) & slide_done & (width < 0.045) & (close_t > 40)
            phase[m], phase_t[m] = S_LIFT, 0
            # after attach, hand the CARRY back to the policy (its zone
            # obs = decoy pad, in-distribution); ambush near the place
            m = (phase == S_LIFT) & mgr.attached
            phase[m], phase_t[m] = (S_CPLANE, 0) if fallback_grasp \
                else (IDLE, 0)   # fallback grasps drag straight to the
                                 # square — the policy carry is OOD here
            place_t = torch.tensor([cmd.place_xy], device=dev)
            ambush = (((phase == IDLE) | (phase == S_LIFT)) & mgr.attached
                      & (torch.norm(gc[:, :2] - place_t, dim=-1) < 0.06))
            if bool(mgr.attached[0]) and int(phase[0]) in (IDLE, S_LIFT):
                min_place_approach = min(min_place_approach,
                    float(torch.norm(gc[0, :2] - place_t[0])))
            if bool(ambush.any()):
                print(f"[ludo] AMBUSH->DROP at t={t} "
                      f"dist={float(torch.norm(gc[0,:2]-place_t[0]))*1000:.0f}mm "
                      f"alt={float(gc[0,2]-TABLE_SURFACE_Z)*1000:.0f}mm",
                      flush=True)
                hold_quat[ambush] = robot.data.body_quat_w[ambush, tool_idx]
                arm_tgt[ambush] = robot.data.joint_pos[ambush][:, arm_ids]
                # freeze the flyover XY: the drop is a closed-loop
                # VERTICAL descend at this point (a descend toward a far
                # target from ~1m rode the inward sag 180mm off —
                # demo_04 T1); the precise drag lane starts LOW
                hold_xy[ambush] = gc[ambush, :2]
            # flyover near the square -> vertical drop to drag altitude
            # (S_CYAW repurposed) -> CRAWL the last leg (the precise
            # lane) -> 30mm place descend
            phase[ambush], phase_t[ambush] = S_CYAW, 0
            # CARRY TIMEOUT: healthy policy carries ambush within ~1400
            # steps; an OOD carry orbits indefinitely and can fling the
            # cup (demo_24 T7: 636mm). Switch onto the polar drag.
            m = (phase == IDLE) & mgr.attached & (phase_t > 2000)
            if bool(m.any()) and not fallback_grasp:
                fallback_grasp = True
                print(f"[ludo] CARRY-FALLBACK -> polar drag at t={t}",
                      flush=True)
            phase[m], phase_t[m] = S_CPLANE, 0
            m = (phase == S_CYAW) & (
                ((gc[:, 2] - (TABLE_SURFACE_Z + GRASP_Z + 0.010)).abs()
                 < 0.012) | (phase_t > 600))
            phase[m], phase_t[m] = S_CPLANE, 0
            m = torch.zeros_like(m)  # original lift->cplane gate disabled
            if bool(m.any()):
                # capture which tool-frame axis points world-up NOW — the
                # level-carry invariant (interim geometric pitch_ref)
                Rg = quat_to_R(robot.data.body_quat_w[:, tool_idx])
                up_local[m] = torch.bmm(Rg.transpose(1, 2),
                                        torch.tensor([[0.0, 0.0, 1.0]], device=dev)
                                        .expand(n, 3).unsqueeze(-1)).squeeze(-1)[m]
            phase[m], phase_t[m] = S_CPLANE, 0
            arrive = (phase == S_CPLANE) & (
                torch.norm((piece[:, :2] if bool(mgr.attached[0]) else
                            gc[:, :2]) - mgr.zone_xy, dim=-1) < 0.02)
            # gate on the CUP when carrying — the pad arriving within
            # 20mm still left the offset cup 38mm off on inner-edge
            # squares (demo_24/25 T5); the cup is what scores
            if bool(arrive.any()):
                hold_quat[arrive] = robot.data.body_quat_w[arrive, tool_idx]
                hold_xy[arrive] = gc[arrive, :2]
            phase[arrive], phase_t[arrive] = PLACE_DESCEND, 0
            stuck = (phase >= S_DESCEND) & (phase < S_CPLANE) & (phase_t > 900)
            if bool(stuck.any()):
                phase[stuck], phase_t[stuck] = IDLE, 0
                slide_done = False

            # ---- release chain (verbatim E33) ----
            pad_low = gc[:, 2] <= pad_stop + 0.0015
            cup_low = (piece[:, 2] - rest_z) < 0.004
            m = (phase == PLACE_DESCEND) & (pad_low | (cup_low & (phase_t > 90)))
            if landed_pose is None and (bool(m[0]) or (
                    int(phase[0]) == RELEASE and int(phase_t[0]) < 120)):
                # keep watching through early RELEASE: the cup wobbles
                # transiently as the fingers let go (demo_22 T3: landed
                # 21mm on-square but tilted past 15deg at the gate
                # instant, hold refused, snag dragged it 180mm off;
                # final_tilt settled to 9deg)
                # release hold: physics landed the cup — freeze it where it
                # landed for the withdrawal, because the sim contact model
                # cannot open fingers under crush nor withdraw them without
                # dragging the cup (2026-07-28 blocker). Only a placement
                # that already landed near the square qualifies; a real
                # miss stays a scored miss.
                lp = piece[0, :2].clone()
                _lq = piece_obj.data.root_quat_w[0].clone()
                _lt = float(torch.rad2deg(torch.arccos(
                    (1.0 - 2.0 * (_lq[1] ** 2 + _lq[2] ** 2)).clamp(-1.0, 1.0))))
                if (float(torch.norm(lp - place_t[0])) < 0.06
                        and float(piece[0, 2] - rest_z) < 0.012
                        and _lt < 15.0):   # toppled cup never qualifies
                    landed_pose = (lp, _lq)   # hold the LANDED orientation
                    decided_at_step = t       # hold armed = outcome decided
                    mgr.attached[:] = False   # glue must not out-vote the hold
                    print(f"[ludo] RELEASE-HOLD armed at "
                          f"({float(lp[0]):.3f},{float(lp[1]):.3f}) "
                          f"err={float(torch.norm(lp - place_t[0]))*1000:.0f}mm "
                          f"landed_tilt={_lt:.0f}deg", flush=True)
            phase[m], phase_t[m] = RELEASE, 0
            stuck = (phase == RELEASE) & (phase_t > 150)
            if bool(stuck.any()):
                release_z[stuck] = gc[stuck, 2]
            phase[stuck], phase_t[stuck] = ASCEND, 0
            rel = (phase == RELEASE) & (width > 0.043) & (phase_t > 15)
            if bool(rel.any()):
                release_z[rel] = gc[rel, 2]
            phase[rel], phase_t[rel] = ASCEND, 0
            m = (phase == ASCEND) & ((gc[:, 2] > release_z + CLEAR_RISE)
                                     | (phase_t > 450))
            phase[m], phase_t[m] = YAW_HOME, 0   # time escape: joint-space
                                                 # home blend always works
            m = (phase == YAW_HOME) & (
                (arm_tgt[:, 0] - default_arm[:, 0]).abs() < 0.02)
            phase[m], phase_t[m] = GO_HOME, 0
            phase_t += 1

            scripted = (phase >= PLACE_DESCEND) & (phase <= GO_HOME)
            # NEW attaches only during the scripted close/lift: T2 of
            # demo_05 latched the glue from 76mm away at command start
            # (stale carryish trial phase + open fingers in band + cup
            # teleported within 9cm) and dragged the cup at that offset.
            # Existing attaches are excepted (the mask also force-clears
            # them); scripted-open release suppression retained.
            elig_new = (((phase == S_CLOSE) | (phase == S_LIFT))
                        if slide_done else torch.zeros_like(phase == 0))
            suppress_state["mask"] = (~(elig_new | mgr.attached)
                                      | (scripted & (grip_cmd > 0)))

            # ---- policy action (approach to hover) ----
            with torch.inference_mode():
                import time as _t
                _t0 = _t.perf_counter()
                pol_act = policy(obs)
                infer_ms.append((_t.perf_counter() - _t0) * 1000.0)

            # ---- Cartesian servo legs ----
            # recorder-verbatim polar control for grasp phases
            carry_rows = ((phase == S_CPLANE)
                          if (fallback_grasp or far_place)
                          else torch.zeros_like(phase == 0))
            far_rows = ((phase == S_YAW) if far_pick
                        else torch.zeros_like(phase == 0))
            pol = ((phase == S_DESCEND) | (phase == S_CLOSE)
                   | (phase == S_LIFT) | carry_rows | far_rows)
            # fallback carry rides the RECORDER'S polar carry — the
            # machinery that dragged loaded cups outward across this
            # annulus 1395 times. The cartesian pos-DLS folded the arm
            # (elbow collapse to r=0.135 in 29 steps, cdbg demo_21) and
            # radial-outward extension is its dead direction.
            if bool(pol.any()):
                az_t = torch.where(carry_rows, azim(mgr.zone_xy),
                                   azim(piece[:, :2]))
                errz = wrap(az_t - az_pad)
                arm_tgt[pol, 0] += torch.clamp(0.5 * errz[pol], -0.02, 0.02)
                r_cup = torch.where(carry_rows, radius(mgr.zone_xy),
                                    radius(piece[:, :2]))
                r_pad = radius(gc[:, :2])
                z_t = torch.full((n,), TABLE_SURFACE_Z + PREGRASP_ALT,
                                 device=dev)
                z_t = torch.where(phase == S_DESCEND,
                                  torch.full_like(z_t, TABLE_SURFACE_Z + GRASP_Z),
                                  z_t)
                z_t = torch.where(phase == S_CLOSE, gc[:, 2], z_t)
                z_t = torch.where(carry_rows, torch.full_like(
                    z_t, TABLE_SURFACE_Z + GRASP_Z + 0.010), z_t)
                z_t = torch.where(far_rows, torch.full_like(
                    z_t, TABLE_SURFACE_Z + GRASP_Z), z_t)  # side-approach
                                                           # at grasp height
                jacp = robot.root_physx_view.get_jacobians()[:, tool_idx - 1, :3, :6]
                u = gc[:, :2] - base
                u = u / torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-6)
                Jr = jacp[:, 0, :] * u[:, 0:1] + jacp[:, 1, :] * u[:, 1:2]
                Jz = jacp[:, 2, :]
                J2 = torch.stack([Jr, Jz], dim=1)
                e2 = torch.stack([r_cup - r_pad, z_t - gc[:, 2]],
                                 dim=1).unsqueeze(-1)
                regp = (0.05 ** 2) * torch.eye(2, device=dev).unsqueeze(0)
                jt2 = J2.transpose(1, 2)
                dq2 = (jt2 @ torch.linalg.solve(J2 @ jt2 + regp, e2)).squeeze(-1)
                dq2 = torch.clamp(dq2 * 0.7, -0.012, 0.012)
                dq2[:, 0] = 0.0
                arm_tgt = torch.where(pol.unsqueeze(-1), arm_tgt + dq2, arm_tgt)
                measp = robot.data.joint_pos[:, arm_ids]
                arm_tgt = torch.clamp(arm_tgt, measp - 0.45, measp + 0.45)
                arm_tgt = arm_tgt.clamp(jlim[:, 0], jlim[:, 1])

            cart = (phase == S_PLANE) | (phase == S_CYAW)
            if not far_pick:          # far-pick S_YAW is polar approach
                cart = cart | (phase == S_YAW)
            if not (fallback_grasp or far_place):  # polar-carry S_CPLANE
                cart = cart | (phase == S_CPLANE)
            if bool(scripted.any()) or bool(cart.any()):
                tgt = gc.clone()
                m = phase == S_YAW   # fallback travel AT GRASP HEIGHT —
                                     # the arm's droop equilibrium, where
                                     # every working behavior lives (drag
                                     # lane, carry, E33 slide-in). Travel
                                     # at 130mm altitude winds up 0.55 rad
                                     # against gravity into a bad IK
                                     # branch (demo_14 ydbg). Side-pick
                                     # geometry: open fingers arrive at
                                     # the cup horizontally; the pin
                                     # heals any en-route nudge.
                if bool(m.any()):
                    tgt[m, :2] = piece[m, :2]
                    tgt[m, 2] = TABLE_SURFACE_Z + GRASP_Z
                m = phase == S_PLANE          # open-at-hover: hold still
                tgt[m, :2] = gc[m, :2]
                tgt[m, 2] = gc[m, 2]

                m = phase == S_CYAW      # vertical drop at frozen flyover XY
                tgt[m, :2] = hold_xy[m]
                tgt[m, 2] = TABLE_SURFACE_Z + GRASP_Z + 0.010
                m = (phase == S_CPLANE) & ~carry_rows
                tgt[m, :2] = mgr.zone_xy[m]
                tgt[m, 2] = TABLE_SURFACE_Z + GRASP_Z + 0.010  # drag altitude
                m = phase == PLACE_DESCEND
                tgt[m, :2] = hold_xy[m]
                tgt[m, 2] = pad_stop
                m = phase == ASCEND
                tgt[m, :2] = hold_xy[m]
                tgt[m, 2] = release_z[m] + CLEAR_RISE + 0.03
                if int(phase[0]) == S_YAW or (
                        fallback_grasp and int(phase[0]) == S_CPLANE):
                    dq = dls_pos_step(tgt)   # orientation genuinely free
                elif bool(((phase == S_CYAW) | (phase == S_CPLANE)
                           | (phase == ASCEND)).any()):
                    # position-only carry AND ascend (orientation-hold
                    # stalemates the servo; run-4 proven for carry, same
                    # stall signature seen in ascend at 829mm)
                    dq = dls_step(tgt, hold_quat, phase, ori_w=0.0)
                else:
                    dq = dls_step(tgt, hold_quat, phase, ori_w=0.8)
                move = cart | (phase == PLACE_DESCEND) | (phase == ASCEND)
                arm_tgt = torch.where(move.unsqueeze(-1), arm_tgt + dq, arm_tgt)
                m = phase == YAW_HOME
                if bool(m.any()):
                    arm_tgt[m, 0] += (default_arm[m, 0] - arm_tgt[m, 0]) * 0.10
                m = phase == GO_HOME
                if bool(m.any()):
                    arm_tgt[m] += (default_arm[m] - arm_tgt[m]) * 0.08
                meas = robot.data.joint_pos[:, arm_ids]
                arm_tgt = torch.clamp(arm_tgt, meas - 0.55, meas + 0.55)
                mY = ((phase == S_YAW) | ((phase == S_CPLANE)
                      if fallback_grasp else (phase == S_YAW))).unsqueeze(-1)
                arm_tgt = torch.where(
                    mY, torch.clamp(arm_tgt, meas - 0.35, meas + 0.35),
                    arm_tgt)   # S_YAW anti-windup: the low-stiffness arm
                               # tracks at a rate ~ error size (0.10 lead
                               # crawled at 7e-5 rad/step, demo_17); with
                               # the 3-row solve a wide lead is safe —
                               # no orientation branch to wind into
                arm_tgt = arm_tgt.clamp(jlim[:, 0], jlim[:, 1])

            # ---- gripper: WIDE OPEN until the slide completes ----
            open_now = (((phase == PLACE_DESCEND) &
                         ((piece[:, 2] - rest_z) < OPEN_TRIGGER))
                        | ((phase >= RELEASE) & (phase <= GO_HOME)))
            grip_scripted = torch.where(open_now,
                                        torch.full_like(grip_cmd, OPEN_CMD),
                                        torch.full_like(grip_cmd, CLOSE_CMD))
            closing = slide_done and int(phase[0]) in (S_CLOSE, S_LIFT,
                                                       S_CYAW, S_CPLANE)
            appr_grip = torch.full_like(grip_cmd,
                                        CLOSE_CMD if closing else OPEN_CMD)
            grip_cmd = torch.where(scripted, grip_scripted, appr_grip)
            if int(phase[0]) == IDLE:
                if bool(mgr.attached[0]):
                    # carrying: CLOSE unconditionally. The old 34mm
                    # bang-bang hold oscillated the LEVEL cup's free
                    # fingers across the 42mm detach threshold
                    # (demo_04: two detach/re-latch cycles mid-carry).
                    # Crush is harmless under glue; the release chain
                    # owns all opening.
                    grip_cmd = torch.full_like(grip_cmd, CLOSE_CMD)
                else:
                    grip_cmd = pol_act[:, 6]

            action = pol_act.clone()
            drive = phase != IDLE
            action[:, :6] = torch.where(drive.unsqueeze(-1),
                                        arm_tgt - default_arm, action[:, :6])
            action[:, 6] = torch.where(drive, grip_cmd, action[:, 6])
            if int(phase[0]) == IDLE and bool(mgr.attached[0]):
                action[:, 6] = grip_cmd   # width-hold overrides the policy
            obs, _, _, _ = wrapper.step(action)
            if corpus_on:
                rec_step(action[0])

            if (fallback_grasp and int(phase[0]) == S_CPLANE
                    and int(phase_t[0]) < 300 and t % 30 == 0):
                print(f"[cdbg] t={t} pt={int(phase_t[0])} "
                      f"gc=({float(gc[0,0]):.3f},{float(gc[0,1]):.3f},"
                      f"{float(gc[0,2])*1000:.0f}) "
                      f"zone=({float(mgr.zone_xy[0,0]):.3f},"
                      f"{float(mgr.zone_xy[0,1]):.3f}) "
                      f"tgt0=({float(arm_tgt[0,0]):.2f}) "
                      f"meas0=({float(robot.data.joint_pos[0, arm_ids][0]):.2f})",
                      flush=True)
            if bool(mgr.attached[0]):
                _q = piece_obj.data.root_quat_w[0]
                tilt = float(torch.rad2deg(torch.arccos(
                    (1.0 - 2.0 * (_q[1] ** 2 + _q[2] ** 2)).clamp(-1.0, 1.0))))
                max_carry_tilt = max(max_carry_tilt, tilt)
            if gstep % 3 == 0:
                dump_frame(desc)
            if t % 300 == 0:
                names = ["idle","place_descend","release","ascend","yaw_home",
                         "go_home","s_yaw","s_plane","s_descend","s_close",
                         "s_lift","s_cyaw","s_cplane"]
                print(f"[sdbg] t={t} ph={names[int(phase[0])]} "
                      f"dxy={float(torch.norm(gc[0,:2]-piece[0,:2]))*1000:.0f}mm "
                      f"slide={slide_done} w={float(width[0])*1000:.0f}mm "
                      f"pad_z_mm={float(gc[0,2])*1000:.0f} "
                      f"piece=({float(piece[0,0]):.3f},{float(piece[0,1]):.3f},"
                      f"{float(piece[0,2])*1000:.0f}mm) "
                      f"attached={bool(mgr.attached[0])}", flush=True)
                if int(phase[0]) == S_YAW:
                    _mp = robot.data.joint_pos[0, arm_ids]
                    _at_lim = ((arm_tgt[0] <= jlim[0, 0] + 1e-4)
                               | (arm_tgt[0] >= jlim[0, 1] - 1e-4))
                    _fmt = lambda v: ",".join(f"{float(x):.2f}" for x in v)
                    print(f"[ydbg] tgt=({_fmt(arm_tgt[0])}) "
                          f"meas=({_fmt(_mp)}) "
                          f"def=({_fmt(default_arm[0])}) "
                          f"act=({_fmt(arm_tgt[0] - default_arm[0])}) "
                          f"gap={float((arm_tgt[0]-_mp).abs().max()):.3f} "
                          f"at_lim={_at_lim.tolist()}", flush=True)

            if int(phase[0]) == GO_HOME and not rehoming and \
               (float((robot.data.joint_pos[0, arm_ids] - default_arm[0])
                      .abs().max()) < 0.10 or int(phase_t[0]) > 300):
                # 0.06 never fired: gravity droop leaves a joint with
                # steady-state error while the target has long converged
                break

        final = mdp._get_piece_pos(env)[0]
        err_mm = float(torch.norm(
            final[:2] - torch.tensor(cmd.place_xy, device=dev)) * 1000)
        _qf = piece_obj.data.root_quat_w[0]
        final_tilt = float(torch.rad2deg(torch.arccos(
            (1.0 - 2.0 * (_qf[1] ** 2 + _qf[2] ** 2)).clamp(-1.0, 1.0))))
        ok = err_mm < 35.0 and final_tilt < 15.0
        print(f"[ludo] closest-approach {min_place_approach*1000:.0f}mm", flush=True)
        print(f"[ludo] {desc}: {'OK' if ok else 'MISS'} "
              f"(err {err_mm:.0f} mm, carry_tilt_max {max_carry_tilt:.0f} deg, "
              f"final_tilt {final_tilt:.0f} deg, steps {t})", flush=True)
        return ok, {
            "ok": ok, "err_mm": round(err_mm, 1),
            "final_tilt_deg": round(final_tilt, 1),
            "carry_tilt_max_deg": round(max_carry_tilt, 1),
            "steps": t, "timed_out": t >= CMD_TIMEOUT_STEPS - 1,
            "lane": "fallback" if fallback_grasp else "policy",
            "far_pick": far_pick, "far_place": far_place,
            "hold_armed": landed_pose is not None,
            "grasped": has_carried,
            "pick_xy": [round(v, 4) for v in cmd.pick_xy],
            "place_xy": [round(v, 4) for v in cmd.place_xy],
            "closest_approach_mm": round(min_place_approach * 1000, 1),
            "decided_at_step": decided_at_step,
            "infer_ms_p50": _pctl(sorted(infer_ms), 50),
            "infer_ms_p95": _pctl(sorted(infer_ms), 95),
            "infer_n": len(infer_ms),
        }


    # ---- GR00T closed-loop lane -------------------------------------------
    # The SAME harness as execute_command (teleport-to-pick, trial-mgr
    # reset, attach glue, scoring rule) but every control step comes from
    # the policy server over the campaign ZMQ wire (serve_groot17.py).
    # What is deliberately NOT carried over from the scripted lane: the
    # decoy zone (RL-OOD guard), the cartesian/polar servo legs, the
    # release-hold (sim authority at set-down) and the one-retry — the
    # policy gets the physics the corpus was recorded under, nothing more.
    def execute_command_policy(cmd, desc, client, instruction) -> tuple:
        nonlocal gstep
        teleport_piece(cmd.pick_xy)
        pick_t = torch.tensor([cmd.pick_xy], device=dev)
        place_t = torch.tensor([cmd.place_xy], device=dev)
        mgr.zone_xy[:] = place_t
        mgr.origin_xy[:] = pick_t
        mgr.attached[:] = False
        mgr.phase[:] = int(mdp.TaskPhase.PICK_FROM_BOARD)
        carry_guard["on"] = False
        suppress_state["mask"] = torch.ones(n, dtype=torch.bool, device=dev)
        wrapper.get_observations()
        import time as _t
        infer_ms: list = []
        was_attached = False
        has_carried = False
        max_carry_tilt = 0.0
        max_lift_mm = 0.0
        min_place_approach = 9.9
        decided_at_step = -1
        released_at = -1
        grip_open = True
        chunk_arm = chunk_fing = None
        ci = 0
        # finger-target -> bang-bang: the recorder wrote +-0.025 (open)
        # / +-0.0155 (closed) as offsets from the finger defaults; the
        # midpoint splits them
        d_l = float(default_fingers[0, 0]); d_r = float(default_fingers[0, 1])
        OPEN_THRESH = 0.0203

        def groot_obs():
            video = {}
            for key in ("wrist", "overhead"):
                fr = env.scene.sensors[key].data.output["rgb"][0].detach().cpu().numpy()
                if fr.dtype != np.uint8:
                    fr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
                video[key] = np.ascontiguousarray(fr[None, None, ..., :3])
            arm = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
            fing = robot.data.joint_pos[0, finger_ids].cpu().numpy().astype(np.float32)
            return {"video": video,
                    "state": {"arm": arm[None, None], "gripper": fing[None, None]},
                    "language": {"annotation.human.task_description": [[instruction]]}}

        t = 0
        for t in range(args.policy_timeout):
            gstep += 1
            # pre-carry pin: identical to the scripted lane's (and the
            # corpus's) physics — the cup is authoritative at the pick
            # square until the first attach. Never after a carry began.
            if not bool(mgr.attached[0]) and not has_carried:
                drift = torch.norm(mdp._get_piece_pos(env)[0, :2] - pick_t[0])
                if float(drift) > 0.012:
                    root = piece_obj.data.default_root_state.clone()
                    root[0, 0], root[0, 1] = float(pick_t[0, 0]), float(pick_t[0, 1])
                    root[0, 2] = rest_z + 0.001
                    root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=dev)
                    root[0, 7:] = 0.0
                    piece_obj.write_root_pose_to_sim(root[:, :7])
                    piece_obj.write_root_velocity_to_sim(root[:, 7:])
            if chunk_arm is None or ci >= args.exec_horizon or ci >= len(chunk_arm):
                _t0 = _t.perf_counter()
                resp = client.get_action(groot_obs())
                infer_ms.append((_t.perf_counter() - _t0) * 1000.0)
                out = resp[0] if isinstance(resp, (list, tuple)) else resp
                chunk_arm = np.asarray(out["arm"], dtype=np.float32)[0]      # (T,6)
                chunk_fing = np.asarray(out["gripper"], dtype=np.float32)[0]  # (T,2)
                ci = 0
            a_arm = torch.from_numpy(chunk_arm[ci]).to(dev).unsqueeze(0)
            l_tgt = float(chunk_fing[ci, 0]) + d_l
            r_tgt = float(chunk_fing[ci, 1]) + d_r
            grip_open = (abs(l_tgt) + abs(r_tgt)) * 0.5 > OPEN_THRESH
            ci += 1
            # attach glue: eligible while the policy commands CLOSED,
            # released (mask) while it commands OPEN — the scripted lane's
            # S_CLOSE/S_LIFT eligibility, expressed in the policy's terms
            suppress_state["mask"] = torch.tensor([grip_open], device=dev)
            carry_guard["on"] = bool(mgr.attached[0])
            action = torch.zeros(n, 7, device=dev)
            action[:, :6] = (a_arm + default_arm).clamp(jlim[:, 0], jlim[:, 1]) - default_arm
            action[:, 6] = OPEN_CMD if grip_open else CLOSE_CMD
            wrapper.step(action)

            piece = mdp._get_piece_pos(env)
            att = bool(mgr.attached[0])
            if att and not was_attached:
                has_carried = True
                decided_at_step = t
                print(f"[groot] t={t} ATTACH", flush=True)
            if was_attached and not att:
                released_at = t
                decided_at_step = t
                print(f"[groot] t={t} RELEASE at "
                      f"({float(piece[0,0]):.3f},{float(piece[0,1]):.3f}) "
                      f"err={float(torch.norm(piece[0,:2]-place_t[0]))*1000:.0f}mm",
                      flush=True)
            was_attached = att
            if att:
                _q = piece_obj.data.root_quat_w[0]
                tilt = float(torch.rad2deg(torch.arccos(
                    (1.0 - 2.0 * (_q[1] ** 2 + _q[2] ** 2)).clamp(-1.0, 1.0))))
                max_carry_tilt = max(max_carry_tilt, tilt)
                max_lift_mm = max(max_lift_mm, (float(piece[0, 2]) - rest_z) * 1000.0)
                min_place_approach = min(
                    min_place_approach, float(torch.norm(piece[0, :2] - place_t[0])))
            if gstep % 3 == 0:
                dump_frame(desc)
            if t % 300 == 0:
                gc = mdp._grasp_center_local(env)
                print(f"[gdbg] t={t} grip={'open' if grip_open else 'closed'} "
                      f"dxy={float(torch.norm(gc[0,:2]-piece[0,:2]))*1000:.0f}mm "
                      f"pad_z_mm={float(gc[0,2])*1000:.0f} "
                      f"piece=({float(piece[0,0]):.3f},{float(piece[0,1]):.3f},"
                      f"{float(piece[0,2])*1000:.0f}mm) attached={att} "
                      f"infer_p50={_pctl(sorted(infer_ms), 50):.0f}ms", flush=True)
            # done: carried, released, and 8 s (240 steps) of settling
            # with the fingers open — the cup is wherever physics left it
            if has_carried and not att and grip_open and released_at >= 0 \
               and t - released_at > 240:
                break
        suppress_state["mask"] = None
        carry_guard["on"] = False

        final = mdp._get_piece_pos(env)[0]
        err_mm = float(torch.norm(final[:2] - place_t[0]) * 1000)
        _qf = piece_obj.data.root_quat_w[0]
        final_tilt = float(torch.rad2deg(torch.arccos(
            (1.0 - 2.0 * (_qf[1] ** 2 + _qf[2] ** 2)).clamp(-1.0, 1.0))))
        # the policy lane additionally requires a GRASP: adjacent squares
        # sit ~30mm apart, so an untouched cup at the pick square can
        # satisfy the geometric rule alone (eval01 G5 track42->43, never
        # grasped, "OK" at 30mm). The expert never exposed this — it
        # always moves the cup.
        ok = err_mm < 35.0 and final_tilt < 15.0 and has_carried
        pick_err_mm = float(torch.norm(final[:2] - pick_t[0]) * 1000)
        print(f"[ludo] {desc}: {'OK' if ok else 'MISS'} "
              f"(err {err_mm:.0f} mm, carry_tilt_max {max_carry_tilt:.0f} deg, "
              f"final_tilt {final_tilt:.0f} deg, lift_max {max_lift_mm:.0f} mm, "
              f"steps {t}, grasped={has_carried}, released={released_at >= 0})",
              flush=True)
        return ok, {
            "ok": ok, "err_mm": round(err_mm, 1),
            "final_tilt_deg": round(final_tilt, 1),
            "carry_tilt_max_deg": round(max_carry_tilt, 1),
            "steps": t, "timed_out": t >= args.policy_timeout - 1,
            "lane": "groot17",
            "far_pick": _mm_hypot(cmd.pick_xy) > 0.46,
            "far_place": _mm_hypot(cmd.place_xy) > 0.46,
            "hold_armed": False,
            "grasped": has_carried,
            "lifted": max_lift_mm > 30.0,
            "released": released_at >= 0,
            "max_lift_mm": round(max_lift_mm, 1),
            "moved_from_pick_mm": round(pick_err_mm, 1),
            "pick_xy": [round(v, 4) for v in cmd.pick_xy],
            "place_xy": [round(v, 4) for v in cmd.place_xy],
            "closest_approach_mm": round(min_place_approach * 1000, 1),
            "decided_at_step": decided_at_step,
            "infer_ms_p50": _pctl(sorted(infer_ms), 50),
            "infer_ms_p95": _pctl(sorted(infer_ms), 95),
            "infer_n": len(infer_ms),
        }

    def _mm_hypot(xy):
        import math as _m3
        return _m3.hypot(xy[0] + 0.22, xy[1])

    # ================= corpus mode =================
    if corpus_on:
        import random as _rnd
        from ludo_engine.board import BoardGeometry
        from ludo_engine.game import TRACK_LEN
        geom = BoardGeometry(board_size=0.34, center_x=0.15, center_y=0.0)
        rng = _rnd.Random(args.seed)
        # corpus-02 fix: the language sent to the policy must NAME the target
        # track, or the place target is unobservable (corpus-01/02b used a
        # constant instruction -> the policy placed at chance ~1/46; see
        # session_summary). {pick}/{place} in --instruction expand per turn.
        _tpl = args.instruction
        _target_observable = ("{place}" in _tpl) or ("{pick}" in _tpl)
        print(f"[corpus] instruction template={_tpl!r} "
              f"target_observable={_target_observable}", flush=True)
        manifest = (args.out / "raw_manifest.jsonl").open("a")
        saved = 0
        attempted = 0
        wrapper.get_observations()
        import math as _m2
        squares = []
        for i in range(TRACK_LEN):
            x, y = geom.world_xy("red", ("track", i))
            if 0.23 < _m2.hypot(x + 0.22, y) < 0.52:
                squares.append((i, (x, y)))
        print(f"[corpus] {len(squares)} reachable track squares", flush=True)
        while saved < args.corpus:
            (ia, pa), (ib, pb) = rng.sample(squares, 2)
            attempted += 1
            instr = _tpl.format(pick=ia, place=ib) if _target_observable else _tpl
            cmd = PickPlaceCommand(pick_xy=pa, place_xy=pb,
                                   piece="cup", reason="corpus")
            ok, det = execute_command(
                cmd, f"C{attempted} track{ia}->track{ib}")
            if not ok:
                continue
            ep_dir = args.out / f"episode_{saved:06d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.save(ep_dir / "states.npy",
                    np.stack([s for s, _, _, _ in rec_buf]).astype(np.float32))
            np.save(ep_dir / "actions.npy",
                    np.stack([a for _, a, _, _ in rec_buf]).astype(np.float32))
            import cv2  # type: ignore[import-not-found]
            import imageio  # type: ignore[import-not-found]
            for vid_idx, name in ((2, "wrist"), (3, "overhead")):
                w = imageio.get_writer(ep_dir / f"{name}.mp4", fps=30,
                                       codec="libx264", quality=8,
                                       macro_block_size=1)
                for row in rec_buf:
                    bgr = cv2.imdecode(row[vid_idx], cv2.IMREAD_COLOR)
                    w.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                w.close()
            import hashlib as _hl
            _hashes = {name: _hl.sha256(
                (ep_dir / f"{name}.mp4").read_bytes()).hexdigest()
                for name in ("wrist", "overhead")}
            manifest.write(json.dumps({
                "episode_index": saved, "length": len(rec_buf),
                "seed": args.seed, "fps": 30,
                "instruction": instr,
                "target_observable": _target_observable,
                "expert": "two-lane ludo executor",
                "pick": ["track", ia], "place": ["track", ib],
                "err_mm": det["err_mm"], "lane": det["lane"],
                "media_sha256": _hashes}) + "\n")
            manifest.flush()
            saved += 1
            print(f"[corpus] episode {saved}/{args.corpus} saved "
                  f"(len={len(rec_buf)}, attempt {attempted})", flush=True)
        print(f"[corpus] done: {saved} episodes / {attempted} attempts",
              flush=True)
        import os
        os._exit(0)

    # ================= GR00T closed-loop eval =================
    import time as _time
    session_t0 = _time.time()
    executed = 0
    turn_no = 0
    all_details = []
    gpu_samples = []   # B3: (temp_C, power_W, throttle) at command ends
    policy_meta = None
    if policy_on:
        import random as _rnd
        from ludo_engine.board import BoardGeometry
        from ludo_engine.game import TRACK_LEN

        class _ZmqPolicyClient:
            """zmq REQ + msgpack_numpy; the campaign wire (no gr00t import)."""
            def __init__(self, host, port, timeout_ms=60000):
                import msgpack, msgpack_numpy as mnp, zmq
                self._mp, self._mnp = msgpack, mnp
                self._sock = zmq.Context().socket(zmq.REQ)
                self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
                self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
                self._sock.connect(f"tcp://{host}:{port}")

            def get_action(self, observation):
                self._sock.send(self._mp.packb(
                    {"endpoint": "get_action",
                     "data": {"observation": observation}},
                    default=self._mnp.encode))
                resp = self._mp.unpackb(self._sock.recv(),
                                        object_hook=self._mnp.decode, raw=False)
                if isinstance(resp, dict) and "error" in resp:
                    raise RuntimeError(f"policy server: {resp['error']}")
                return resp

        client = _ZmqPolicyClient(args.policy_host, args.policy_port)
        geom = BoardGeometry(board_size=0.34, center_x=0.15, center_y=0.0)
        rng = _rnd.Random(args.seed)
        import math as _m4
        squares = [(i, geom.world_xy("red", ("track", i))) for i in range(TRACK_LEN)]
        squares = [(i, xy) for i, xy in squares
                   if 0.23 < _m4.hypot(xy[0] + 0.22, xy[1]) < 0.52]
        print(f"[groot] closed-loop eval: {args.commands} episodes, "
              f"{len(squares)} reachable squares, server "
              f"{args.policy_host}:{args.policy_port}, exec_horizon "
              f"{args.exec_horizon}", flush=True)
        wrapper.get_observations()
        policy_meta = {
            "server": f"{args.policy_host}:{args.policy_port}",
            "exec_horizon": args.exec_horizon,
            "policy_timeout_steps": args.policy_timeout,
            "instruction_template": args.instruction,
            "target_observable": ("{place}" in args.instruction),
            "note": ("the place square is trial-manager state only (not "
                     "rendered); with a constant instruction the policy "
                     "has no observable placement goal — grasp/lift/"
                     "release are the meaningful funnel stages then"),
        }
        while executed < args.commands:
            (ia, pa), (ib, pb) = rng.sample(squares, 2)
            turn_no += 1
            cmd = PickPlaceCommand(pick_xy=pa, place_xy=pb,
                                   piece="cup", reason="groot17-eval")
            instr = args.instruction.format(pick=ia, place=ib)
            desc = f"G{turn_no} track{ia}->track{ib}"
            print(f"[ludo] EPISODE {turn_no}: {desc} | \"{instr}\"", flush=True)
            ok, det = execute_command_policy(cmd, desc, client, instr)
            det["attempt"] = 1
            det["instruction"] = instr
            gpu_samples.append(_gpu_health_sample())
            executed += 1
            turns_log.write(json.dumps({
                "turn": turn_no, "desc": desc, "commands": 1,
                "ok": [ok], "attempts": [det]}) + "\n")
            turns_log.flush()
            all_details.append(det)
            g = sum(1 for d in all_details if d["grasped"])
            r = sum(1 for d in all_details if d["released"])
            o = sum(1 for d in all_details if d["ok"])
            print(f"[groot] funnel {executed}/{args.commands}: grasped {g} "
                  f"released {r} ok {o}", flush=True)
        print(f"[ludo] groot17 eval done: {executed} episodes", flush=True)

    if args.dice_in > 0:
        import numpy as _np
        from game_core.dice import die_value_up, settled, face_alignment
        die_obj = env.scene["die"]; cup_obj = env.scene["cup"]
        _dev = env.device
        _ph = torch.full((1,), IDLE, dtype=torch.long, device=_dev)
        _def = robot.data.default_joint_pos[:, arm_ids].clone()
        _up = torch.zeros(1, 3, device=_dev); _up[:, 2] = 1.0
        za = torch.zeros(1, 7, device=_dev); za[:, 6] = OPEN_CMD
        for _ in range(40):
            wrapper.step(za)
        GZ = TABLE_SURFACE_Z + 0.012      # die grasp height (18mm die)

        def servo(tx, ty, tz, grip, steps, tol=0.008, cap=True, on_step=None):
            tgt = torch.tensor([[tx, ty, tz]], device=_dev, dtype=torch.float32)
            at = robot.data.joint_pos[:, arm_ids].clone()
            for _ in range(steps):
                dq = level_dq(tgt, _up, _ph)   # position + keep gripper LEVEL
                at = at + dq
                meas = robot.data.joint_pos[:, arm_ids]
                at = torch.clamp(at, meas - 0.55, meas + 0.55).clamp(
                    jlim[:, 0], jlim[:, 1])
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = at - _def
                a[:, 6] = grip
                wrapper.step(a)
                if on_step is not None:
                    on_step()
                dump_frame("dice-in")
                gc = mdp._grasp_center_local(env)
                if cap and float(torch.norm(tgt - gc)) < tol:
                    break

        # full 6-DoF servo (position + target orientation) for the invert-dump:
        # the level servo can only keep the gripper pointing down; pouring the
        # die out needs the wrist to roll ~180 deg, so drive to a target quat.
        def servo_ori(tx, ty, tz, hq, grip, steps, on_step=None):
            tgt = torch.tensor([[tx, ty, tz]], device=_dev, dtype=torch.float32)
            at = robot.data.joint_pos[:, arm_ids].clone()
            for _ in range(steps):
                dq = dls_step(tgt, hq, _ph, ori_w=1.2)
                at = at + dq
                meas = robot.data.joint_pos[:, arm_ids]
                at = torch.clamp(at, meas - 0.55, meas + 0.55).clamp(
                    jlim[:, 0], jlim[:, 1])
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = at - _def
                a[:, 6] = grip
                wrapper.step(a)
                if on_step is not None:
                    on_step()
                dump_frame("dice-dump")

        # kinematic carry (same convention as the A/A+ token/cup carry): the die
        # is glued to the grasp center while the fingers are closed, then freed
        # over the cup. The level-IK stalls ~28-41mm short of any point, so a
        # friction grip is unreliable; gluing to the grasp-center matches how the
        # accepted pick-place demos already move objects.
        def glue_die(pos3):
            root = die_obj.data.default_root_state.clone()
            root[0, 0], root[0, 1], root[0, 2] = float(pos3[0]), float(pos3[1]), float(pos3[2])
            root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=_dev)
            root[0, 7:] = 0.0
            die_obj.write_root_pose_to_sim(root[:, :7])
            die_obj.write_root_velocity_to_sim(root[:, 7:])

        def servo_glue(tx, ty, tz, grip, steps, tol=0.010):
            tgt = torch.tensor([[tx, ty, tz]], device=_dev, dtype=torch.float32)
            at = robot.data.joint_pos[:, arm_ids].clone()
            for _ in range(steps):
                dq = level_dq(tgt, _up, _ph)
                at = at + dq
                meas = robot.data.joint_pos[:, arm_ids]
                at = torch.clamp(at, meas - 0.55, meas + 0.55).clamp(
                    jlim[:, 0], jlim[:, 1])
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = at - _def
                a[:, 6] = grip
                wrapper.step(a)
                glue_die(mdp._grasp_center_local(env)[0])          # die follows gripper
                dump_frame("dice-in carry")
                if float(torch.norm(tgt - mdp._grasp_center_local(env))) < tol:
                    break

        ok = 0
        for k in range(args.dice_in):
            d = die_obj.data.root_pos_w[0].cpu().numpy()
            c = cup_obj.data.root_pos_w[0].cpu().numpy()
            servo(d[0], d[1], GZ + 0.09, OPEN_CMD, 200)            # above die, level
            servo(d[0], d[1], GZ, OPEN_CMD, 160, tol=0.010)        # descend near die
            gc = mdp._grasp_center_local(env)[0]
            glue_die(gc)                                           # pin die at grasp center
            print(f"[dicein] att{k+1} pinned gc=({float(gc[0]):.3f},{float(gc[1]):.3f},{float(gc[2]):.3f})",
                  flush=True)
            for _ in range(40):                                    # close on the pinned die
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = CLOSE_CMD; wrapper.step(a)
                glue_die(mdp._grasp_center_local(env)[0]); dump_frame("dice-in close")
            servo_glue(d[0], d[1], GZ + 0.16, CLOSE_CMD, 200)      # lift (die glued)
            print(f"[dicein] att{k+1} post-lift die_z={float(die_obj.data.root_pos_w[0,2]):.3f} "
                  f"(table {TABLE_SURFACE_Z:.3f})", flush=True)
            servo_glue(c[0], c[1], c[2] + 0.16, CLOSE_CMD, 250)    # over cup
            servo_glue(c[0], c[1], c[2] + 0.10, CLOSE_CMD, 150)    # lower toward rim
            for _ in range(30):                                    # center die over cup mouth
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = CLOSE_CMD; wrapper.step(a)
                glue_die((c[0], c[1], c[2] + 0.06)); dump_frame("dice-in over-cup")
            for _ in range(70):                                    # release: open, die FREE
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = OPEN_CMD; wrapper.step(a); dump_frame("dice-in release")
            for _ in range(150):
                wrapper.step(za); dump_frame("dice-in settle")
            dp = die_obj.data.root_pos_w[0].cpu().numpy()
            cp = cup_obj.data.root_pos_w[0].cpu().numpy()
            r = float(_np.hypot(dp[0] - cp[0], dp[1] - cp[1]))
            inside = (r < 0.016) and (dp[2] > cp[2]) and (dp[2] < cp[2] + 0.05)
            ok += int(inside)
            print(f"[dicein] attempt {k+1}: {'IN ' if inside else 'MISS'} "
                  f"r={r*1000:.0f}mm dz={(dp[2]-cp[2])*1000:.0f}mm", flush=True)
            if not (args.dump and k == args.dice_in - 1):
                root = die_obj.data.default_root_state.clone()
                die_obj.write_root_pose_to_sim(root[:, :7])
                die_obj.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
                for _ in range(20):
                    wrapper.step(za)
        print(f"[dicein] die-in-cup {ok}/{args.dice_in}", flush=True)

        if args.dump:
            import math as _mth
            # die must be in the cup; if the last pick missed, drop it in
            dp0 = die_obj.data.root_pos_w[0].cpu().numpy()
            cp0 = cup_obj.data.root_pos_w[0].cpu().numpy()
            if float(_np.hypot(dp0[0] - cp0[0], dp0[1] - cp0[1])) > 0.016:
                root = die_obj.data.default_root_state.clone()
                root[0, 0], root[0, 1] = float(cp0[0]), float(cp0[1])
                root[0, 2] = float(cp0[2]) + 0.05
                die_obj.write_root_pose_to_sim(root[:, :7])
                die_obj.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
                for _ in range(80):
                    wrapper.step(za); dump_frame("dump-setup")

            c = cup_obj.data.root_pos_w[0].cpu().numpy()
            # approach + descend to the cup body, gripper open
            servo(c[0], c[1], c[2] + 0.14, OPEN_CMD, 200)
            servo(c[0], c[1], c[2] + 0.05, OPEN_CMD, 160, tol=0.012)
            # glue the cup to the gripper: position at the grasp center,
            # orientation = the wrist's rotation since grab, so when the wrist
            # rolls the cup rolls with it (kinematic hold; the die stays a free
            # rigid body and tumbles/pours via physics).
            q_ref = robot.data.body_quat_w[:, tool_idx].clone()   # (1,4)

            def glue_cup():
                gc = mdp._grasp_center_local(env)[0]
                q = robot.data.body_quat_w[:, tool_idx]
                dqr = quat_mul(q, quat_conjugate(q_ref))          # (1,4)
                root = cup_obj.data.default_root_state.clone()
                root[0, 0:3] = gc
                root[0, 3:7] = dqr[0]
                root[0, 7:] = 0.0
                cup_obj.write_root_pose_to_sim(root[:, :7])
                cup_obj.write_root_velocity_to_sim(root[:, 7:])

            for _ in range(40):                                   # close on the cup
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = CLOSE_CMD; wrapper.step(a)
                glue_cup(); dump_frame("dump-grab")
            base_z = c[2] + 0.20
            servo(c[0], c[1], base_z, CLOSE_CMD, 200, on_step=glue_cup)  # lift
            print("[dicein] DUMP: cup lifted, shaking", flush=True)
            # SHAKE: oscillate horizontally, die tumbles inside (physics)
            for i in range(180):
                phse = i / 180.0 * 6 * _mth.pi
                ox = 0.03 * _mth.sin(phse)
                oy = 0.02 * _mth.sin(1.7 * phse)
                servo(c[0] + ox, c[1] + oy, base_z, CLOSE_CMD, 1,
                      cap=False, on_step=glue_cup)
            # move over an empty dump spot
            dump_xy = (0.10, -0.02)
            servo(dump_xy[0], dump_xy[1], base_z, CLOSE_CMD, 220, on_step=glue_cup)
            print("[dicein] DUMP: inverting to pour", flush=True)
            # INVERT: roll the wrist ~180 deg about its local x -> mouth down
            roll180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=_dev)
            q_inv = quat_mul(q_ref, roll180)                      # (1,4)
            servo_ori(dump_xy[0], dump_xy[1], base_z, q_inv, CLOSE_CMD, 400,
                      on_step=glue_cup)
            for _ in range(120):                                  # hold inverted, die pours
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = CLOSE_CMD; wrapper.step(a)
                glue_cup(); dump_frame("dump-pour")
            # raise the cup away and let the die settle on the table
            servo_ori(dump_xy[0], dump_xy[1], base_z + 0.12, q_inv, CLOSE_CMD,
                      150, on_step=glue_cup)
            # let the die settle on a clear table (hold the cup up out of the
            # way) and READ only once it stops tumbling
            def _hold_step():
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = CLOSE_CMD; wrapper.step(a)
                glue_cup(); dump_frame("dump-settle")

            val = None; is_settled = False; redrops = 0
            dqn = die_obj.data.root_quat_w[0].cpu().numpy()
            _tilts = [(0.92, 0.20, 0.34, 0.0), (0.90, 0.30, 0.0, 0.31),
                      (0.88, 0.0, 0.34, 0.33), (0.90, 0.0, 0.0, 0.44)]
            for attempt in range(5):
                for _ in range(150):
                    _hold_step()
                    # bleed off excess spin (die has low angular damping) so it
                    # comes to rest quickly without changing the resting face
                    vel = die_obj.data.root_state_w[:, 7:13].clone()
                    if float(torch.norm(vel[0, 3:6])) > 0.6:
                        vel[0, 3:6] *= 0.75
                        die_obj.write_root_velocity_to_sim(vel)
                    dqn = die_obj.data.root_quat_w[0].cpu().numpy()
                    lvn = die_obj.data.root_lin_vel_w[0].cpu().numpy()
                    avn = die_obj.data.root_ang_vel_w[0].cpu().numpy()
                    if settled(lvn, avn, dqn):
                        val = die_value_up(dqn); is_settled = True; break
                if is_settled:
                    break
                # landed cocked (edge/corner) -> re-drop from 4 cm with a tilt so
                # it topples onto a face (a cocked die is re-rolled)
                dp = die_obj.data.root_pos_w[0].cpu().numpy()
                root = die_obj.data.default_root_state.clone()
                root[0, 0], root[0, 1] = float(dp[0]), float(dp[1])
                root[0, 2] = TABLE_SURFACE_Z + 0.05
                root[0, 3:7] = torch.tensor(_tilts[redrops % len(_tilts)], device=_dev)
                root[0, 7:] = 0.0
                die_obj.write_root_pose_to_sim(root[:, :7])
                die_obj.write_root_velocity_to_sim(root[:, 7:])
                redrops += 1
            if val is None:
                val = die_value_up(dqn)
            lvn = die_obj.data.root_lin_vel_w[0].cpu().numpy()
            avn = die_obj.data.root_ang_vel_w[0].cpu().numpy()
            dpz = float(die_obj.data.root_pos_w[0, 2])
            on_table = dpz < TABLE_SURFACE_Z + 0.03
            print(f"[dicein] DUMP RESULT: rolled={val} settled={bool(is_settled)} "
                  f"align={face_alignment(dqn):.2f} redrops={redrops} "
                  f"|v|={float(_np.linalg.norm(lvn)):.3f} |w|={float(_np.linalg.norm(avn)):.2f} "
                  f"die_z={dpz:.3f} on_table={bool(on_table)}", flush=True)

        import os as _os; _os._exit(0)

    if not policy_on:
        # ================= the game =================
        # board placed in the arm's reach annulus (~0.19-0.55 m from
        # base): near edge at r=0.26, far corner at r=0.55
        game = LudoGameAdapter(board_size=0.34, center=(0.15, 0.0),
                               seed=args.seed)
        if args.endgame:
            # 2-player race to a real winner: 3 tokens already home, one
            # active token each near home so a genuine game finishes fast.
            # Software+reach verified (all squares in the arm annulus): with
            # --seed 0, red CAPTURES green then races home to win in 15 turns.
            from ludo_engine.game import LudoGame, GamePhase, TOKENS
            game.game = LudoGame(players=("red", "green"))
            game.game.positions = {}
            for c in ("red", "green"):
                for t in range(TOKENS):
                    game.game.positions[(c, t)] = ("done", t)
            game.game.positions[("red", 0)] = ("track", 39)
            game.game.positions[("green", 0)] = ("track", 40)
            game.game.turn_idx = 0
            game.game.phase = GamePhase.ROLLING
            game.game.finished_order = []
        else:
            # demo opening: one token per color deployed on its start
            # square (house rule) so play flows from the first roll
            from ludo_engine.game import START
            for color in game.game.players:
                game.game.positions[(color, 0)] = ("track", START[color])
        wrapper.get_observations()

        # ---- milestone C: physical rolls feed the engine ----
        roll_provider = None
        if args.physical_rolls > 0:
            import math as _mroll
            from game_core.dice import die_value_up, settled
            die_obj = env.scene["die"]; cup_obj = env.scene["cup"]
            _dev = env.device
            _ph = torch.full((1,), IDLE, dtype=torch.long, device=_dev)
            _def = robot.data.default_joint_pos[:, arm_ids].clone()
            _up = torch.zeros(1, 3, device=_dev); _up[:, 2] = 1.0

            def _rstep(grip, on_step=None, desc="roll"):
                a = torch.zeros(1, 7, device=_dev)
                a[:, :6] = robot.data.joint_pos[:, arm_ids] - _def
                a[:, 6] = grip; wrapper.step(a)
                if on_step is not None:
                    on_step()
                dump_frame(desc)

            def _rservo(tx, ty, tz, grip, steps, tol=0.010, on_step=None,
                        ori=None):
                tgt = torch.tensor([[tx, ty, tz]], device=_dev, dtype=torch.float32)
                at = robot.data.joint_pos[:, arm_ids].clone()
                for _ in range(steps):
                    dq = (dls_step(tgt, ori, _ph, ori_w=1.2) if ori is not None
                          else level_dq(tgt, _up, _ph))
                    at = at + dq
                    meas = robot.data.joint_pos[:, arm_ids]
                    at = torch.clamp(at, meas - 0.55, meas + 0.55).clamp(
                        jlim[:, 0], jlim[:, 1])
                    a = torch.zeros(1, 7, device=_dev)
                    a[:, :6] = at - _def; a[:, 6] = grip
                    wrapper.step(a)
                    if on_step is not None:
                        on_step()
                    dump_frame("roll" if ori is None else "roll-dump")
                    if tol > 0 and float(torch.norm(
                            tgt - mdp._grasp_center_local(env))) < tol:
                        break

            def physical_roll():
                zo = torch.zeros(1, 7, device=_dev); zo[:, 6] = OPEN_CMD
                c = cup_obj.data.root_pos_w[0].cpu().numpy()
                # ensure the die is in the cup
                rr = die_obj.data.default_root_state.clone()
                rr[0, 0], rr[0, 1] = float(c[0]), float(c[1])
                rr[0, 2] = float(c[2]) + 0.05
                die_obj.write_root_pose_to_sim(rr[:, :7])
                die_obj.write_root_velocity_to_sim(torch.zeros_like(rr[:, 7:]))
                for _ in range(80):
                    wrapper.step(zo); dump_frame("roll-setup")
                q_ref = robot.data.body_quat_w[:, tool_idx].clone()

                def glue_cup():
                    gc = mdp._grasp_center_local(env)[0]
                    q = robot.data.body_quat_w[:, tool_idx]
                    dqr = quat_mul(q, quat_conjugate(q_ref))
                    r = cup_obj.data.default_root_state.clone()
                    r[0, 0:3] = gc; r[0, 3:7] = dqr[0]; r[0, 7:] = 0.0
                    cup_obj.write_root_pose_to_sim(r[:, :7])
                    cup_obj.write_root_velocity_to_sim(r[:, 7:])

                _rservo(c[0], c[1], c[2] + 0.14, OPEN_CMD, 200)
                _rservo(c[0], c[1], c[2] + 0.05, OPEN_CMD, 160)
                for _ in range(40):
                    _rstep(CLOSE_CMD, glue_cup, "roll-grab")
                bz = c[2] + 0.20
                _rservo(c[0], c[1], bz, CLOSE_CMD, 200, on_step=glue_cup)
                for i in range(140):                       # shake
                    ph = i / 140.0 * 6 * _mroll.pi
                    _rservo(c[0] + 0.03 * _mroll.sin(ph),
                            c[1] + 0.02 * _mroll.sin(1.7 * ph), bz,
                            CLOSE_CMD, 1, tol=-1.0, on_step=glue_cup)
                _rservo(0.10, -0.02, bz, CLOSE_CMD, 200, on_step=glue_cup)
                q_inv = quat_mul(q_ref, torch.tensor([[0.0, 1.0, 0.0, 0.0]],
                                                     device=_dev))
                _rservo(0.10, -0.02, bz, CLOSE_CMD, 400, on_step=glue_cup,
                        ori=q_inv)                          # invert-pour
                for _ in range(120):
                    _rstep(CLOSE_CMD, glue_cup, "roll-pour")
                _rservo(0.10, -0.02, bz + 0.12, CLOSE_CMD, 150, on_step=glue_cup,
                        ori=q_inv)
                # settle + read (bleed spin; re-drop a cocked landing)
                val = None
                _tl = [(0.92, 0.2, 0.34, 0.0), (0.9, 0.3, 0.0, 0.31),
                       (0.88, 0.0, 0.34, 0.33), (0.9, 0.0, 0.0, 0.44)]
                rd = 0
                for attempt in range(5):
                    got = False
                    for _ in range(150):
                        _rstep(CLOSE_CMD, glue_cup, "roll-settle")
                        vv = die_obj.data.root_state_w[:, 7:13].clone()
                        if float(torch.norm(vv[0, 3:6])) > 0.6:
                            vv[0, 3:6] *= 0.75
                            die_obj.write_root_velocity_to_sim(vv)
                        dqn = die_obj.data.root_quat_w[0].cpu().numpy()
                        lvn = die_obj.data.root_lin_vel_w[0].cpu().numpy()
                        avn = die_obj.data.root_ang_vel_w[0].cpu().numpy()
                        if settled(lvn, avn, dqn):
                            val = die_value_up(dqn); got = True; break
                    if got:
                        break
                    dp = die_obj.data.root_pos_w[0].cpu().numpy()
                    r = die_obj.data.default_root_state.clone()
                    r[0, 0], r[0, 1] = float(dp[0]), float(dp[1])
                    r[0, 2] = TABLE_SURFACE_Z + 0.05
                    r[0, 3:7] = torch.tensor(_tl[rd % 4], device=_dev)
                    r[0, 7:] = 0.0
                    die_obj.write_root_pose_to_sim(r[:, :7])
                    die_obj.write_root_velocity_to_sim(r[:, 7:])
                    rd += 1
                if val is None:
                    val = die_value_up(die_obj.data.root_quat_w[0].cpu().numpy())
                print(f"[ludo] PHYSICAL ROLL = {val}", flush=True)
                return int(val)

            roll_provider = physical_roll

        phys_turns = 0
        while executed < args.commands and not game.game_over:
            if roll_provider is not None:
                if phys_turns >= args.physical_rolls:
                    break
                _rv = roll_provider()
                plan = game.plan_turn(roll=_rv)
                phys_turns += 1
            else:
                plan = game.plan_turn()
            turn_no += 1
            if plan is None:
                continue
            print(f"[ludo] TURN {turn_no}: {plan.description}", flush=True)
            results = []
            details = []
            for cmd in plan.commands:
                ok, det = execute_command(
                    cmd, f"T{turn_no} {cmd.reason}: {plan.description}")
                det["attempt"] = 1
                gpu_samples.append(_gpu_health_sample())
                details.append(det)
                results.append(ok)
                executed += 1
                if not ok:  # one retry per command (product persistence)
                    ok, det = execute_command(cmd, f"T{turn_no} RETRY {cmd.reason}")
                    det["attempt"] = 2
                    details.append(det)
                    results[-1] = ok
            turns_log.write(json.dumps({
                "turn": turn_no, "desc": plan.description,
                "commands": len(plan.commands), "ok": results,
                "attempts": details}) + "\n")
            turns_log.flush()
            all_details.extend(details)

    print(f"[ludo] game session done: {turn_no} turns, {executed} commands",
          flush=True)
    if not policy_on and not corpus_on:
        _winner = game.result() if game.game_over else "(no winner - cap reached)"
        print(f"[ludo] WINNER: {_winner}", flush=True)
    # session summary — measurements with method, honestly scored,
    # failures as data, operational cost. Append-only variant naming.
    firsts = [d for d in all_details if d["attempt"] == 1]
    oks1 = [d for d in firsts if d["ok"]]
    errs1 = sorted(d["err_mm"] for d in oks1)

    def _pct(v, p):
        return v[min(len(v) - 1, int(p * len(v)))] if v else None

    summary = {
        "scoring_rule": (SCORING_RULE + "; policy lane: AND the cup was "
                         "grasped (adjacent-square false-positive guard)"
                         if policy_on else SCORING_RULE),
        "data_kind": "simulated",
        "turns": turn_no,
        "commands": executed,
        "first_try_ok": len(oks1),
        "first_try_rate": round(len(oks1) / max(1, len(firsts)), 3),
        "with_retry_ok": sum(
            1 for d in all_details if d["ok"]),
        "retries_used": sum(1 for d in all_details if d["attempt"] == 2),
        "ok_err_mm_p50": _pct(errs1, 0.50),
        "ok_err_mm_p95": _pct(errs1, 0.95),
        "lane_counts": {
            "policy": sum(1 for d in firsts if d["lane"] == "policy"),
            "fallback": sum(1 for d in firsts if d["lane"] == "fallback"),
            "groot17": sum(1 for d in firsts if d["lane"] == "groot17")},
        "failures": [d for d in all_details if not d["ok"]],
        "wall_clock_s": round(_time.time() - session_t0, 1),
        "sim_steps_total": sum(d["steps"] for d in all_details),
    }
    if policy_meta is not None:
        policy_meta["funnel"] = {
            "episodes": len(firsts),
            "grasped": sum(1 for d in firsts if d.get("grasped")),
            "lifted": sum(1 for d in firsts if d.get("lifted")),
            "released": sum(1 for d in firsts if d.get("released")),
            "ok": len(oks1),
            "timed_out": sum(1 for d in firsts if d.get("timed_out")),
        }
        summary["policy_eval"] = policy_meta
    # B1 aggregate: policy inference latency across the whole session
    all_infer = sorted(x for d in all_details
                       for x in [d.get("infer_ms_p50")] if x is not None)
    summary["infer_ms_session"] = {
        "p50_of_attempt_p50s": _pctl(all_infer, 50),
        "worst_attempt_p95": max((d.get("infer_ms_p95") or 0)
                                 for d in all_details) if all_details else None,
        "total_calls": sum(d.get("infer_n") or 0 for d in all_details),
    }
    # B3 aggregate: GPU health at command boundaries (host GPU via
    # nvidia-smi; NOT the tegrastats subprocess.run(timeout) anti-pattern)
    temps = [s[0] for s in gpu_samples if s[0] is not None]
    powers = [s[1] for s in gpu_samples if s[1] is not None]
    throttles = [s[2] for s in gpu_samples
                 if s[2] not in (None, "0x0000000000000000", "Not Active")]
    summary["gpu_health"] = {
        "samples": len(gpu_samples),
        "temp_c_peak": max(temps) if temps else None,
        "temp_c_mean": round(sum(temps) / len(temps), 1) if temps else None,
        "power_w_peak": max(powers) if powers else None,
        "throttle_events": throttles[:5],
        "throttled": bool(throttles),
        "note": "sampled at command boundaries only; not continuous",
    }
    sp = args.out / "session_summary.json"
    i = 1
    while sp.exists():
        sp = args.out / f"session_summary.{i}.json"
        i += 1
    sp.write_text(json.dumps(summary, indent=2))
    (args.out / "LUDO_DONE").touch()
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

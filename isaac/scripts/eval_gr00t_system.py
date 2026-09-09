#!/usr/bin/env python3
"""E49 system evaluation: GR00T policy + layered supervision (S0/S1).

S0 (--supervisor off): identical to eval_gr00t_sequence — the control.
S1 (--supervisor):     vectorized L2 retry supervisor. Per-env failure
    signatures (sustained lost pinch after a grasp, or piece drop while
    lifted) trigger a scripted recovery — gripper open, shoulder/elbow
    raised toward ready — then hand back to the policy with a fresh
    action chunk. Retries and wall-clock are reported: the system row's
    honest metric is cycles per hour, not only per episode.

S2 (--supervisor --gemini): S1 plus L3 phase gates. At phase boundaries
    the scene is static, so a 2.6 s cloud judgment costs throughput, not
    correctness (the cycles/hour metric captures the cost honestly):
      grasp-confirm: on a 0->1 phase transition, ER-2 sees the wrist
        camera and answers whether the cup is actually held; a "no"
        verdict triggers recovery immediately (false-grasp catch).
      place-verify: on a rising dock flag, ER-2 sees the overhead
        camera and answers whether the cup is upright in the zone; a
        "no" triggers recovery (bad-place catch).
    API discipline per GEMINI_LEDGER: temperature 0.7 (never 0.0),
    bounded 20 s timeout, one retry, stale/failed answers treated as
    "no verdict" (gate passes open — the L2 signatures still cover).

    ~/.venv/isaacsim5/bin/python isaac/scripts/eval_gr00t_system.py \
        --host 192.168.1.170 --port 5594 --num_envs 16 --steps 8500 \
        --staged --supervisor [--gemini] --headless
"""

from __future__ import annotations

import argparse
import base64
import json as _json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaac.scripts.eval_gr00t_sequence import MiniGrootClient  # noqa: E402

INSTRUCTION = "pick up the cup, carry it to the plate, and set it down gently"
EXEC_HORIZON = 8
PINCH_FAIL_STEPS = 12      # sustained closed-empty pinch => lost grasp
RECOVER_MAX_STEPS = 140
RECOVER_MIN_STEPS = 40

GRASP_PROMPT = (
    "Look at this robot wrist-camera image. Is the red cup securely held "
    "between the gripper fingers (grasped and lifted or lifting)? Answer "
    'JSON only: {"held": true} or {"held": false}.'
)
PLACE_PROMPT = (
    "Look at this overhead image of a robot workspace. Is the red cup "
    "upright and resting stably on the table surface in the placement "
    'area (not tipped over, not held by the robot)? Answer JSON only: '
    '{"placed": true} or {"placed": false}.'
)


def _gemini_yes_no(png_bytes: bytes, prompt: str, key: str,
                   field: str, timeout: float = 20.0) -> bool | None:
    """One bounded ER-2 judgment. Returns True/False, or None on any
    failure (timeout, parse, HTTP) — callers treat None as no-verdict."""
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-robotics-er-2-preview:generateContent")
    body = _json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(png_bytes).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.7},
    }).encode()
    for _ in range(2):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json", "x-goog-api-key": key})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = _json.load(r)
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            text = text.replace("```json", "").replace("```", "").strip()
            return bool(_json.loads(text)[field])
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.1.170")
    parser.add_argument("--port", type=int, default=5594)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--supervisor", action="store_true",
                        help="enable the L2 retry supervisor (S1)")
    parser.add_argument("--save_gate_frames", type=str, default=None,
                        help="dir to dump each gate-judged frame + verdict "
                        "(calibration audit)")
    parser.add_argument("--gemini", action="store_true",
                        help="enable L3 Gemini phase gates (S2; implies "
                        "--supervisor mechanics for gate-triggered retries)")
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
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
        "Synria-Chess-PickPlace-v0", device="cuda", num_envs=args.num_envs
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
    finger_ids = [int(rids["left"]), int(rids["right"])]
    n = env.num_envs
    dev = env.device
    default_arm = robot.data.default_joint_pos[:, arm_ids]
    default_l = robot.data.default_joint_pos[:, rids["left"]]
    mgr = mdp._get_trial_mgr(env)

    client = MiniGrootClient(args.host, args.port)
    gemini_key = None
    if args.gemini:
        args.supervisor = True  # gates need the recovery machinery
        gemini_key = (Path.home() / ".config/gemini/api_key").read_text().strip()
    print(f"[sys] connected {args.host}:{args.port} | supervisor="
          f"{'ON' if args.supervisor else 'OFF'} | gemini="
          f"{'ON' if args.gemini else 'OFF'}", flush=True)

    def frames():
        outs = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            outs.append(rgb[..., :3])
        return outs[0], outs[1]

    totals = {"grasp": 0, "carry5s": 0, "dock": 0, "cycle": 0}
    prev_cycles = mgr.cycles.clone()
    prev_f = {"carry5s": mgr.f_carry5s.clone(), "dock": mgr.f_dock.clone()}
    prev_phase = mgr.phase.clone()

    # supervisor state (vectorized)
    sup_mode = torch.zeros(n, dtype=torch.long, device=dev)     # 0 policy, 1 recovering
    sup_t = torch.zeros(n, dtype=torch.long, device=dev)
    pinch_run = torch.zeros(n, dtype=torch.long, device=dev)
    was_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_ep_len = env.episode_length_buf.clone()
    retry_events = 0
    chunk = None
    chunk_i = 0
    need_fresh_chunk = torch.zeros(n, dtype=torch.bool, device=dev)
    t_start = time.time()

    # S2 gate state
    gate_stats = {"queries": 0, "negatives": 0, "gate_retries": 0,
                  "no_verdict": 0}
    GATE_SETTLE = 15   # steps to let the scene settle before judging
    pend_grasp = torch.full((n,), -1, dtype=torch.long, device=dev)
    pend_place = torch.full((n,), -1, dtype=torch.long, device=dev)

    def _png(cam_key: str, env_i: int) -> bytes:
        import io
        from PIL import Image
        rgb = env.scene.sensors[cam_key].data.output["rgb"][env_i]
        fr = rgb.detach().cpu().numpy()
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(fr[..., :3]).save(buf, format="PNG")
        return buf.getvalue()

    for step in range(args.steps):
        if chunk is None or chunk_i >= EXEC_HORIZON or bool(need_fresh_chunk.any()):
            joint_state = robot.data.joint_pos
            state_arm = joint_state[:, arm_ids].cpu().numpy().astype(np.float32)
            state_grip = joint_state[:, finger_ids].cpu().numpy().astype(np.float32)
            wrist_f, over_f = frames()
            obs = {
                "video": {"wrist": wrist_f[:, None], "overhead": over_f[:, None]},
                "state": {"arm": state_arm[:, None], "gripper": state_grip[:, None]},
                "language": {
                    "annotation.human.task_description": [[args.instruction]] * n,
                },
            }
            resp = client.get_action(obs)
            if isinstance(resp, (list, tuple)):
                resp = resp[0]
            arm_chunk = np.asarray(resp["arm"], dtype=np.float32)
            grip_chunk = np.asarray(resp["gripper"], dtype=np.float32)
            if arm_chunk.ndim == 2:
                arm_chunk, grip_chunk = arm_chunk[None], grip_chunk[None]
            chunk = np.concatenate([arm_chunk, grip_chunk], axis=-1)
            chunk_i = 0
            need_fresh_chunk[:] = False

        act = torch.from_numpy(chunk[:, min(chunk_i, chunk.shape[1] - 1)]).to(dev)
        chunk_i += 1
        tgt = (default_arm + act[:, :6]).clamp(
            torch.tensor([-2.749, -2.0, -0.5, -2.79, -1.57, -3.14159], device=dev),
            torch.tensor([2.749, 2.0, 3.14159, 2.79, 1.57, 3.14159], device=dev),
        )
        l_target = (default_l + act[:, 6]).clamp(0.0, 0.025)
        action = torch.cat(
            [tgt - default_arm,
             ((l_target > 0.020).float() * 2.0 - 1.0).unsqueeze(-1)], dim=1)

        if args.supervisor:
            # --- per-env episode-boundary reset of supervisor latches ---
            cur_ep = env.episode_length_buf
            rolled = cur_ep < prev_ep_len
            if bool(rolled.any()):
                was_lifted[rolled] = False
                pinch_run[rolled] = 0
                sup_mode[rolled] = 0
                sup_t[rolled] = 0
            prev_ep_len = cur_ep.clone()
            # --- failure detection (policy-mode envs only) ---
            width = mdp._gripper_width_m(env).squeeze(-1)
            grasped = mgr.phase >= 1
            lifted = mdp._grasp_center_local(env)[:, 2] > (TABLE_SURFACE_Z + 0.11)
            # latch "was lifted" only while actually HOLDING the cup —
            # the bare hand is high at ready pose (smoke bug: 16 false
            # drop-triggers at step 0)
            was_lifted |= (lifted & grasped)
            pinched_empty = grasped & (width < 0.033) & (mgr.phase < 2) & ~lifted
            pinch_run = torch.where(pinched_empty, pinch_run + 1,
                                    torch.zeros_like(pinch_run))
            dropped = was_lifted & (mgr.phase == 0)
            trigger = (sup_mode == 0) & ((pinch_run > PINCH_FAIL_STEPS) | dropped)
            if bool(trigger.any()):
                retry_events += int(trigger.sum())
                sup_mode[trigger] = 1
                sup_t[trigger] = 0
                was_lifted[trigger] = False
                pinch_run[trigger] = 0
            # --- scripted recovery for recovering envs ---
            rec = sup_mode == 1
            if bool(rec.any()):
                cur = robot.data.joint_pos[:, arm_ids]
                tgt_r = cur.clone()
                tgt_r[:, 1] = default_arm[:, 1]
                tgt_r[:, 2] = default_arm[:, 2]
                step_q = cur + (tgt_r - cur) * 0.06
                rec_action = torch.cat(
                    [step_q - default_arm,
                     torch.ones(n, 1, device=dev)], dim=1)
                action = torch.where(rec.unsqueeze(-1), rec_action, action)
                sup_t[rec] += 1
                high = mdp._grasp_center_local(env)[:, 2] > (TABLE_SURFACE_Z + 0.11)
                done_rec = rec & ((sup_t > RECOVER_MAX_STEPS) |
                                  ((sup_t > RECOVER_MIN_STEPS) & high))
                if bool(done_rec.any()):
                    sup_mode[done_rec] = 0
                    sup_t[done_rec] = 0
                    need_fresh_chunk |= done_rec

        env.step(action)

        new_grasp = (prev_phase == 0) & (mgr.phase == 1)
        totals["grasp"] += int(new_grasp.sum())
        new_dock = None
        for k in ("carry5s", "dock"):
            f = getattr(mgr, f"f_{k}")
            rising = f & ~prev_f[k]
            if k == "dock":
                new_dock = rising
            totals[k] += int(rising.sum())
            prev_f[k] = f.clone()
        totals["cycle"] += int(((mgr.cycles - prev_cycles) > 0).sum())
        prev_cycles = mgr.cycles.clone()
        prev_phase = mgr.phase.clone()

        if args.gemini:
            # schedule gates on rising edges (judged after a settle delay)
            pend_grasp[new_grasp & (pend_grasp < 0)] = step + GATE_SETTLE
            if new_dock is not None:
                pend_place[new_dock & (pend_place < 0)] = step + GATE_SETTLE
            # serve at most ONE due gate per step (bounds cloud blocking)
            due_g = ((pend_grasp >= 0) & (pend_grasp <= step)).nonzero(as_tuple=True)[0]
            due_p = ((pend_place >= 0) & (pend_place <= step)).nonzero(as_tuple=True)[0]
            if len(due_g) > 0:
                i = int(due_g[0])
                pend_grasp[i] = -1
                if int(mgr.phase[i]) >= 1:   # still claims a grasp
                    gate_stats["queries"] += 1
                    png = _png("wrist", i)
                    v = _gemini_yes_no(png, GRASP_PROMPT, gemini_key, "held")
                    if args.save_gate_frames:
                        d = Path(args.save_gate_frames); d.mkdir(parents=True, exist_ok=True)
                        (d / f"grasp_s{step}_e{i}_v{v}.png").write_bytes(png)
                    if v is None:
                        gate_stats["no_verdict"] += 1
                    elif not v:
                        gate_stats["negatives"] += 1
                        gate_stats["gate_retries"] += 1
                        sup_mode[i] = 1
                        sup_t[i] = 0
            elif len(due_p) > 0:
                i = int(due_p[0])
                pend_place[i] = -1
                gate_stats["queries"] += 1
                png = _png("overhead", i)
                v = _gemini_yes_no(png, PLACE_PROMPT, gemini_key, "placed")
                if args.save_gate_frames:
                    d = Path(args.save_gate_frames); d.mkdir(parents=True, exist_ok=True)
                    (d / f"place_s{step}_e{i}_v{v}.png").write_bytes(png)
                if v is None:
                    gate_stats["no_verdict"] += 1
                elif not v:
                    gate_stats["negatives"] += 1
                    gate_stats["gate_retries"] += 1
                    sup_mode[i] = 1
                    sup_t[i] = 0

        if step % 500 == 0:
            print(f"[sys] step={step} totals={totals} retries={retry_events}",
                  flush=True)

    wall_h = (time.time() - t_start) / 3600
    ep_eq = n * args.steps / int(env.max_episode_length)
    cfg = "S2" if args.gemini else ("S1" if args.supervisor else "S0")
    print(f"[sys] ======== SYSTEM RESULT ({cfg}) ========", flush=True)
    print(f"[sys] episode-equivalents: {ep_eq:.1f} | wall: {wall_h:.2f} h "
          f"| retries: {retry_events}", flush=True)
    if args.gemini:
        print(f"[sys] gates: {gate_stats}", flush=True)
    for k, v in totals.items():
        print(f"[sys] {k}: {v} total, {v / ep_eq:.3f}/episode, "
              f"{v / wall_h:.1f}/hour", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

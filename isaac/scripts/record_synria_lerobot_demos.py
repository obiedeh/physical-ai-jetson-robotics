"""Record success-filtered Synria cup demos with cameras → raw LeRobot input.

Runs the SAME hybrid expert as ``record_synria_demos.py`` (policy picks +
scripted wander/release/retreat, attach-on-grasp env) on the recorder env
variant (wrist + overhead 224×224 RGB), and saves ONLY segments that end in
a completed sequence (trial-manager cycle increment). Failed attempts and
env-reset truncations are discarded.

Output (raw, converted to LeRobot v2.1 by ``convert_synria_lerobot.py``
in the gr00t venv — pandas/pyarrow are not available in the Isaac venv):

    <out_root>/
        episode_000000/
            states.npy    (T, 8) float32 — Joint1..6 pos (rad), left/right finger (m)
            actions.npy   (T, 8) float32 — env actions (offsets from default pose,
                                           same convention the policy was trained on)
            wrist.mp4     (T frames, 224×224, 30 fps)
            overhead.mp4
        ...
        raw_manifest.jsonl  (one line per episode: index, length, seed, env)

Per-frame alignment: state_t is read BEFORE stepping, action_t is the
action applied at t, frames are read AFTER the previous step completes —
i.e. everything at index t describes the same control tick t (30 Hz).

Usage:
    ~/.venv/isaacsim5/bin/python isaac/scripts/record_synria_lerobot_demos.py \
        --headless --checkpoint <E8_final.pt> [--num_envs 16] [--steps 6000] \
        [--max_episodes 120] [--seed 7] [--out_root reports/training/synria_cup_lerobot_raw]
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

# Expert constants — derived from isaac/scripts/record_synria_demos.py, with
# two carry-funnel fixes found by the [dbg] probe (see GR00T ledger G5):
# the piece settles ~2-3 cm after pose-freeze, so take over from a higher
# lift and only treat a near-table drop as a slip; and LOWER back to the
# grasp-moment pose before opening so the set-down is gentle.
POLICY, WANDER, LOWER, RELEASE, RETREAT = range(5)
# G7: the yaw sweep is unnecessary (move_dist ≫ 0.20 m accumulates during
# the policy's own carry) and squeezing commanded at 31 mm into the
# kinematically-attached 40 mm cup rattled the arm ±5 cm, resetting the
# consecutive hold clock. WANDER is now a STATIC hold: frozen arm pose,
# fingers commanded at the cup's width (no squeeze force).
YAW_AMP = 0.0
YAW_PERIOD = 60
HOLD_FINGER_M = 0.019  # 38 mm width: pads at the cup surface, below 42 mm detach
LIFT_SOLID = 0.08     # was 0.05: settle after freeze dropped piece to ~lift threshold
SLIP_EXIT = 0.005     # was 0.02: only a true drop aborts the wander
LOWER_STEPS = 60      # joint-space interpolation back to the grasp pose

MAX_SEG_STEPS = 900   # one full episode; longer attempts are discarded
FPS = 30              # control rate: sim dt 1/60 × decimation 2
INSTRUCTION = "pick up the cup, carry it, and set it back down where it started"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--max_episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out_root", type=str, default="reports/training/synria_cup_lerobot_raw"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--pure_policy",
        action="store_true",
        help=(
            "Disable the scripted expert entirely — record the policy's own "
            "successful cycles (viable from E10-era checkpoints whose staged "
            "cycle rate is 15-19%%, vs ~2%% for the hybrid expert)."
        ),
    )
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import cv2  # type: ignore[import-not-found]
    import gymnasium as gym  # type: ignore[import-not-found]
    import imageio  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

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
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

    # G17: surface-pick staging for demo throughput. The v10 funnel showed
    # takeovers start too late in the 900-step episode (33/98 truncated,
    # 3 cycles/175 ep-eq). The ZONE stage pins the cup ON THE TABLE under
    # the settling hand with phase = PICK — the policy still performs a
    # real from-surface grasp, and the whole pick→carry→set-down→return
    # sequence fits comfortably in the horizon. PREGRASP/CARRY_ELAPSED
    # stay 0 (they'd start with the cup already in hand — no pick in the
    # demo). Caveat for the dataset: 80% of demos start with the cup at
    # the rendezvous point rather than a random square.
    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.8
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

    # G14: gate the attach event on a script-controlled suppression mask.
    # Pre-step clears of mgr.attached (G10-G12) always lost the race — the
    # event re-attaches INSIDE env.step, and while attached the cup rides
    # the hand so the 9 cm 'near' break can never be reached, while wedged
    # fingers can be unable to close below the 4 mm band. Wrapping the
    # event func on THIS recorder cfg only (repo env untouched): for
    # suppressed envs it clears the glue, blocks new attaches via the
    # event's own pin_steps==0 guard, and therefore skips the kinematic
    # write — the cup is fully physical during release/retreat.
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
    rc = TASK_TRAIN_CFG["Synria-Chess-PickPlace-v0"]()
    runner = OnPolicyRunner(wrapper, rc.to_dict(), log_dir=None, device="cuda")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda")
    print(f"[rec] policy loaded from {args.checkpoint}", flush=True)

    env = raw.unwrapped
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    finger_ids = [int(rids["left"]), int(rids["right"])]
    dev = env.device
    n = env.num_envs
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    default_arm = robot.data.default_joint_pos[:, arm_ids]
    default_l = robot.data.default_joint_pos[:, rids["left"]]
    default_r = robot.data.default_joint_pos[:, rids["right"]]
    # URDF joint limits — the env clips position targets to these before
    # applying; the dataset must store the APPLIED targets (validator G13:
    # raw policy outputs reach ±12 rad and would poison normalization).
    jlim = torch.tensor(
        [[-2.749, 2.749], [-2.0, 2.0], [-0.5, 3.14159],
         [-2.79, 2.79], [-1.57, 1.57], [-3.14159, 3.14159]],
        device=dev,
    )

    mgr = mdp._get_trial_mgr(env)
    mode = torch.zeros(n, dtype=torch.long, device=dev)
    prev_act = torch.zeros(n, 8, device=dev)
    freeze_pose = default_arm.clone()
    wander_comp = torch.zeros(n, 6, device=dev)  # G18 droop compensation
    grasp_pose = default_arm.clone()  # arm pose at the grasp moment (cup at table)
    hold_l = torch.full((n,), 0.02, device=dev)
    mode_t = torch.zeros(n, dtype=torch.long, device=dev)
    prev_phase = mgr.phase.clone()
    prev_cycles = mgr.cycles.clone()
    prev_ep = env.episode_length_buf.clone()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Default joint pose — needed downstream to reconstruct absolute targets
    # from the offset-convention actions (target = default + action).
    (out_root / "defaults.json").write_text(
        json.dumps(
            {
                "arm_default_rad": default_arm[0].cpu().tolist(),
                "left_finger_default_m": float(default_l[0]),
                "right_finger_default_m": float(default_r[0]),
                "joint_limits_rad": {
                    "Joint1": [-2.749, 2.749], "Joint2": [-2.0, 2.0],
                    "Joint3": [-0.5, 3.14159], "Joint4": [-2.79, 2.79],
                    "Joint5": [-1.57, 1.57], "Joint6": [-3.14159, 3.14159],
                },
                "finger_limits_m": {"left": [0.0, 0.025], "right": [-0.025, 0.0]},
            },
            indent=2,
        )
    )
    manifest_path = out_root / "raw_manifest.jsonl"
    manifest = manifest_path.open("a")

    # Per-env rolling buffers of the CURRENT attempt (cleared on save/reset).
    bufs: list[deque] = [deque(maxlen=MAX_SEG_STEPS + 1) for _ in range(n)]
    saved = 0
    # Funnel counters (G16): where do attempts die?
    funnel = {
        "takeover": 0,      # POLICY -> WANDER
        "carry_done": 0,    # WANDER -> LOWER (phase 2 reached in WANDER)
        "released": 0,      # LOWER -> RELEASE
        "retreat": 0,       # RELEASE -> RETREAT
        "setdown": 0,       # phase 2 -> 3 observed
        "cycle": 0,         # cycles incremented (episode saved-eligible)
        "trunc_in_script": 0,  # env reset while mode >= WANDER
        "stuck_recover": 0,    # stuck-RETREAT recovery fired
    }

    def grab_frames() -> tuple[np.ndarray, np.ndarray]:
        outs = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"]
            frame = rgb.detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            outs.append(frame[..., :3])
        return outs[0], outs[1]

    def save_episode(e: int) -> None:
        nonlocal saved
        seg = list(bufs[e])
        if len(seg) < 30 or len(seg) > MAX_SEG_STEPS:
            return  # degenerate or over-long attempt
        ep_dir = out_root / f"episode_{saved:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        states = np.stack([s for s, _, _, _ in seg]).astype(np.float32)
        actions = np.stack([a for _, a, _, _ in seg]).astype(np.float32)
        np.save(ep_dir / "states.npy", states)
        np.save(ep_dir / "actions.npy", actions)
        for vid_idx, name in ((2, "wrist"), (3, "overhead")):
            writer = imageio.get_writer(
                ep_dir / f"{name}.mp4", fps=FPS, codec="libx264",
                quality=8, macro_block_size=1,
            )
            for row in seg:
                bgr = cv2.imdecode(row[vid_idx], cv2.IMREAD_COLOR)
                writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            writer.close()
        manifest.write(
            json.dumps(
                {
                    "episode_index": saved,
                    "length": len(seg),
                    "env": e,
                    "seed": args.seed,
                    "fps": FPS,
                    "instruction": INSTRUCTION,
                    "checkpoint": str(args.checkpoint),
                }
            )
            + "\n"
        )
        manifest.flush()
        saved += 1
        print(f"[rec] episode {saved}/{args.max_episodes} saved (len={len(seg)}, env={e})",
              flush=True)

    obs = wrapper.get_observations()

    for step in range(args.steps):
        piece = mdp._get_piece_pos(env)
        phase = mgr.phase

        # --- expert mode transitions ------------------------------------
        if args.pure_policy:
            # Funnel: count set-downs, keep phases fresh; no expert modes.
            funnel["setdown"] += int(
                (
                    (prev_phase == int(TaskPhase.PICK_FROM_ZONE))
                    & (phase == int(TaskPhase.RETURN_TO_BOARD))
                ).sum()
            )
            prev_phase = phase.clone()
            with torch.inference_mode():
                action = policy(obs).clone()
            joint_state = robot.data.joint_pos
            state8 = torch.cat(
                [joint_state[:, arm_ids], joint_state[:, finger_ids]], dim=1
            ).cpu().numpy().astype(np.float32)
            arm_tgt = (default_arm + action[:, :6]).clamp(jlim[:, 0], jlim[:, 1])
            l_tgt = (default_l + action[:, 6]).clamp(0.0, 0.025)
            r_tgt = (default_r + action[:, 7]).clamp(-0.025, 0.0)
            act_rec = torch.cat(
                [
                    arm_tgt - default_arm,
                    (l_tgt - default_l).unsqueeze(-1),
                    (r_tgt - default_r).unsqueeze(-1),
                ],
                dim=1,
            )
            act_np = act_rec.detach().cpu().numpy().astype(np.float32)
            wrist_f, over_f = grab_frames()
            for e in range(n):
                ok_w, jw = cv2.imencode(
                    ".jpg", cv2.cvtColor(wrist_f[e], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                ok_o, jo = cv2.imencode(
                    ".jpg", cv2.cvtColor(over_f[e], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok_w and ok_o:
                    bufs[e].append((state8[e], act_np[e], jw, jo))
            obs, _, _, _ = wrapper.step(action)
            prev_act = action.clone()
            cyc_now = (mgr.cycles - prev_cycles) > 0
            funnel["cycle"] += int(cyc_now.sum())
            for e in cyc_now.nonzero().flatten().tolist():
                save_episode(int(e))
                bufs[int(e)].clear()
            prev_cycles = mgr.cycles.clone()
            ep = env.episode_length_buf
            finished = ep < prev_ep
            for e in finished.nonzero().flatten().tolist():
                bufs[int(e)].clear()
            prev_ep = ep.clone()
            if step % 200 == 0:
                print(
                    f"[rec] step={step} saved={saved} pure-policy "
                    f"cycles={funnel['cycle']} setdowns={funnel['setdown']}",
                    flush=True,
                )
            if saved >= args.max_episodes:
                print("[rec] max episodes reached — stopping", flush=True)
                break
            continue

        # Snapshot the grasp-moment arm pose: phase just flipped into
        # PLACE_ON_ZONE means the grasp was confirmed with the cup at
        # table height — LOWER interpolates back to this pose to set down.
        grasped = (phase == int(TaskPhase.PLACE_ON_ZONE)) & (
            prev_phase == int(TaskPhase.PICK_FROM_BOARD)
        )
        if bool(grasped.any()):
            grasp_pose[grasped] = robot.data.joint_pos[grasped][:, arm_ids]

        take = (
            (mode == POLICY)
            & (phase == int(TaskPhase.PLACE_ON_ZONE))
            & ((piece[:, 2] - rest_z) >= LIFT_SOLID)
            & (mgr.hold_steps >= 20)
        )
        if bool(take.any()):
            # Freeze at the policy's last COMMANDED target, not the measured
            # pose: with 5 Nm arm effort the PD saturates under load, so the
            # measured pose is the sagging equilibrium — commanding it as
            # the target dropped the cup ~7 cm (G7 probe). The policy's own
            # target is the one that demonstrably held the cup aloft.
            freeze_pose[take] = (default_arm + prev_act[:, :6])[take]
            # Fingers: hold at their ACTUAL positions — any other target
            # drives them against the kinematically-locked cup (G6 rattle).
            hold_l[take] = robot.data.joint_pos[take, rids["left"]].clamp(0.0, 0.025)
            wander_comp[take] = 0.0
            mode[take], mode_t[take] = WANDER, 0
            funnel["takeover"] += int(take.sum())
        # Carry requirement met — lower the cup back to the grasp pose.
        m = (mode == WANDER) & (phase == int(TaskPhase.PICK_FROM_ZONE))
        if bool(m.any()):
            mode[m], mode_t[m] = LOWER, 0
            funnel["carry_done"] += int(m.sum())
        # True drop (attach lost / never engaged): hand back to the policy.
        m = (mode == WANDER) & ((piece[:, 2] - rest_z) < SLIP_EXIT) & (mode_t > 10)
        mode[m], mode_t[m] = POLICY, 0
        # Lowered to (near) table height — begin the release sequence.
        m = (mode == LOWER) & (
            (mode_t >= LOWER_STEPS) | ((piece[:, 2] - rest_z) < 0.02)
        )
        if bool(m.any()):
            # Re-capture the no-squeeze width at set-down (G15).
            hold_l[m] = robot.data.joint_pos[m, rids["left"]].clamp(0.0, 0.025)
            mode[m], mode_t[m] = RELEASE, 0
            funnel["released"] += int(m.sum())
        # G14: publish the suppression mask consumed by _gated_attach inside
        # env.step — envs releasing or retreating (still-carryish) get a
        # fully physical cup. The release DECISION stays deliberate (cup is
        # lowered to table height before this engages).
        carryish = (phase == int(TaskPhase.PLACE_ON_ZONE)) | (
            phase == int(TaskPhase.PICK_FROM_ZONE)
        )
        suppress_state["mask"] = ((mode == RELEASE) & (mode_t >= 10)) | (
            (mode == RETREAT) & carryish
        )
        # Stuck-RETREAT recovery: if the set-down never fires (cup shoved
        # out of tolerance etc.), stop wasting the env and let the policy
        # start a fresh attempt.
        m = (mode == RETREAT) & carryish & (mode_t > 200)
        if bool(m.any()):
            funnel["stuck_recover"] += int(m.sum())
        mode[m], mode_t[m] = POLICY, 0
        # Set-down observation (phase 2 -> 3 anywhere).
        funnel["setdown"] += int(
            (
                (prev_phase == int(TaskPhase.PICK_FROM_ZONE))
                & (phase == int(TaskPhase.RETURN_TO_BOARD))
            ).sum()
        )
        width_now = mdp._gripper_width_m(env).squeeze(-1)
        # G12: RELEASE no longer opens in place (in-step re-attach makes the
        # glued cup unopenable, probes v5/v6). It closes below the 4 mm
        # attach band, detaches, and withdraws the arm home with fingers
        # still closed — once tool0 is >9 cm from the cup the attach 'near'
        # condition is broken for good. RETREAT then opens the fingers far
        # from the cup so the measured width can reach the 40 mm released
        # bar without anything blocking.
        m = (mode == RELEASE) & (mode_t >= 50)
        if bool(m.any()):
            funnel["retreat"] += int(m.sum())
        mode[m], mode_t[m] = RETREAT, 0
        cyc = (mgr.cycles - prev_cycles) > 0
        m = (mode == RETREAT) & cyc
        mode[m], mode_t[m] = POLICY, 0
        mode_t += 1
        prev_phase = phase.clone()

        # --- compose action ---------------------------------------------
        with torch.inference_mode():
            pol_act = policy(obs)
        action = pol_act.clone()

        m = mode == WANDER
        if bool(m.any()):
            # G18: integral droop compensation. The policy held altitude by
            # dynamically re-commanding against gravity droop (5 Nm effort
            # saturates); any FROZEN target settles at target−droop, which
            # for many grips puts the cup under the +0.02 lift threshold
            # and resets the 150-step hold clock (v10/v11 funnel: 85% of
            # takeovers die here). Push the target further along the
            # command-vs-measured error, sign-free, capped at ±0.3 rad.
            err = (freeze_pose + wander_comp) - robot.data.joint_pos[:, arm_ids]
            wander_comp = torch.where(
                m.unsqueeze(-1),
                (wander_comp + 0.2 * err.clamp(-0.02, 0.02)).clamp(-0.3, 0.3),
                wander_comp,
            )
            yaw = YAW_AMP * torch.sin(mode_t.float() * 6.28318 / YAW_PERIOD)
            wt = freeze_pose + wander_comp
            wt[:, 0] = wt[:, 0] + yaw
            action[m, :6] = (wt - default_arm)[m]
            action[m, 6] = (hold_l - default_l)[m]
            action[m, 7] = (-hold_l - default_r)[m]
        m = mode == LOWER
        if bool(m.any()):
            alpha = (mode_t.float() / LOWER_STEPS).clamp(0.0, 1.0).unsqueeze(-1)
            wt = (freeze_pose + wander_comp) * (1.0 - alpha) + grasp_pose * alpha
            action[m, :6] = (wt - default_arm)[m]
            action[m, 6] = (hold_l - default_l)[m]
            action[m, 7] = (-hold_l - default_r)[m]
        m = mode == RELEASE
        if bool(m.any()):
            # G15: exit the cup the way we came in. The fingers sit INSIDE
            # the cup mouth (inner diameter ~36 mm < the 40 mm released
            # bar, v8), so they cannot open in place — lift them out
            # vertically first.
            #   t<10:  hold the set-down pose, fingers at their current
            #          width (stop squeezing; cup rests on the table).
            #   t>=10: attach suppressed — raise the arm back to the aloft
            #          freeze pose (reverse of LOWER), fingers held.
            settle = m & (mode_t < 10)
            lifting = m & (mode_t >= 10)
            action[settle, :6] = (grasp_pose - default_arm)[settle]
            alpha = ((mode_t - 10).float() / 30.0).clamp(0.0, 1.0).unsqueeze(-1)
            lift_tgt = grasp_pose * (1.0 - alpha) + freeze_pose * alpha
            action[lifting, :6] = (lift_tgt - default_arm)[lifting]
            # Settle: keep the no-squeeze hold. Lift-out (G19): command OPEN
            # — holding grip width friction-drags the freed cup up with the
            # hand (bank1: retreat→setdown 6/18); pressing outward on the
            # inner walls pins the cup DOWN while the fingers slide up past
            # the rim and pop free.
            action[settle, 6] = (hold_l - default_l)[settle]
            action[settle, 7] = (-hold_l - default_r)[settle]
            action[lifting, 6] = (0.025 - default_l)[lifting]
            action[lifting, 7] = (-0.025 - default_r)[lifting]
        m = mode == RETREAT
        if bool(m.any()):
            # Arm home; fingers OPEN — far from the cup nothing blocks
            # them, so measured width clears the 40 mm released bar and the
            # set-down can fire (home detection is arm-joints-only).
            action[m, :6] = 0.0
            action[m, 6] = (0.025 - default_l)[m]
            action[m, 7] = (-0.025 - default_r)[m]

        # --- record this control tick ------------------------------------
        joint_state = robot.data.joint_pos  # (n, dof)
        state8 = torch.cat(
            [joint_state[:, arm_ids], joint_state[:, finger_ids]], dim=1
        ).cpu().numpy().astype(np.float32)
        # Store applied-target offsets: clip (default + action) to limits,
        # re-express as offsets. The env receives the raw action unchanged.
        arm_tgt = (default_arm + action[:, :6]).clamp(jlim[:, 0], jlim[:, 1])
        l_tgt = (default_l + action[:, 6]).clamp(0.0, 0.025)
        r_tgt = (default_r + action[:, 7]).clamp(-0.025, 0.0)
        act_rec = torch.cat(
            [
                arm_tgt - default_arm,
                (l_tgt - default_l).unsqueeze(-1),
                (r_tgt - default_r).unsqueeze(-1),
            ],
            dim=1,
        )
        act_np = act_rec.detach().cpu().numpy().astype(np.float32)
        wrist_f, over_f = grab_frames()
        for e in range(n):
            ok_w, jw = cv2.imencode(".jpg", cv2.cvtColor(wrist_f[e], cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
            ok_o, jo = cv2.imencode(".jpg", cv2.cvtColor(over_f[e], cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok_w and ok_o:
                bufs[e].append((state8[e], act_np[e], jw, jo))

        obs, _, _, _ = wrapper.step(action)
        prev_act = action.clone()

        # --- bookkeeping: save successes, drop failures -------------------
        cyc_now = (mgr.cycles - prev_cycles) > 0
        funnel["cycle"] += int(cyc_now.sum())
        for e in cyc_now.nonzero().flatten().tolist():
            save_episode(int(e))
            bufs[int(e)].clear()
        prev_cycles = mgr.cycles.clone()

        ep = env.episode_length_buf
        finished = ep < prev_ep
        funnel["trunc_in_script"] += int((finished & (mode >= WANDER)).sum())
        for e in finished.nonzero().flatten().tolist():
            bufs[int(e)].clear()  # attempt truncated by reset — not a demo
            mode[int(e)] = POLICY
            mode_t[int(e)] = 0
        prev_ep = ep.clone()

        if step % 200 == 0:
            counts = [int((mode == p).sum()) for p in range(5)]
            print(
                f"[rec] step={step} saved={saved} "
                f"modes(policy/wander/lower/release/retreat)={counts} "
                f"funnel={funnel}",
                flush=True,
            )
        if args.debug and step % 30 == 0:
            # Carry-funnel diagnosis: why doesn't carry_done fire in WANDER?
            for e in (mode >= WANDER).nonzero().flatten().tolist()[:3]:
                att = bool(mgr.attached[e]) if hasattr(mgr, "attached") else None
                print(
                    f"[dbg] e{e} mode={int(mode[e])} t={int(mode_t[e])} "
                    f"phase={int(mgr.phase[e])} "
                    f"hold={int(mgr.hold_steps[e])} move={float(mgr.move_dist[e]):.3f} "
                    f"width={float(width_now[e])*1000:.1f}mm "
                    f"piece_z-rest={float(piece[e, 2]) - rest_z:.3f} attached={att}",
                    flush=True,
                )
        if saved >= args.max_episodes:
            print("[rec] max episodes reached — stopping", flush=True)
            break

    manifest.close()
    print(f"[rec] ======== RESULT: {saved} episodes → {out_root} ========", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

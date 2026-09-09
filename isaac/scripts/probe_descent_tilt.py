"""Does the cup tip DURING the descent, or fail the release some other way?

E27c measured that only 38.9% of upright plate arrivals ever place — 91 of
149 fail after arriving in a placeable state, and nothing recorded at grasp
time distinguishes them. `release_at_zone()` requires upright at the moment
of release, but nothing in the reward asks the cup to STAY upright while it
is lowered, and the cup is kinematically attached so it inherits every
degree of wrist rotation on the way down.

This probe tests that directly, and — just as importantly — tests the
alternatives, so a confirmation is not simply the first story that fits.
For every episode that arrives at the plate upright it records, at three
moments (arrival / lowest cup-z of the descent / last step in SETDOWN):

    tilt        degrees off vertical
    cup_z       height above the resting height
    speed       for the `gentle` clause
    attached    for the `released` clause
    at_target   for the placement-radius clause

Then it partitions the FAILURES by which clause was never satisfiable:

    TIPPED        arrived upright, tilt exceeded the gate during descent
    NEVER_LOWERED never got within the resting band
    NOT_GENTLE    reached the band but was still moving
    NEVER_RELEASED reached a placeable state and stayed attached
    OTHER         none of the above

A separate file on purpose: eval_synria_sequence.py is the committed metric
source and was locked mid-E29 (its control and treatment arms must be
measured by one build). This probe touches neither it nor the reward.

    ~/.venv/isaacsim5/bin/python isaac/scripts/probe_descent_tilt.py \
        --headless --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

UPRIGHT_COS = 0.966          # cos(15 deg) — the gate release_at_zone uses
RESTING_BAND_M = 0.02        # |cup_z - rest_z| <= this counts as resting
GENTLE_MAX_VEL = 0.10        # m/s
PLACEMENT_RADIUS_M = 0.15


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(prog="probe_descent_tilt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1700)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
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
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG
    from isaac.isaaclab_tasks.synria_pickplace.trial_state import TaskPhase

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

    env = raw.unwrapped
    mgr = mdp._get_trial_mgr(env)
    dev, n = env.device, env.num_envs
    rest_z = TABLE_SURFACE_Z + 0.05 / 2.0
    zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731

    arrived = zb()                                    # arrived at plate upright
    placed = zb()
    tilt_arr = torch.zeros(n, device=dev)             # tilt at that arrival
    tilt_max = torch.zeros(n, device=dev)             # worst tilt during SETDOWN
    min_z = torch.full((n,), 1e9, device=dev)         # lowest cup z in SETDOWN
    tilt_at_lowz = torch.zeros(n, device=dev)
    ever_resting = zb()
    ever_rest_gentle = zb()
    ever_placeable = zb()                             # resting+gentle+upright+at_target
    ever_detached_in_setdown = zb()
    rows: list = []
    prev_len = env.episode_length_buf.clone()
    prev_phase = mgr.phase.clone()

    obs = wrapper.get_observations()
    for step in range(args.steps):
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = wrapper.step(act)

        piece = mdp._get_piece_pos(env)
        q = env.scene["piece"].data.root_quat_w
        up_z = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0)
        tilt = torch.rad2deg(torch.arccos(up_z))
        upright = up_z >= UPRIGHT_COS
        vel = torch.norm(env.scene["piece"].data.root_lin_vel_w, dim=-1)
        phase = mgr.phase
        in_setdown = phase == int(TaskPhase.PICK_FROM_ZONE)
        at_target = torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1) <= PLACEMENT_RADIUS_M
        lifted = mdp._piece_lifted_mask(env)

        # Arrival: first moment at the plate, lifted, upright.
        arrive_now = in_setdown & lifted & at_target & upright & ~arrived
        if bool(arrive_now.any()):
            arrived |= arrive_now
            tilt_arr = torch.where(arrive_now, tilt, tilt_arr)
            tilt_max = torch.where(arrive_now, tilt, tilt_max)

        # Track the descent only for envs that arrived upright.
        track = arrived & in_setdown
        tilt_max = torch.where(track, torch.maximum(tilt_max, tilt), tilt_max)
        lower = track & (piece[:, 2] < min_z)
        min_z = torch.where(lower, piece[:, 2], min_z)
        tilt_at_lowz = torch.where(lower, tilt, tilt_at_lowz)

        resting = (piece[:, 2] - rest_z).abs() <= RESTING_BAND_M
        ever_resting |= track & resting
        ever_rest_gentle |= track & resting & (vel <= GENTLE_MAX_VEL)
        ever_placeable |= track & resting & (vel <= GENTLE_MAX_VEL) & upright & at_target
        ever_detached_in_setdown |= track & ~mgr.attached

        placed |= (prev_phase == int(TaskPhase.PICK_FROM_ZONE)) & (
            phase == int(TaskPhase.RETURN_TO_BOARD)
        )
        prev_phase = phase.clone()

        ep_len = env.episode_length_buf
        done = ep_len < prev_len
        if bool(done.any()):
            for i in torch.nonzero(done & arrived).squeeze(-1).tolist():
                rows.append({
                    "placed": bool(placed[i]),
                    "tilt_arrival_deg": round(float(tilt_arr[i]), 2),
                    "tilt_max_setdown_deg": round(float(tilt_max[i]), 2),
                    "tilt_at_lowest_z_deg": round(float(tilt_at_lowz[i]), 2),
                    "lowest_z_above_rest_mm": round(
                        (float(min_z[i]) - rest_z) * 1000, 1
                    ) if float(min_z[i]) < 1e8 else None,
                    "ever_resting": bool(ever_resting[i]),
                    "ever_rest_gentle": bool(ever_rest_gentle[i]),
                    "ever_placeable": bool(ever_placeable[i]),
                    "ever_detached": bool(ever_detached_in_setdown[i]),
                })
            for t in (arrived, placed, ever_resting, ever_rest_gentle,
                      ever_placeable, ever_detached_in_setdown):
                t &= ~done
            min_z = torch.where(done, torch.full_like(min_z, 1e9), min_z)
        prev_len = ep_len.clone()
        if step % 500 == 0:
            print(f"[probe] step={step} arrivals={len(rows)}", flush=True)

    # ---- partition the failures -------------------------------------------
    ok = [r for r in rows if r["placed"]]
    bad = [r for r in rows if not r["placed"]]

    def cause(r: dict) -> str:
        if r["tilt_max_setdown_deg"] > 15.0:
            return "TIPPED"
        if not r["ever_resting"]:
            return "NEVER_LOWERED"
        if not r["ever_rest_gentle"]:
            return "NOT_GENTLE"
        if r["ever_placeable"] and not r["ever_detached"]:
            return "NEVER_RELEASED"
        return "OTHER"

    counts: dict = {}
    for r in bad:
        counts[cause(r)] = counts.get(cause(r), 0) + 1

    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else float("nan")

    print("[probe] ======== DESCENT TILT ========", flush=True)
    print(f"[probe] upright arrivals: {len(rows)}  placed: {len(ok)}  "
          f"failed: {len(bad)}", flush=True)
    print(f"[probe] tilt at arrival      median {med([r['tilt_arrival_deg'] for r in rows]):.1f} deg",
          flush=True)
    print(f"[probe] tilt at lowest cup-z median {med([r['tilt_at_lowest_z_deg'] for r in rows]):.1f} deg",
          flush=True)
    print(f"[probe] worst tilt in setdown median {med([r['tilt_max_setdown_deg'] for r in rows]):.1f} deg",
          flush=True)
    for nm, grp in (("PLACED", ok), ("FAILED", bad)):
        if grp:
            print(f"[probe]   {nm}: arrival {med([r['tilt_arrival_deg'] for r in grp]):.1f} -> "
                  f"lowest-z {med([r['tilt_at_lowest_z_deg'] for r in grp]):.1f} -> "
                  f"worst {med([r['tilt_max_setdown_deg'] for r in grp]):.1f} deg | "
                  f"lowest z above rest {med([r['lowest_z_above_rest_mm'] for r in grp]):.1f} mm",
                  flush=True)
    print("[probe] FAILURE CAUSES:", flush=True)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"[probe]   {k:<15} {v:>4}  ({v / max(1, len(bad)):.1%})", flush=True)

    out = args.out or (_REPO_ROOT / "reports/eval"
                       / f"descent_tilt_{Path(args.checkpoint).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "upright_arrivals": len(rows), "placed": len(ok), "failed": len(bad),
        "failure_causes": counts, "episodes": rows,
    }, indent=2) + "\n")
    print(f"[probe] report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

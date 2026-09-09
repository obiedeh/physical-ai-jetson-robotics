"""Probe: why doesn't grasp_confirmed fire in staged in-hand episodes?

Resets a small env batch and prints, for the first N steps of each of the
first few envs: gripper width, piece height above rest, tool0-piece
distance, trial phase, consec counter, and the grasp reward — the raw
inputs to every grasp criterion. No inference; direct observation.

Usage:
    $ISAAC_PYTHON isaac/scripts/diag_grasp_confirm.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(prog="diag_grasp_confirm")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Drive with a trained policy instead of zero actions.")
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args.checkpoint)
    finally:
        app.close()


def _run(checkpoint: str | None = None) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PIECE_HEIGHT_M,
        TABLE_SURFACE_Z,
    )

    env_cfg = parse_env_cfg("Synria-Chess-PickPlace-v0", device="cuda", num_envs=8)
    env_cfg.seed = 123
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped

    policy = None
    wrapper = None
    if checkpoint:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
        from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

        from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG

        wrapper = RslRlVecEnvWrapper(raw)
        rc = TASK_TRAIN_CFG["Synria-Chess-PickPlace-v0"]()
        runner = OnPolicyRunner(wrapper, rc.to_dict(), log_dir=None, device="cuda")
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device="cuda")
        print(f"[probe] policy loaded from {checkpoint}")
    else:
        env.reset()

    mgr = mdp._get_trial_mgr(env)
    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    zero_action = torch.zeros((8, 8), device=env.device)

    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    print("[probe] joint_names:", list(robot.joint_names))
    print("[probe] body_names :", list(robot.body_names))
    print("[probe] rids arm:", rids["arm"], "left:", rids["left"],
          "right:", rids["right"], "tool0:", rids["tool0"])
    print("[probe] raw tool0 pos env0:", robot.data.body_pos_w[0, rids["tool0"]].tolist())
    print("[probe] raw piece pos env0:", mdp._get_piece_pos(env)[0].tolist())
    print("[probe] default_joint_pos env0:", robot.data.default_joint_pos[0].tolist())
    print("[probe] phases after reset:", mgr.phase.tolist())
    obs = wrapper.get_observations() if wrapper is not None else None
    for step in range(130):
        width = mdp._gripper_width_m(env).squeeze(-1)
        piece = mdp._get_piece_pos(env)
        tool = mdp._tool0_pos_local(env)
        dist = torch.norm(tool - piece, dim=-1)
        zdist = torch.norm(piece[:, :2] - mgr.zone_xy, dim=-1)
        if step % 10 == 0 or step in (29, 30, 31, 32):
            for i in range(4):
                print(
                    f"[probe] step={step} env={i} phase={int(mgr.phase[i])} "
                    f"width={float(width[i]) * 1000:6.1f}mm "
                    f"lift={float(piece[i, 2] - rest_z) * 1000:6.1f}mm "
                    f"dist={float(dist[i]) * 1000:6.1f}mm "
                    f"zone={float(zdist[i]) * 1000:6.1f}mm "
                    f"pin={int(mgr.pin_steps[i])}"
                )
        if policy is not None:
            with torch.inference_mode():
                act = policy(obs)
            obs, _, _, _ = wrapper.step(act)
        else:
            env.step(zero_action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded probe: boot the recorder env variant, verify camera sensors.

Checks (GR00T ledger gate B2):
    1. Env constructs with wrist + overhead TiledCameras (RL obs unchanged).
    2. Cameras deliver 224x224x3 RGB tensors.
    3. Dumps first-env frames to PNG for visual mount verification.

Usage:
    $ISAAC_PYTHON isaac/scripts/diag_recorder_cameras.py --headless \
        [--num_envs 2] [--steps 30] [--out /tmp/recorder_cam_probe]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("/tmp/recorder_cam_probe"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True  # tiled cameras need the render pipeline
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import-not-found]

    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
        SynriaLudoPickPlaceRecorderEnvCfg,
    )

    cfg = SynriaLudoPickPlaceRecorderEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()

    policy_obs_dim = obs["policy"].shape[-1]
    print(f"[probe] policy obs dim = {policy_obs_dim} (must equal RL task's dim)")

    zero_action = torch.zeros(env.num_envs, 8, device=env.device)
    for _ in range(args.steps):
        obs, *_ = env.step(zero_action)

    args.out.mkdir(parents=True, exist_ok=True)
    ok = True
    for key in ("wrist", "overhead"):
        cam = env.scene.sensors[key]
        rgb = cam.data.output["rgb"]  # (num_envs, H, W, 3|4)
        print(f"[probe] {key}: shape={tuple(rgb.shape)} dtype={rgb.dtype}")
        if rgb.shape[1:3] != (224, 224):
            print(f"[probe] FAIL: {key} resolution != 224x224")
            ok = False
        frame = rgb[0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frame = frame[..., :3]
        try:
            from PIL import Image

            Image.fromarray(frame).save(args.out / f"{key}.png")
            print(f"[probe] wrote {args.out / (key + '.png')}")
        except ImportError:
            np.save(args.out / f"{key}.npy", frame)
            print(f"[probe] PIL missing; wrote {args.out / (key + '.npy')}")
        if frame.std() < 1.0:
            print(f"[probe] WARN: {key} frame is near-constant (std={frame.std():.3f}) — "
                  "camera may face nothing")
    print(f"[probe] RESULT: {'PASS' if ok else 'FAIL'}")

    env.close()
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage-1 perception probe: SEE the cup, KNOW its position and size.

From the overhead camera only (no privileged state), per env:
    1. Segment the red cup (HSV threshold).
    2. Pixel centroid + apparent radius.
    3. Back-project to env-local world XY using the known top-down camera
       pose (pos (0.10, 0, table+0.85), focal 18 mm, aperture 20.955 mm,
       224 px) and the known cup rest height.
    4. Estimate diameter from apparent radius; gripper aperture = Ø + margin.

Scores estimates against simulator ground truth and PASSes when the mean
XY error < 2 cm and diameter error < 8 mm (good enough to grasp a 40 mm
cup with a 50 mm-stroke gripper).

    ~/.venv/isaacsim5/bin/python isaac/scripts/diag_cup_perception.py \
        --headless [--num_envs 4] [--steps 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Overhead camera model (recorder_env_cfg): env-local pose + pinhole.
CAM_XY = (0.10, 0.0)
CAM_HEIGHT_ABOVE_TABLE = 0.85
FOCAL_MM = 18.0
APERTURE_MM = 20.955  # Isaac default horizontal aperture
RES = 224


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
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
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PIECE_RADIUS_M,
        TABLE_SURFACE_Z,
    )

    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg(
        "Synria-Chess-PickPlace-v0", device="cuda", num_envs=args.num_envs
    )
    env_cfg.seed = 123
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    add_recorder_cameras(env_cfg)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    env.reset()
    n = env.num_envs

    # meters-per-pixel at the cup's rest depth (top-down camera).
    depth = CAM_HEIGHT_ABOVE_TABLE  # cup top ≈ table level vs camera height
    view_w = 2.0 * depth * (APERTURE_MM / 2.0 / FOCAL_MM)
    mpp = view_w / RES
    print(f"[percept] view width {view_w:.3f} m, {mpp*1000:.2f} mm/px", flush=True)

    zero = torch.zeros(n, 8, device=env.device)
    xy_errs, d_errs, detects = [], [], 0
    for step in range(args.steps):
        env.step(zero)
        if step < 10:  # let the scene settle and renderer warm up
            continue
        rgb = env.scene.sensors["overhead"].data.output["rgb"].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        gt = mdp._get_piece_pos(env).cpu().numpy()  # env-local
        for e in range(n):
            bgr = cv2.cvtColor(rgb[e][..., :3], cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | cv2.inRange(
                hsv, (168, 80, 60), (180, 255, 255)
            )
            m = cv2.moments(mask)
            if m["m00"] < 10:
                continue
            detects += 1
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            area = float((mask > 0).sum())
            radius_px = float(np.sqrt(area / np.pi))
            # Top-down mapping (verified by this probe): image up (−v) is
            # env +X, image left (−u) is env +Y.
            x = CAM_XY[0] + (RES / 2.0 - v) * mpp
            y = CAM_XY[1] + (RES / 2.0 - u) * mpp
            d_est = 2.0 * radius_px * mpp
            xy_errs.append(float(np.hypot(x - gt[e, 0], y - gt[e, 1])))
            d_errs.append(abs(d_est - 2.0 * PIECE_RADIUS_M))

    xy_errs, d_errs = np.array(xy_errs), np.array(d_errs)
    print(f"[percept] detections: {detects}", flush=True)
    if len(xy_errs) == 0:
        print("[percept] RESULT: FAIL — no detections", flush=True)
        return 1
    print(
        f"[percept] XY error mean {xy_errs.mean()*100:.2f} cm "
        f"(p95 {np.percentile(xy_errs, 95)*100:.2f} cm)",
        flush=True,
    )
    print(
        f"[percept] diameter error mean {d_errs.mean()*1000:.1f} mm "
        f"(true {2*PIECE_RADIUS_M*1000:.0f} mm)",
        flush=True,
    )
    ok = xy_errs.mean() < 0.02 and d_errs.mean() < 0.008
    print(f"[percept] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

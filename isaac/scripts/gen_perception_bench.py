"""Render the G1 perception benchmark set: overhead frames at KNOWN cup poses.

Layer contract (consumed by the Gemini layer's scorer, which never touches
Isaac): a directory of overhead_XXX.png plus truth.jsonl rows
    {"i", "cup_world_xy_mm", "cup_px", "mm_per_px"}
cup_px is the cup centre projected through the overhead camera's measured
intrinsics + pose; mm_per_px is the local scale at the cup, from projecting
a 10 mm offset. Verify the projection VISUALLY (crosshair overlay on the
first 3 frames) before trusting the truth file — this project's standing
lesson about instruments.

    ~/.venv/isaacsim5/bin/python isaac/scripts/gen_perception_bench.py --headless \
        [--n 60] [--out reports/perception_bench]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--marker", action="store_true",
                   help="spawn a visible magenta zone-marker disc, randomize "
                   "its pose per frame, and record its own segmented pixel "
                   "truth (G2v2: direct-pixel zone truth, no affine)")
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "reports/perception_bench")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        return _run(args)
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush()
        import os; os._exit(1)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from PIL import Image, ImageDraw
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab.utils.math import quat_conjugate, quat_apply  # type: ignore
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import add_recorder_cameras
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z

    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0

    env_cfg = parse_env_cfg("Synria-Chess-PickPlace-v0", device="cuda", num_envs=1)
    env_cfg.seed = 77
    if args.marker:
        # Visual-only zone marker: thin kinematic disc, no collider, bright
        # MAGENTA — chromatically disjoint from the red cup so each has its
        # own single-channel segmentation truth.
        from isaaclab.assets import RigidObjectCfg  # type: ignore[import-not-found]
        env_cfg.scene.zone_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ZoneMarker",
            spawn=sim_utils.CylinderCfg(
                radius=0.045,
                height=0.003,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.05, 0.9)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.20, 0.0)
            ),
        )
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    add_recorder_cameras(env_cfg)
    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    env.reset()
    cup = env.scene["piece"]
    cam = env.scene.sensors["overhead"]
    dev = env.device
    marker = env.scene["zone_marker"] if args.marker else None

    args.out.mkdir(parents=True, exist_ok=True)
    truth = (args.out / "truth.jsonl").open("w")

    def red_centroid(rgb: "np.ndarray"):
        """Truth from the image itself: the cup is the scene's only bright-red
        object (piece diffuse 0.85,0.1,0.1; board wood, arm near-black). No
        camera-convention math to get wrong — and the crosshair overlays
        verify it by eye before anything is scored."""
        r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
        mask = (r > 120) & (r > g + 60) & (r > b + 60)
        if mask.sum() < 12:
            return None, 0
        ys, xs = np.nonzero(mask)
        return (float(xs.mean()), float(ys.mean())), int(mask.sum())

    def magenta_centroid(rgb: "np.ndarray"):
        r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
        mask = (r > 120) & (b > 120) & (r > g + 60) & (b > g + 60)
        if mask.sum() < 12:
            return None, 0
        ys, xs = np.nonzero(mask)
        return (float(xs.mean()), float(ys.mean())), int(mask.sum())

    rng = np.random.default_rng(7)
    step0 = torch.zeros(1, env.action_manager.total_action_dim, device=dev)
    saved = 0
    while saved < args.n:
        # random pose on the board, generous workspace
        wx = float(rng.uniform(0.06, 0.46))
        wy = float(rng.uniform(-0.26, 0.26))
        root = cup.data.default_root_state.clone()
        root[0, 0], root[0, 1] = wx, wy
        root[0, 2] = TABLE_SURFACE_Z + 0.025 + 0.001
        root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=dev)
        root[0, 7:] = 0.0
        cup.write_root_pose_to_sim(root[:, :7])
        cup.write_root_velocity_to_sim(root[:, 7:])
        if marker is not None:
            # marker at least 0.12 m from the cup, same workspace
            while True:
                mx = float(rng.uniform(0.06, 0.46))
                my = float(rng.uniform(-0.26, 0.26))
                if ((mx - wx) ** 2 + (my - wy) ** 2) ** 0.5 > 0.12:
                    break
            mroot = marker.data.default_root_state.clone()
            mroot[0, 0], mroot[0, 1] = mx, my
            mroot[0, 2] = TABLE_SURFACE_Z + 0.0015 + 0.0005
            mroot[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=dev)
            mroot[0, 7:] = 0.0
            marker.write_root_pose_to_sim(mroot[:, :7])
            marker.write_root_velocity_to_sim(mroot[:, 7:])
        for _ in range(12):
            env.step(step0)
        cw = cup.data.root_pos_w[0]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb = rgb[..., :3]
        px, blob = red_centroid(rgb)
        if px is None:
            continue
        # mm/px scale: shift the cup +20 mm in x, re-render, measure the
        # centroid displacement — empirical, per-sample
        root2 = root.clone(); root2[0, 0] += 0.020
        cup.write_root_pose_to_sim(root2[:, :7]); cup.write_root_velocity_to_sim(root2[:, 7:])
        for _ in range(6):
            env.step(step0)
        rgb2 = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb2.dtype != np.uint8:
            rgb2 = (np.clip(rgb2, 0, 1) * 255).astype(np.uint8)
        px2, _ = red_centroid(rgb2[..., :3])
        if px2 is None:
            continue
        dpix = ((px[0]-px2[0])**2 + (px[1]-px2[1])**2) ** 0.5
        if dpix < 1.5:
            continue
        mm_per_px = 20.0 / dpix
        img = Image.fromarray(rgb)
        H, W = rgb.shape[0], rgb.shape[1]
        img.save(args.out / f"overhead_{saved:03d}.png")
        if saved < 3:   # visual verification overlays
            ov = img.copy(); d = ImageDraw.Draw(ov)
            u, v = px
            d.line([(u-12, v), (u+12, v)], fill=(0, 255, 0), width=2)
            d.line([(u, v-12), (u, v+12)], fill=(0, 255, 0), width=2)
            if marker is not None:
                vpx, _ = magenta_centroid(rgb)
                if vpx is not None:
                    mu, mv = vpx
                    d.line([(mu-12, mv), (mu+12, mv)], fill=(0, 255, 255), width=2)
                    d.line([(mu, mv-12), (mu, mv+12)], fill=(0, 255, 255), width=2)
            ov.save(args.out / f"verify_{saved:03d}.png")
        zone = mdp._get_trial_mgr(env).zone_xy[0]
        marker_fields = {}
        if marker is not None:
            mpx, mblob = magenta_centroid(rgb)
            if mpx is None:
                continue
            mw = marker.data.root_pos_w[0]
            marker_fields = {
                "marker_world_xy_mm": [round(float(mw[0])*1000, 1),
                                       round(float(mw[1])*1000, 1)],
                "marker_px": [round(mpx[0], 1), round(mpx[1], 1)],
                "marker_blob_px": mblob,
            }
        truth.write(json.dumps({
            "i": saved,
            **marker_fields,
            "cup_world_xy_mm": [round(float(cw[0])*1000, 1), round(float(cw[1])*1000, 1)],
            "zone_world_xy_mm": [round(float(zone[0])*1000, 1), round(float(zone[1])*1000, 1)],
            "cup_px": [round(px[0], 1), round(px[1], 1)],
            "mm_per_px": round(mm_per_px, 3),
            "blob_px": blob,
            "image_wh": [W, H],
        }) + "\n")
        saved += 1
        if saved % 20 == 0:
            print(f"[bench] {saved}/{args.n}", flush=True)
    truth.close()
    print(f"[bench] wrote {saved} samples to {args.out}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

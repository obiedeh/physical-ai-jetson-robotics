"""Render overhead frames of Ludo board states for the board-reading benchmark.

Runs inside Isaac Sim 5.1's Python on the Linux RTX workstation:

    ~/.venv/isaacsim5/bin/python isaac/scripts/render_ludo_board_bench.py \
        --out reports/ludo_board_bench --n 30 --seed 2026

Draws the 15x15 board of `ludo_engine/board.py` (track, home columns, yards),
places the 16 tokens of each sampled state, and saves one frame per state with
`truth.jsonl`. No arm, no physics: this is a perception input, not a game run.
The camera and the grid come from `physical_ai_lab.gemini_robotics.ludo_board`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from ludo_engine.board import _HOME_GRID, _TRACK_GRID, _YARD_GRID  # noqa: E402
from ludo_engine.game import START  # noqa: E402
from physical_ai_lab.gemini_robotics.ludo_board import (  # noqa: E402
    COLORS, GRID, TOKEN_HEIGHT, OverheadCamera, sample_states,
)

TABLE_Z = 0.0
BOARD_THICKNESS = 0.005
BOARD_TOP = TABLE_Z + BOARD_THICKNESS
TOKEN_RADIUS = 0.011
RGB = {"red": (0.80, 0.05, 0.05), "blue": (0.05, 0.15, 0.80),
       "green": (0.05, 0.50, 0.12), "yellow": (0.90, 0.70, 0.02)}


def tint(rgb, amount):
    return tuple(c + (1.0 - c) * amount for c in rgb)


def box(stage, path, size, pos, rgb):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*pos))
    api.SetScale(Gf.Vec3f(*size))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return prim


def build(stage, cam: OverheadCamera):
    geo = cam.geometry
    cell = geo.board_size / GRID
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(500.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(20.0, 15.0, 0.0))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(280.0)
    box(stage, "/World/Table", (1.2, 0.8, 0.04), (geo.center_x, geo.center_y, TABLE_Z - 0.02), (0.35, 0.35, 0.35))
    box(stage, "/World/Board", (geo.board_size, geo.board_size, BOARD_THICKNESS),
        (geo.center_x, geo.center_y, TABLE_Z + BOARD_THICKNESS / 2), (0.12, 0.12, 0.12))

    def tile(name, col, row, rgb, span=1.0, lift=0.0):
        x, y = geo._grid_to_world(col, row)
        side = cell * span - 0.0016
        box(stage, f"/World/Tiles/{name}", (side, side, 0.0006), (x, y, BOARD_TOP + 0.0003 + lift), rgb)

    UsdGeom.Xform.Define(stage, "/World/Tiles")
    start_of = {sq: c for c, sq in START.items()}
    for i, (col, row) in enumerate(_TRACK_GRID):
        rgb = tint(RGB[start_of[i]], 0.55) if i in start_of else (0.93, 0.93, 0.90)
        tile(f"track_{i}", col, row, rgb)
    for color, cells in _HOME_GRID.items():
        for j, (col, row) in enumerate(cells):
            tile(f"home_{color}_{j}", col, row, tint(RGB[color], 0.55))
    for color, cells in _YARD_GRID.items():
        col = sum(c for c, _ in cells) / 4.0
        row = sum(r for _, r in cells) / 4.0
        x = geo.center_x + (row - 7) * cell
        y = geo.center_y + (col - 7) * cell
        box(stage, f"/World/Tiles/yard_{color}", (cell * 6 - 0.002, cell * 6 - 0.002, 0.0006),
            (x, y, BOARD_TOP + 0.0003), tint(RGB[color], 0.35))
        for j, (c, r) in enumerate(cells):
            tile(f"yard_{color}_{j}", c, r, (0.93, 0.93, 0.90), lift=0.0006)

    UsdGeom.Xform.Define(stage, "/World/Tokens")
    tokens = {}
    for color in COLORS:
        for t in range(4):
            cyl = UsdGeom.Cylinder.Define(stage, f"/World/Tokens/{color}_{t}")
            cyl.CreateRadiusAttr(TOKEN_RADIUS)
            cyl.CreateHeightAttr(TOKEN_HEIGHT)
            cyl.CreateAxisAttr("Z")
            cyl.CreateDisplayColorAttr([Gf.Vec3f(*RGB[color])])
            UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(0, 0, -1))
            tokens[(color, t)] = cyl

    camera = UsdGeom.Camera.Define(stage, "/World/OverheadCamera")
    camera.CreateFocalLengthAttr(cam.focal_mm)
    camera.CreateHorizontalApertureAttr(cam.aperture_mm)
    camera.CreateVerticalApertureAttr(cam.aperture_mm * cam.image_wh[1] / cam.image_wh[0])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(geo.center_x, geo.center_y, BOARD_TOP + cam.distance))
    return tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "ludo_board_bench")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size", type=int, default=672)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cam = OverheadCamera((args.size, args.size))
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    tokens = build(stage, cam)
    product = rep.create.render_product("/World/OverheadCamera", cam.image_wh)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for _ in range(20):
        sim_app.update()

    states = sample_states(args.n, args.seed)
    with (args.out / "truth.jsonl").open("w") as fh:
        for state in states:
            for tok in state["tokens"]:
                col, row = tok["cell"]
                x, y = cam.geometry._grid_to_world(col, row)
                z = BOARD_TOP + TOKEN_HEIGHT / 2 + tok["stack"] * TOKEN_HEIGHT
                UsdGeom.XformCommonAPI(tokens[(tok["color"], tok["token"])]).SetTranslate(Gf.Vec3d(x, y, z))
                tok["px"] = [round(v, 1) for v in cam.cell_to_px(col, row, TOKEN_HEIGHT * (tok["stack"] + 1))]
            for _ in range(4):
                rep.orchestrator.step(rt_subframes=16)
            frame = rgb.get_data()
            Image.fromarray(frame[..., :3]).save(args.out / f"board_{state['i']:03d}.png")
            state["image"] = f"board_{state['i']:03d}.png"
            state["image_wh"] = list(cam.image_wh)
            state["camera"] = {"focal_mm": cam.focal_mm, "aperture_mm": cam.aperture_mm, "distance_m": cam.distance}
            fh.write(json.dumps(state) + "\n")
            print(f"rendered {state['image']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - needs Isaac Sim
    try:
        code = main()
    finally:
        sim_app.close()
    sys.exit(code)

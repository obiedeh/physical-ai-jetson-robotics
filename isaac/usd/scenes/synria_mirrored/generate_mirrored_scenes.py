#!/usr/bin/env python3
"""Generate left-side (board-mirrored) variants of the Alicia-D no-robot scenes.

Each output is a thin USD that references the original scene's ``/World`` and
adds a single ``+GAME_X_SHIFT`` translate to the ``/World/Game`` group, moving
the board to the +X end of the table. Everything else (table, floor, cameras,
lights) composes through unchanged.

This keeps the board position *baked* — so it spawns and clones cleanly across
all training envs — while staying a one-line override rather than a full copy
of the geometry. It matches the runtime shift the interactive viewer applies in
``scripts/view_synria_scene.py`` (GAME_X_SHIFT), so the trained env and the
viewer share the same left-side layout for all three games.

Run with the Isaac Sim venv python (for ``pxr``):
    ~/.venv/isaacsim5/bin/python isaac/usd/scenes/synria_mirrored/generate_mirrored_scenes.py
"""
from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdGeom

ALICIA_SCENES_DIR = Path(
    "/home/oedeh/github/Alicia-D-ROS2/scripts/isaac_sim/scenes"
)
OUT_DIR = Path(__file__).resolve().parent
# Each board is mirrored across table centre (world X = 0): a board centred at
# X = b moves to -b, i.e. a translate of -2b. This preserves each game's
# arm->board reach when the arm mount also flips -0.1524 -> +0.1524, and works
# per-game (chess board sits at -0.508, ludo at ~-0.452, etc.).

SCENES = {
    "ludo": "ludotable_norobot.usda",
    "chess": "chesstable_norobot.usda",
    "checkers": "checkerstable_norobot.usda",
}

TEMPLATE = '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Board mirrored to +X (camera-left). Auto-generated; do not hand-edit."
)

def Xform "World" (
    prepend references = @{orig}@</World>
)
{{
    over "Game"
    {{
        double3 xformOp:translate = ({sx}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
'''


def _board_x(usd_path: Path) -> float:
    """World-space X of the /World/Game bounding-box centre for a scene USD."""
    stage = Usd.Stage.Open(str(usd_path))
    bb = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Game")).ComputeWorldBound(
        Usd.TimeCode.Default(), "default"
    )
    return bb.ComputeAlignedBox().GetMidpoint()[0]


def main() -> None:
    for game, fname in SCENES.items():
        orig = ALICIA_SCENES_DIR / fname
        if not orig.is_file():
            raise SystemExit(f"Source scene missing: {orig}")
        board_x = _board_x(orig)
        shift = -2.0 * board_x  # mirror the board across world X = 0
        out = OUT_DIR / fname.replace("_norobot", "_norobot_left")
        out.write_text(TEMPLATE.format(orig=orig, sx=shift))

        # Verify the composed board landed at the mirror position (-board_x).
        mirrored_x = _board_x(out)
        print(f"{game:9s} -> {out.name}  board X {board_x:+.4f} -> "
              f"{mirrored_x:+.4f} (shift {shift:+.4f})")
        assert abs(mirrored_x + board_x) < 1e-3, (
            f"{game}: board not mirrored (orig {board_x}, got {mirrored_x})"
        )
    print("All mirrored scenes generated and verified.")


if __name__ == "__main__":
    main()

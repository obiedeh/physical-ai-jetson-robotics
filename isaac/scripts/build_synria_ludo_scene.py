"""Build the Synria-arm + Ludo tabletop training scene.

Runs inside Isaac Sim 5.1's Python on the Linux RTX 5090 workstation.
Output: ``isaac/usd/scenes/synria_ludo/synria_ludo_v0.usda``.

Scene layout (meters, Z-up):

- Floor, table, Synria arm (shared scaffolding from ``_synria_scene_common``)
- Ludo board: 0.40 x 0.40 x 0.005 m, centered at (0.10, 0)
  with four colored home quadrants (red, blue, green, yellow)
- 16 tokens (4 per color) staged in the four home corners
- 1 die placed beside the board
- Overhead RGB camera (board-state view) + operator camera (review view)

Run on RTX:

    bash scripts/linux_rtx/build_synria_ludo_scene.sh
"""

from __future__ import annotations

from _synria_scene_common import (  # noqa: I001  (boots SimulationApp on import)
    BOARD_CENTER_X,
    BOARD_CENTER_Y,
    REPO_ROOT,
    TABLE_SURFACE_Z,
    UsdGeom,
    add_staging_zones,
    build_common_scene,
    create_new_stage,
    ensure_arm_usd_exists,
    make_box,
    make_cube,
    make_cylinder,
    save_stage,
    sim_app,
)

SCENE_OUT = REPO_ROOT / "isaac" / "usd" / "scenes" / "synria_ludo" / "synria_ludo_v0.usda"

# Board layout
BOARD_SIZE = 0.40
BOARD_THICKNESS = 0.005
BOARD_TOP_Z = TABLE_SURFACE_Z + BOARD_THICKNESS / 2.0

# Token layout
TOKEN_RADIUS = 0.012
TOKEN_HEIGHT = 0.020
TOKEN_Z = BOARD_TOP_Z + BOARD_THICKNESS / 2.0 + TOKEN_HEIGHT / 2.0

_HOME_QUADRANT_OFFSET = 0.14
_HOME_TOKEN_SPREAD = 0.04
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.10, 0.10),
    "blue": (0.10, 0.20, 0.85),
    "green": (0.10, 0.65, 0.20),
    "yellow": (0.95, 0.80, 0.10),
}
_HOME_CORNERS: dict[str, tuple[float, float]] = {
    "red": (-_HOME_QUADRANT_OFFSET, +_HOME_QUADRANT_OFFSET),
    "blue": (+_HOME_QUADRANT_OFFSET, +_HOME_QUADRANT_OFFSET),
    "green": (-_HOME_QUADRANT_OFFSET, -_HOME_QUADRANT_OFFSET),
    "yellow": (+_HOME_QUADRANT_OFFSET, -_HOME_QUADRANT_OFFSET),
}
DIE_SIZE = 0.018
DIE_POS = (BOARD_CENTER_X + BOARD_SIZE / 2.0 + 0.05, BOARD_CENTER_Y, BOARD_TOP_Z + DIE_SIZE / 2.0)


def _add_ludo_board(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/LudoBoard")
    make_box(
        stage,
        "/World/LudoBoard/Base",
        size=(BOARD_SIZE, BOARD_SIZE, BOARD_THICKNESS),
        position=(BOARD_CENTER_X, BOARD_CENTER_Y, BOARD_TOP_Z),
        rgb=(0.95, 0.95, 0.92),
        rigid=False,
    )
    quadrant_size = BOARD_SIZE * 0.40
    quadrant_z = BOARD_TOP_Z + BOARD_THICKNESS / 2.0 + 0.0005
    for color_name, (dx, dy) in _HOME_CORNERS.items():
        make_box(
            stage,
            f"/World/LudoBoard/Quadrant_{color_name}",
            size=(quadrant_size, quadrant_size, 0.0008),
            position=(BOARD_CENTER_X + dx, BOARD_CENTER_Y + dy, quadrant_z),
            rgb=COLORS[color_name],
            rigid=False,
        )


def _add_ludo_tokens(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/LudoTokens")
    for color_name, (dx, dy) in _HOME_CORNERS.items():
        cx = BOARD_CENTER_X + dx
        cy = BOARD_CENTER_Y + dy
        for i, (ox, oy) in enumerate([
            (-_HOME_TOKEN_SPREAD, -_HOME_TOKEN_SPREAD),
            (+_HOME_TOKEN_SPREAD, -_HOME_TOKEN_SPREAD),
            (-_HOME_TOKEN_SPREAD, +_HOME_TOKEN_SPREAD),
            (+_HOME_TOKEN_SPREAD, +_HOME_TOKEN_SPREAD),
        ]):
            make_cylinder(
                stage,
                f"/World/LudoTokens/{color_name}_{i + 1}",
                radius=TOKEN_RADIUS,
                height=TOKEN_HEIGHT,
                position=(cx + ox, cy + oy, TOKEN_Z),
                rgb=COLORS[color_name],
            )


def _add_die(stage) -> None:
    make_cube(
        stage,
        "/World/LudoDie",
        size=DIE_SIZE,
        position=DIE_POS,
        rgb=(0.95, 0.95, 0.95),
        rigid=True,
    )


def build():
    ensure_arm_usd_exists()
    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)

    stage = create_new_stage()
    build_common_scene(stage, board_top_z=BOARD_TOP_Z)
    _add_ludo_board(stage)
    _add_ludo_tokens(stage)
    _add_die(stage)
    add_staging_zones(stage, board_size=BOARD_SIZE)

    save_stage(stage, str(SCENE_OUT))
    print(f"Wrote scene to {SCENE_OUT}")
    return SCENE_OUT


if __name__ == "__main__":  # pragma: no cover - hardware-gated
    try:
        build()
    finally:
        sim_app.close()

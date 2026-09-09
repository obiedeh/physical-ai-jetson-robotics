"""Build the Synria-arm + checkers tabletop training scene.

Runs inside Isaac Sim 5.1's Python on the Linux RTX 5090 workstation.
Output: ``isaac/usd/scenes/synria_checkers/synria_checkers_v0.usda``.

Scene layout (meters, Z-up):

- Floor, table, Synria arm (shared scaffolding from ``_synria_scene_common``)
- 8 x 8 board, same geometry as the chess scene (0.045 m squares,
  0.36 m playing area, 0.02 m border, 0.40 m total)
- 24 checker pieces (12 per side, white / black), placed only on the
  dark squares of ranks 1-3 (white) and ranks 6-8 (black) per standard
  American checkers / English draughts opening
- Overhead RGB camera (board-state view) + operator camera (review view)

Each piece is a low cylinder (disc) sitting on its square. No kings on
opening setup; the trainer / game logic crowns them later.

Run on RTX:

    bash scripts/linux_rtx/build_synria_checkers_scene.sh
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
    make_cylinder,
    save_stage,
    sim_app,
)

SCENE_OUT = REPO_ROOT / "isaac" / "usd" / "scenes" / "synria_checkers" / "synria_checkers_v0.usda"

# Board layout — identical to the chess scene
SQUARE_SIZE = 0.045
SQUARES_PER_SIDE = 8
PLAYING_AREA = SQUARE_SIZE * SQUARES_PER_SIDE
BOARD_BORDER = 0.02
BOARD_SIZE = PLAYING_AREA + 2 * BOARD_BORDER
BOARD_THICKNESS = 0.005
BOARD_TOP_Z = TABLE_SURFACE_Z + BOARD_THICKNESS / 2.0
SQUARE_TOP_Z = TABLE_SURFACE_Z + BOARD_THICKNESS + 0.0005

LIGHT_SQUARE = (0.92, 0.86, 0.72)
DARK_SQUARE = (0.45, 0.28, 0.20)
BORDER_COLOR = (0.20, 0.14, 0.10)

# Checker piece dimensions: low, wide disc.
PIECE_RADIUS = 0.018
PIECE_HEIGHT = 0.008
PIECE_BASE_Z = TABLE_SURFACE_Z + BOARD_THICKNESS  # disc rests on board top
PIECE_CENTER_Z = PIECE_BASE_Z + PIECE_HEIGHT / 2.0

WHITE = (0.92, 0.88, 0.78)
BLACK = (0.15, 0.12, 0.10)


def _square_center_xy(file_idx: int, rank_idx: int) -> tuple[float, float]:
    half = PLAYING_AREA / 2.0
    x = BOARD_CENTER_X - half + (file_idx + 0.5) * SQUARE_SIZE
    y = BOARD_CENTER_Y - half + (rank_idx + 0.5) * SQUARE_SIZE
    return x, y


def _is_dark_square(file_idx: int, rank_idx: int) -> bool:
    return (file_idx + rank_idx) % 2 == 0


def _add_checkers_board(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/CheckersBoard")

    # Border
    make_box(
        stage,
        "/World/CheckersBoard/Border",
        size=(BOARD_SIZE, BOARD_SIZE, BOARD_THICKNESS),
        position=(BOARD_CENTER_X, BOARD_CENTER_Y, BOARD_TOP_Z),
        rgb=BORDER_COLOR,
        rigid=False,
    )

    UsdGeom.Xform.Define(stage, "/World/CheckersBoard/Squares")
    for file_idx in range(SQUARES_PER_SIDE):
        for rank_idx in range(SQUARES_PER_SIDE):
            rgb = DARK_SQUARE if _is_dark_square(file_idx, rank_idx) else LIGHT_SQUARE
            x, y = _square_center_xy(file_idx, rank_idx)
            file_letter = "abcdefgh"[file_idx]
            rank_num = rank_idx + 1
            make_box(
                stage,
                f"/World/CheckersBoard/Squares/sq_{file_letter}{rank_num}",
                size=(SQUARE_SIZE, SQUARE_SIZE, 0.001),
                position=(x, y, SQUARE_TOP_Z),
                rgb=rgb,
                rigid=False,
            )


def _add_checkers_pieces(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/CheckersPieces")
    UsdGeom.Xform.Define(stage, "/World/CheckersPieces/white")
    UsdGeom.Xform.Define(stage, "/World/CheckersPieces/black")

    # White: dark squares on ranks 1, 2, 3 (rank_idx 0, 1, 2). 12 pieces total.
    # Black: dark squares on ranks 6, 7, 8 (rank_idx 5, 6, 7). 12 pieces total.
    layouts = [
        ("white", range(0, 3), WHITE),
        ("black", range(5, 8), BLACK),
    ]
    for side, rank_range, rgb in layouts:
        for rank_idx in rank_range:
            for file_idx in range(SQUARES_PER_SIDE):
                if not _is_dark_square(file_idx, rank_idx):
                    continue
                x, y = _square_center_xy(file_idx, rank_idx)
                file_letter = "abcdefgh"[file_idx]
                rank_num = rank_idx + 1
                path = f"/World/CheckersPieces/{side}/piece_{file_letter}{rank_num}"
                UsdGeom.Xform.Define(stage, path)
                make_cylinder(
                    stage,
                    f"{path}/Disc",
                    radius=PIECE_RADIUS,
                    height=PIECE_HEIGHT,
                    position=(x, y, PIECE_CENTER_Z),
                    rgb=rgb,
                )


def build():
    ensure_arm_usd_exists()
    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)

    stage = create_new_stage()
    build_common_scene(stage, board_top_z=BOARD_TOP_Z)
    _add_checkers_board(stage)
    _add_checkers_pieces(stage)
    add_staging_zones(stage, board_size=BOARD_SIZE)

    save_stage(stage, str(SCENE_OUT))
    print(f"Wrote scene to {SCENE_OUT}")
    return SCENE_OUT


if __name__ == "__main__":  # pragma: no cover - hardware-gated
    try:
        build()
    finally:
        sim_app.close()

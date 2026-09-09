"""Build the Synria-arm + chess tabletop training scene.

Runs inside Isaac Sim 5.1's Python on the Linux RTX 5090 workstation.
Output: ``isaac/usd/scenes/synria_chess/synria_chess_v0.usda``.

Scene layout (meters, Z-up):

- Floor, table, Synria arm (shared scaffolding from ``_synria_scene_common``)
- Chess board: 8 x 8 squares, 0.045 m per square -> 0.36 m playing area,
  with a 0.02 m border -> 0.40 m total, centered at (0.10, 0)
- 32 pieces in standard chess opening position, two colors (white / black)
  with distinct shapes per piece type (pawn / rook / knight / bishop / queen / king)
- Overhead RGB camera (board-state view) + operator camera (review view)

The pieces use composite primitives that are visually distinguishable but
deliberately simple placeholders. Final visual fidelity (PBR materials,
proper piece meshes) is a follow-up.

Run on RTX:

    bash scripts/linux_rtx/build_synria_chess_scene.sh
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
    make_cone,
    make_cube,
    make_cylinder,
    make_sphere,
    save_stage,
    sim_app,
)

SCENE_OUT = REPO_ROOT / "isaac" / "usd" / "scenes" / "synria_chess" / "synria_chess_v0.usda"

# Board layout
SQUARE_SIZE = 0.045
SQUARES_PER_SIDE = 8
PLAYING_AREA = SQUARE_SIZE * SQUARES_PER_SIDE   # 0.36 m
BOARD_BORDER = 0.02
BOARD_SIZE = PLAYING_AREA + 2 * BOARD_BORDER     # 0.40 m
BOARD_THICKNESS = 0.005
BOARD_TOP_Z = TABLE_SURFACE_Z + BOARD_THICKNESS / 2.0
SQUARE_TOP_Z = TABLE_SURFACE_Z + BOARD_THICKNESS + 0.0005  # tiny lift to avoid z-fight

LIGHT_SQUARE = (0.92, 0.86, 0.72)
DARK_SQUARE = (0.45, 0.28, 0.20)
BORDER_COLOR = (0.20, 0.14, 0.10)

# Piece colors
WHITE = (0.92, 0.88, 0.78)
BLACK = (0.15, 0.12, 0.10)

# Piece base footprint
PIECE_RADIUS = 0.012
PIECE_BASE_Z = TABLE_SURFACE_Z + BOARD_THICKNESS  # sits on top of the board


def _square_center_xy(file_idx: int, rank_idx: int) -> tuple[float, float]:
    """Return world XY of the center of square (file_idx, rank_idx).

    file_idx 0 = file 'a', 7 = file 'h'.
    rank_idx 0 = rank 1, 7 = rank 8.
    """
    half = PLAYING_AREA / 2.0
    x = BOARD_CENTER_X - half + (file_idx + 0.5) * SQUARE_SIZE
    y = BOARD_CENTER_Y - half + (rank_idx + 0.5) * SQUARE_SIZE
    return x, y


def _add_chess_board(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/ChessBoard")

    # Border
    make_box(
        stage,
        "/World/ChessBoard/Border",
        size=(BOARD_SIZE, BOARD_SIZE, BOARD_THICKNESS),
        position=(BOARD_CENTER_X, BOARD_CENTER_Y, BOARD_TOP_Z),
        rgb=BORDER_COLOR,
        rigid=False,
    )

    # 8 x 8 alternating squares laid on top of the border
    UsdGeom.Xform.Define(stage, "/World/ChessBoard/Squares")
    for file_idx in range(SQUARES_PER_SIDE):
        for rank_idx in range(SQUARES_PER_SIDE):
            is_light = (file_idx + rank_idx) % 2 == 1
            rgb = LIGHT_SQUARE if is_light else DARK_SQUARE
            x, y = _square_center_xy(file_idx, rank_idx)
            file_letter = "abcdefgh"[file_idx]
            rank_num = rank_idx + 1
            make_box(
                stage,
                f"/World/ChessBoard/Squares/sq_{file_letter}{rank_num}",
                size=(SQUARE_SIZE, SQUARE_SIZE, 0.001),
                position=(x, y, SQUARE_TOP_Z),
                rgb=rgb,
                rigid=False,
            )


def _piece_path(side: str, piece_type: str, file_letter: str, rank_num: int) -> str:
    return f"/World/ChessPieces/{side}/{piece_type}_{file_letter}{rank_num}"


def _add_pawn(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    # Base cylinder + small sphere on top
    base_h = 0.022
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    top_z = PIECE_BASE_Z + base_h + 0.006
    make_sphere(stage, f"{path}/Top", radius=0.006, position=(x, y, top_z), rgb=rgb)


def _add_rook(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    # Base cylinder + small cube top (battlements proxy)
    base_h = 0.028
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    top_size = 0.018
    top_z = PIECE_BASE_Z + base_h + top_size / 2.0
    make_cube(stage, f"{path}/Top", size=top_size, position=(x, y, top_z), rgb=rgb)


def _add_knight(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    # Base cylinder + a small angled box (proxy for horse head)
    base_h = 0.026
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    head_z = PIECE_BASE_Z + base_h + 0.008
    make_box(
        stage,
        f"{path}/Head",
        size=(0.020, 0.014, 0.018),
        position=(x, y, head_z),
        rgb=rgb,
        rigid=True,
    )


def _add_bishop(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    # Base cylinder + cone + tiny sphere tip
    base_h = 0.024
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    cone_h = 0.016
    cone_z = PIECE_BASE_Z + base_h + cone_h / 2.0
    make_cone(stage, f"{path}/Cap", radius=PIECE_RADIUS * 0.9, height=cone_h,
              position=(x, y, cone_z), rgb=rgb)
    tip_z = PIECE_BASE_Z + base_h + cone_h + 0.004
    make_sphere(stage, f"{path}/Tip", radius=0.004, position=(x, y, tip_z), rgb=rgb)


def _add_queen(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    base_h = 0.030
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    crown_z = PIECE_BASE_Z + base_h + 0.008
    make_sphere(stage, f"{path}/Crown", radius=0.010, position=(x, y, crown_z), rgb=rgb)
    spike_h = 0.010
    spike_z = crown_z + 0.010 + spike_h / 2.0
    make_cylinder(stage, f"{path}/Spike", radius=0.003, height=spike_h,
                  position=(x, y, spike_z), rgb=rgb)


def _add_king(stage, path: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    base_h = 0.034
    base_z = PIECE_BASE_Z + base_h / 2.0
    make_cylinder(stage, f"{path}/Base", radius=PIECE_RADIUS, height=base_h,
                  position=(x, y, base_z), rgb=rgb)
    crown_z = PIECE_BASE_Z + base_h + 0.008
    make_sphere(stage, f"{path}/Crown", radius=0.011, position=(x, y, crown_z), rgb=rgb)
    # Cross: vertical box + horizontal box
    cross_center_z = crown_z + 0.011 + 0.006
    make_box(stage, f"{path}/CrossV", size=(0.003, 0.003, 0.012),
             position=(x, y, cross_center_z), rgb=rgb)
    make_box(stage, f"{path}/CrossH", size=(0.010, 0.003, 0.003),
             position=(x, y, cross_center_z), rgb=rgb)


_PIECE_BUILDERS = {
    "pawn": _add_pawn,
    "rook": _add_rook,
    "knight": _add_knight,
    "bishop": _add_bishop,
    "queen": _add_queen,
    "king": _add_king,
}

# Standard chess opening rank layout (file a-h)
_BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]


def _add_chess_pieces(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/ChessPieces")
    UsdGeom.Xform.Define(stage, "/World/ChessPieces/white")
    UsdGeom.Xform.Define(stage, "/World/ChessPieces/black")

    # White on ranks 1 (back) and 2 (pawns); black on ranks 7 (pawns) and 8 (back).
    layout = [
        ("white", 0, _BACK_RANK),
        ("white", 1, ["pawn"] * 8),
        ("black", 6, ["pawn"] * 8),
        ("black", 7, _BACK_RANK),
    ]

    for side, rank_idx, pieces in layout:
        rgb = WHITE if side == "white" else BLACK
        rank_num = rank_idx + 1
        for file_idx, piece_type in enumerate(pieces):
            x, y = _square_center_xy(file_idx, rank_idx)
            file_letter = "abcdefgh"[file_idx]
            path = _piece_path(side, piece_type, file_letter, rank_num)
            UsdGeom.Xform.Define(stage, path)
            _PIECE_BUILDERS[piece_type](stage, path, x, y, rgb)


def build():
    ensure_arm_usd_exists()
    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)

    stage = create_new_stage()
    build_common_scene(stage, board_top_z=BOARD_TOP_Z)
    _add_chess_board(stage)
    _add_chess_pieces(stage)
    add_staging_zones(stage, board_size=BOARD_SIZE)

    save_stage(stage, str(SCENE_OUT))
    import sys as _sys
    print(f"Wrote scene to {SCENE_OUT}", flush=True)
    _sys.stdout.flush()
    return SCENE_OUT


if __name__ == "__main__":  # pragma: no cover - hardware-gated
    import sys as _sys
    import traceback as _tb
    try:
        build()
    except Exception as exc:
        print(f"BUILD FAILED: {exc}", flush=True)
        _tb.print_exc()
        _sys.exit(1)
    finally:
        sim_app.close()

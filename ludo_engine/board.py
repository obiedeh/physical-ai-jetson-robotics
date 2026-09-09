"""Board geometry: logical Ludo positions -> world XY on the sim board.

The one place engine-land meets robot-land. Mirrors the layout of
isaac/scripts/build_synria_ludo_scene.py: 0.40 m board centered at
(0.10, 0), Z-up. The controller receives (world_xy_from, world_xy_to)
per move and never sees game state; the engine never sees meters except
through this adapter.

MVP mapping: the 52-square track is laid out on the standard Ludo cross
(15x15 grid, cell = board_size/15). Yard slots sit in the four corner
quadrants; home columns run from each color's side toward the centre.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import START, TRACK_LEN

# standard 15x15 Ludo grid coordinates for the 52-square track, starting
# at red's start square (grid col,row), proceeding clockwise. Grid origin
# = board's lower-left cell centre.
_TRACK_GRID = [
    (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
    (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0),
    (7, 0), (8, 0),
    (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
    (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6),
    (14, 7), (14, 8),
    (13, 8), (12, 8), (11, 8), (10, 8), (9, 8),
    (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
    (7, 14), (6, 14),
    (6, 13), (6, 12), (6, 11), (6, 10), (6, 9),
    (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    (0, 7), (0, 6),
]
_YARD_GRID = {
    "red": [(2, 2), (2, 3), (3, 2), (3, 3)],
    "blue": [(2, 11), (2, 12), (3, 11), (3, 12)],
    "green": [(11, 11), (11, 12), (12, 11), (12, 12)],
    "yellow": [(11, 2), (11, 3), (12, 2), (12, 3)],
}
_HOME_GRID = {
    "red": [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)],
    "blue": [(7, 13), (7, 12), (7, 11), (7, 10), (7, 9), (7, 8)],
    "green": [(13, 7), (12, 7), (11, 7), (10, 7), (9, 7), (8, 7)],
    "yellow": [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)],
}


@dataclass
class BoardGeometry:
    board_size: float = 0.40
    center_x: float = 0.10
    center_y: float = 0.0

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        cell = self.board_size / 15.0
        x = self.center_x + (row - 7) * cell
        y = self.center_y + (col - 7) * cell
        return (round(x, 4), round(y, 4))

    def world_xy(self, color: str, pos: tuple) -> tuple[float, float]:
        kind = pos[0]
        if kind == "yard":
            c, r = _YARD_GRID[color][pos[1]]
        elif kind == "track":
            c, r = _TRACK_GRID[pos[1] % TRACK_LEN]
        elif kind in ("home", "done"):
            idx = min(pos[1], 5) if kind == "home" else 5
            c, r = _HOME_GRID[color][idx]
        else:
            raise ValueError(f"unknown position kind {pos}")
        return self._grid_to_world(c, r)

    def move_to_command(self, move) -> dict:
        """A Move -> the controller-facing command dict."""
        return {
            "color": move.color,
            "token": move.token,
            "pick_xy": self.world_xy(move.color, move.src),
            "place_xy": self.world_xy(move.color, move.dst),
            "captures": [
                {"color": c, "token": t,
                 "pick_xy": self.world_xy(c, ("track", move.dst[1]))
                 if move.dst[0] == "track" else None}
                for (c, t) in move.captures
            ],
        }

"""Ludo board states, overhead-camera geometry and scoring for the board-reading benchmark.

Pure Python: no simulator and no network. The renderer
(`isaac/scripts/render_ludo_board_bench.py`) and the scorer (`board_bench.py`)
both import the geometry from here so they cannot drift apart.

A state is scored on occupied cells: the set of (color, col, row) on the
15x15 grid of `ludo_engine/board.py`. Tokens sharing a cell are stacked and
cannot be counted from above, so the count per cell is not scored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ludo_engine.board import _HOME_GRID, _TRACK_GRID, _YARD_GRID, BoardGeometry
from ludo_engine.game import COLORS, TRACK_LEN, GamePhase, LudoGame

GRID = 15
TOKEN_HEIGHT = 0.020


def cell_of(color: str, pos: tuple) -> tuple[int, int]:
    """Grid (col, row) of a logical position, same mapping as BoardGeometry.world_xy."""
    kind = pos[0]
    if kind == "yard":
        return _YARD_GRID[color][pos[1]]
    if kind == "track":
        return _TRACK_GRID[pos[1] % TRACK_LEN]
    if kind in ("home", "done"):
        return _HOME_GRID[color][min(pos[1], 5) if kind == "home" else 5]
    raise ValueError(f"unknown position kind {pos}")


def sample_states(n: int, seed: int, min_turns: int = 20, max_turns: int = 320) -> list[dict]:
    """Play `n` random legal games and snapshot each at a random turn count."""
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        game = LudoGame()
        turns = rng.randint(min_turns, max_turns)
        for _ in range(turns):
            if game.phase is GamePhase.FINISHED:
                break
            game.take_turn(rng.randint(1, 6), chooser=rng.choice)
        tokens, level = [], {}
        for (color, idx), pos in sorted(game.positions.items()):
            col, row = cell_of(color, pos)
            stack = level.get((col, row), 0)
            level[(col, row)] = stack + 1
            tokens.append({"color": color, "token": idx, "pos": [pos[0], pos[1]],
                           "cell": [col, row], "stack": stack})
        states.append({"i": len(states), "turns": turns, "tokens": tokens})
    return states


def occupied(tokens: list[dict]) -> set[tuple[str, int, int]]:
    return {(t["color"], t["cell"][0], t["cell"][1]) for t in tokens}


@dataclass(frozen=True)
class OverheadCamera:
    """Pinhole camera straight above the board centre, image right = world +X,
    image up = world +Y. `distance` is camera height above the board top."""

    image_wh: tuple[int, int]
    focal_mm: float = 36.0
    aperture_mm: float = 20.955
    distance: float = 0.90
    geometry: BoardGeometry = field(default_factory=BoardGeometry)

    def _scale(self, height_above_board: float) -> float:
        return (self.focal_mm / self.aperture_mm) * self.image_wh[0] / (self.distance - height_above_board)

    def world_to_px(self, x: float, y: float, height_above_board: float = 0.0) -> tuple[float, float]:
        k = self._scale(height_above_board)
        w, h = self.image_wh
        return (w / 2 + (x - self.geometry.center_x) * k, h / 2 - (y - self.geometry.center_y) * k)

    def px_to_cell(self, px: float, py: float, height_above_board: float = TOKEN_HEIGHT) -> tuple[int, int] | None:
        """Grid (col, row) under a pixel, read on the plane of a token top. None if off the board."""
        k = self._scale(height_above_board)
        w, h = self.image_wh
        x = self.geometry.center_x + (px - w / 2) / k
        y = self.geometry.center_y - (py - h / 2) / k
        cell = self.geometry.board_size / GRID
        row = round((x - self.geometry.center_x) / cell) + 7
        col = round((y - self.geometry.center_y) / cell) + 7
        if 0 <= col < GRID and 0 <= row < GRID:
            return (col, row)
        return None

    def cell_to_px(self, col: int, row: int, height_above_board: float = TOKEN_HEIGHT) -> tuple[float, float]:
        cell = self.geometry.board_size / GRID
        return self.world_to_px(self.geometry.center_x + (row - 7) * cell,
                                self.geometry.center_y + (col - 7) * cell, height_above_board)


def score_state(truth: set, pred: set) -> dict:
    """Cell-level comparison of two sets of (color, col, row)."""
    hit = truth & pred
    truth_xy = {(c, r) for _, c, r in truth}
    return {
        "n_truth": len(truth), "n_pred": len(pred), "n_hit": len(hit),
        "missed": sorted(truth - pred), "extra": sorted(pred - truth),
        "wrong_color": sum((c, r) in truth_xy for _, c, r in pred - truth),
        "exact": truth == pred,
    }


__all__ = ["COLORS", "GRID", "TOKEN_HEIGHT", "OverheadCamera", "cell_of", "occupied",
           "sample_states", "score_state"]

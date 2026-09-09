"""Ludo brain -> arm commands. Wraps the tested ludo_engine rules core.

Physical mapping notes:
- a capture means the arm FIRST returns the opponent token to its yard
  (clear the square), then moves the attacking token in.
- dice come from outside (real die on the table read by perception, or
  a software RNG in pure-sim mode).
"""

from __future__ import annotations

import random

from ludo_engine.board import BoardGeometry
from ludo_engine.game import GamePhase, LudoGame

from .commands import PickPlaceCommand, TurnPlan


class LudoGameAdapter:
    def __init__(self, board_size: float = 0.40, center=(0.10, 0.0),
                 seed: int = 0):
        self.game = LudoGame()
        self.geom = BoardGeometry(board_size, center[0], center[1])
        self.rng = random.Random(seed)

    def plan_turn(self, roll: int | None = None) -> TurnPlan | None:
        """Advance one turn. Returns the physical plan, or None when the
        roll produced no legal move (turn still consumed per rules)."""
        color = self.game.current
        roll = roll if roll is not None else self.rng.randint(1, 6)
        move = self.game.take_turn(roll)
        if move is None:
            return None
        plan = TurnPlan(
            game="ludo",
            description=f"{color} rolls {roll}: token {move.token} "
                        f"{move.src} -> {move.dst}"
                        + (f", captures {list(move.captures)}"
                           if move.captures else ""),
        )
        for (c, t) in move.captures:
            # captured token goes home: pick at the contested square,
            # place at its yard slot (rules core already re-assigned it)
            yard_pos = self.game.positions[(c, t)]
            plan.commands.append(PickPlaceCommand(
                self.geom.world_xy(c, move.dst),
                self.geom.world_xy(c, yard_pos),
                piece=f"{c} token {t}", reason="capture-return"))
        plan.commands.append(PickPlaceCommand(
            self.geom.world_xy(color, move.src),
            self.geom.world_xy(color, move.dst),
            piece=f"{color} token {move.token}", reason="move"))
        return plan

    @property
    def game_over(self) -> bool:
        return self.game.phase == GamePhase.FINISHED

    def result(self) -> str:
        return " > ".join(self.game.finished_order)

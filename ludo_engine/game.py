"""Ludo rules core.

Standard 4-color rules, MVP-configurable:
- 4 tokens per color; tokens start in the home yard.
- Roll a 6 to leave the yard (enters the color's start square).
- Main track: 52 squares, clockwise, shared by all colors.
- Landing on an opponent's single token captures it (back to its yard),
  except on safe squares (each color's start square + the 4 star squares).
- After 51 track squares from its start, a token turns into its color's
  home column (6 cells); an EXACT roll is required to reach the final
  home cell.
- Rolling a 6 grants another turn; three consecutive 6s forfeit the turn.

Positions are encoded as:
    ("yard", i)          in the home yard, slot i in [0,4)
    ("track", s)         absolute track square s in [0,52)
    ("home", h)          home-column cell h in [0,6), 5 = finished slot
    ("done", i)          finished, order i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

COLORS = ("red", "blue", "green", "yellow")
TRACK_LEN = 52
HOME_LEN = 6
TOKENS = 4
# absolute start square per color (standard quarter offsets)
START = {"red": 0, "blue": 13, "green": 26, "yellow": 39}
# star squares (safe), absolute — standard board: 8 past each start
STARS = {8, 21, 34, 47}
SAFE = set(START.values()) | STARS


class GamePhase(Enum):
    ROLLING = "rolling"
    MOVING = "moving"
    FINISHED = "finished"


@dataclass(frozen=True)
class Move:
    color: str
    token: int
    src: tuple
    dst: tuple
    captures: tuple = ()   # ((color, token), ...) sent back to yard


@dataclass
class LudoGame:
    players: tuple = COLORS
    positions: dict = field(default_factory=dict)   # (color, token) -> pos
    turn_idx: int = 0
    phase: GamePhase = GamePhase.ROLLING
    consecutive_sixes: int = 0
    last_roll: int | None = None
    finished_order: list = field(default_factory=list)

    def __post_init__(self):
        if not self.positions:
            for c in self.players:
                for t in range(TOKENS):
                    self.positions[(c, t)] = ("yard", t)

    # ---- helpers -------------------------------------------------------
    @property
    def current(self) -> str:
        return self.players[self.turn_idx]

    def _track_progress(self, color: str, square: int) -> int:
        """How many squares a token on absolute `square` has travelled
        from its color's start (0..51)."""
        return (square - START[color]) % TRACK_LEN

    def _occupants(self, pos: tuple):
        return [k for k, v in self.positions.items() if v == pos]

    # ---- rules ---------------------------------------------------------
    def legal_moves(self, roll: int) -> list[Move]:
        color = self.current
        moves = []
        for t in range(TOKENS):
            src = self.positions[(color, t)]
            kind = src[0]
            if kind == "done":
                continue
            if kind == "yard":
                if roll == 6:
                    dst = ("track", START[color])
                    moves.append(self._mk(color, t, src, dst))
                continue
            if kind == "track":
                prog = self._track_progress(color, src[1])
                np_ = prog + roll
                if np_ <= 50:
                    dst = ("track", (src[1] + roll) % TRACK_LEN)
                    moves.append(self._mk(color, t, src, dst))
                elif np_ <= 50 + HOME_LEN:
                    h = np_ - 51
                    if h < HOME_LEN:
                        moves.append(self._mk(color, t, src, ("home", h)))
                continue
            if kind == "home":
                h = src[1] + roll
                if h < HOME_LEN:
                    moves.append(self._mk(color, t, src, ("home", h)))
        return moves

    def _mk(self, color, t, src, dst) -> Move:
        captures = []
        if dst[0] == "track" and dst[1] not in SAFE:
            occ = self._occupants(dst)
            enemies = [k for k in occ if k[0] != color]
            # single enemy token is captured; a blockade (2+) would make
            # the square unenterable in stricter rules — MVP: capture all
            for e in enemies:
                captures.append(e)
        return Move(color, t, src, dst, tuple(captures))

    def apply(self, move: Move) -> None:
        assert move.color == self.current, "not this player's turn"
        for e in move.captures:
            # return to lowest free yard slot
            used = {v[1] for k, v in self.positions.items()
                    if k[0] == e[0] and v[0] == "yard"}
            slot = min(set(range(TOKENS)) - used)
            self.positions[e] = ("yard", slot)
        dst = move.dst
        if dst == ("home", HOME_LEN - 1):
            dst = ("done", len([1 for v in self.positions.values()
                                if v[0] == "done"]))
        self.positions[(move.color, move.token)] = dst
        if all(self.positions[(move.color, t)][0] == "done"
               for t in range(TOKENS)):
            if move.color not in self.finished_order:
                self.finished_order.append(move.color)

    def take_turn(self, roll: int, chooser=None) -> Move | None:
        """Roll + choose + apply. `chooser(moves) -> Move` picks among
        legal moves (default: first capture, else furthest-along token).
        Returns the applied move, or None if no legal move existed.
        Advances the turn per the 6-rule."""
        assert 1 <= roll <= 6
        self.last_roll = roll
        if roll == 6:
            self.consecutive_sixes += 1
        else:
            self.consecutive_sixes = 0
        moved = None
        if self.consecutive_sixes < 3:
            moves = self.legal_moves(roll)
            if moves:
                moved = (chooser or self._default_choice)(moves)
                self.apply(moved)
        if len(self.finished_order) >= len(self.players) - 1:
            self.phase = GamePhase.FINISHED
        elif roll != 6 or self.consecutive_sixes >= 3:
            if self.consecutive_sixes >= 3:
                self.consecutive_sixes = 0
            self.turn_idx = (self.turn_idx + 1) % len(self.players)
        return moved

    def _default_choice(self, moves: list[Move]) -> Move:
        withcap = [m for m in moves if m.captures]
        if withcap:
            return withcap[0]

        def prog(m: Move) -> int:
            if m.src[0] == "track":
                return self._track_progress(m.color, m.src[1])
            if m.src[0] == "home":
                return 51 + m.src[1]
            return -1
        return max(moves, key=prog)

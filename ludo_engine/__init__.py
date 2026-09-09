"""Ludo game engine — L4 task layer for the Synria arm product track.

Pure Python, zero robot/sim dependencies by design (layer independence):
the engine consumes dice rolls and emits *moves* — (color, token_id,
from_pos, to_pos) — and a separate adapter maps board positions to world
coordinates for whichever controller executes them (scripted expert,
ACT, or GR00T). The engine never knows an arm exists.
"""

from .game import LudoGame, Move, GamePhase  # noqa: F401
from .board import BoardGeometry  # noqa: F401
